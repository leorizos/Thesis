from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisagreementScaler:
    """Tracks per-term Pearson disagreement scale for power-law normalization.

    Per batch: tracks the epoch's running max in a scratch buffer.
    Per epoch: applies EMA of (momentum * d_max + (1-momentum) * epoch_max),
               then resets the scratch buffer.
    After lock_epoch: d_max is frozen, no further updates.

    This avoids a single outlier batch locking d_max permanently while still
    providing a stable, slowly-adapting normalizer across training.
    """
    def __init__(self, init=0.1, momentum=0.99, lock_epoch=180):
        self.d_max      = {'L1': init, 'L2': init, 'L3': init}
        self.d_mean     = {'L1': init, 'L2': init, 'L3': init}
        self.momentum   = momentum
        self.lock_epoch = lock_epoch
        self._epoch_max = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}
        self._epoch_sum = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}
        self._epoch_n   = {'L1': 0,    'L2': 0,    'L3': 0}
        self.d_std      = {'L1': init, 'L2': init, 'L3': init}
        self._epoch_sq  = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}

    def update(self, d_L1, d_L2, d_L3):
        """Track per-batch max into epoch scratch buffer. Call once per batch."""
        for key, d in [('L1', d_L1), ('L2', d_L2), ('L3', d_L3)]:
            batch_max = d.max().item()
            if batch_max > self._epoch_max[key]:
                self._epoch_max[key] = batch_max
            self._epoch_sum[key] += d.sum().item()
            self._epoch_n[key]   += d.numel()
            self._epoch_sq[key]  += (d ** 2).sum().item()

    def step_epoch(self, epoch):
        """Apply EMA update from this epoch's max, then reset scratch. Call once per epoch.

        Epoch 1: hard update to set a real baseline (avoids init=0.1 saturation).
        Epochs 2-lock_epoch: EMA smoothing.
        After lock_epoch: d_max frozen.
        """
        if epoch <= self.lock_epoch:
            if epoch == 1:
                # Hard update: set d_max directly from first epoch's observed max
                for key in self.d_max:
                    self.d_max[key] = self._epoch_max[key]
                    self.d_mean[key] = self._epoch_sum[key] / self._epoch_n[key]
                    self.d_std[key] = (self._epoch_sq[key] / self._epoch_n[key] - self.d_mean[key] ** 2) ** 0.5
            else:
                m = self.momentum
                for key in self.d_max:
                    self.d_max[key] = m * self.d_max[key] + (1 - m) * self._epoch_max[key]
                    epoch_mean = self._epoch_sum[key] / self._epoch_n[key]
                    self.d_mean[key] = m * self.d_mean[key] + (1 - m) * epoch_mean
                    epoch_std = (self._epoch_sq[key] / self._epoch_n[key] - epoch_mean ** 2) ** 0.5
                    self.d_std[key] = m * self.d_std[key] + (1 - m) * epoch_std
        self._epoch_max = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}
        self._epoch_sum = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}
        self._epoch_n   = {'L1': 0,   'L2': 0,   'L3': 0}
        self._epoch_sq  = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}


def compute_lambda_power(disagreement, d_max, alpha, lambda_max):
    """Per-sample sigma weight via power function on running-max-normalised disagreement.

    Higher disagreement → higher lambda, continuously and without a fixed midpoint.

        d_norm = clamp(disagreement / d_max, max=1.0)
        lambda = d_norm ^ alpha * lambda_max   ∈ [0, lambda_max]

    alpha < 1 → concave curve, assigns more weight even at moderate disagreement.
    alpha = 1 → linear mapping.
    alpha > 1 → convex curve, concentrates weight on high-disagreement samples.

    Args:
        disagreement: [N] Pearson distances (unnormalized)
        d_max:        running maximum for this term (scalar float)
        alpha:        exponent controlling curve shape
        lambda_max:   ceiling for sigma weight
    Returns:
        [N] lambda values in [0, lambda_max]
    """
    d_norm = (disagreement / d_max).clamp(max=1.0)
    return d_norm.pow(alpha) * lambda_max


class GCN(nn.Module):
    """Learnable confusion matrix for adaptive sigma softening."""
    def __init__(self, num_classes):
        super(GCN, self).__init__()
        identity = torch.eye(num_classes)
        noise = torch.randn(num_classes, num_classes) * 1e-4
        self.M = nn.Parameter(identity + noise)
    
    def forward(self, sigma_matrix):
        return sigma_matrix @ self.M


def soften_sigma_with_gcn(sigma, target_net, output_net, anchor_target, anchor_net,
                          labels, anchor_labels, gcn, alpha=0.1, eps=1e-7,
                          sigma_s_mode='ab'):
    """
    Adaptively soften precomputed sigma using a learnable GCN.

    Implements ASL-style dual optimization:
    1. Σ_t_hat = α * GCN(sigma) + (1-α) * sigma
    2. loss_G = MSE(Σ_t_hat, Σ_s) where Σ_s = student's class-level patterns

    sigma_s_mode controls how Σ_s is computed:
        'ab' (default) — merged batch+anchor: sim(batch+anchor, batch+anchor) [B+A, B+A] → [C, C], 100% class coverage
        'aa'           — anchor-to-anchor: sim(anchor_i, anchor_j)           [A, A] → [C, C]
        'bb'           — batch-to-batch:   sim(batch_class_i, batch_class_j) [B, B] → [C, C]
    """
    num_classes = sigma.size(0)
    device = sigma.device
    C = num_classes

    output_net = F.normalize(output_net, p=2, dim=1)   # [B, D]
    anchor_net = F.normalize(anchor_net, p=2, dim=1)   # [A, D]

    if sigma_s_mode == 'ab':
        # Merged batch+anchor features [B+A, D], 100% class coverage
        merged = torch.cat([output_net, anchor_net], dim=0)          # [B+A, D]
        merged_labels = torch.cat([labels, anchor_labels], dim=0)    # [B+A]
        sim = (torch.mm(merged, merged.t()) + 1) / 2                 # [B+A, B+A]
        row_labels = merged_labels
        col_labels = merged_labels

    elif sigma_s_mode == 'aa':
        # Similarity between anchors and anchors [A, A]
        sim = (torch.mm(anchor_net, anchor_net.t()) + 1) / 2
        row_labels = anchor_labels   # [A]
        col_labels = anchor_labels   # [A]

    elif sigma_s_mode == 'bb':
        # Similarity between batch samples and batch samples [B, B]
        sim = (torch.mm(output_net, output_net.t()) + 1) / 2
        row_labels = labels          # [B]
        col_labels = labels          # [B]

    else:
        raise ValueError(f"Unknown sigma_s_mode '{sigma_s_mode}'. Choose from 'ab', 'aa', 'bb'.")

    # Aggregate to Σ_s [C, C] — vectorized with scatter_add_
    flat_idx = row_labels.unsqueeze(1) * C + col_labels.unsqueeze(0)  # [N_row, N_col]

    Sigma_s_flat = torch.zeros(C * C, device=device)
    Sigma_s_flat.scatter_add_(0, flat_idx.reshape(-1), sim.detach().reshape(-1))

    counts_flat = torch.zeros(C * C, device=device)
    counts_flat.scatter_add_(0, flat_idx.reshape(-1), torch.ones_like(sim).reshape(-1))

    Sigma_s = Sigma_s_flat.reshape(C, C)
    counts  = counts_flat.reshape(C, C)

    mask = (counts > 0).float()
    Sigma_s = Sigma_s / (counts + eps)

    Sigma_g = gcn(sigma)

    # ASL equation 1: Blend
    Sigma_t_hat = alpha * Sigma_g + (1 - alpha) * sigma

    # ASL equation 2: Train GCN
    loss_G = F.mse_loss(Sigma_t_hat * mask, Sigma_s * mask)

    return Sigma_t_hat.detach(), loss_G

def pearson_distance_rows(P, Q, eps=1e-7):
    """
    Pearson distance between corresponding rows of P and Q.
    Measures how differently P and Q distribute probability mass,
    independent of mean offset.

    Args:
        P, Q: [N, D] tensors (row distributions)
    Returns:
        distance: [N] tensor in [0, 2]
    """
    P_c = P - P.mean(dim=1, keepdim=True)
    Q_c = Q - Q.mean(dim=1, keepdim=True)
    return 1.0 - F.cosine_similarity(P_c, Q_c, dim=1, eps=eps)  # [N]


def adaptive_lambda(disagreement, lambda_max, k=10.0, d0=0.3, eps=1e-7, return_stats=False, d0_override=None, k_override=None):
    """
    Per-sample sigma weight via sigmoid gate on normalised Pearson disagreement.

    When teacher and student agree (low disagreement) → sigma weight ≈ 0.0 (pure teacher).
    When they disagree strongly → sigma weight → lambda_max.

        gate   = sigmoid(k * (d_norm - d0))
        lambda = gate * lambda_max   ∈ [0, lambda_max]

    Args:
        disagreement: [N] Pearson distances (unnormalized)
        lambda_max:   scalar ceiling for sigma weight (e.g. 0.3 → max 30% sigma)
        k:            sigmoid steepness (default 10.0)
        d0:           sigmoid midpoint on normalized scale (default 0.3)
        return_stats: if True, also return (raw_dis, gate) before scaling
    Returns:
        [N] per-sample sigma weight values (and optionally raw_dis, gate)
    """
    raw_dis = disagreement.detach()
    if d0_override is not None:
        midpoint = d0_override
    elif d0 == 'ma':
        midpoint = disagreement.mean()
    else:
        disagreement = disagreement / (disagreement.max() + eps)
        midpoint = d0
    k_eff = k_override if k_override is not None else k
    gate = torch.sigmoid(k_eff * (disagreement - midpoint))
    if return_stats:
        return gate * lambda_max, raw_dis, gate.detach()
    return gate * lambda_max  # [N]


def cell_wise_lambda(T_sim, S_sim, lambda_max, k=10.0, d0=0.3):
    """
    Cell-wise sigma weight matrix via sigmoid gate on absolute disagreement.

    When teacher and student agree on a cell → sigma weight ≈ 0.0 (pure teacher).
    When they disagree strongly on a cell → sigma weight → lambda_max.

        disagreement = |T_sim - S_sim|             (already in [0, 1])
        gate         = sigmoid(k * (disagreement - d0))
        lambda       = gate * lambda_max            ∈ [0, lambda_max]

    Args:
        T_sim, S_sim: tensors of same shape, values in [0, 1]
        lambda_max:   scalar ceiling for sigma weight (e.g. 0.3 → max 30% sigma)
        k:            sigmoid steepness (default 10.0)
        d0:           sigmoid midpoint (default 0.3)
    Returns:
        sigma weight matrix of same shape as inputs
    """
    disagreement = (T_sim - S_sim).abs()
    if d0 == 'ma':
        midpoint = disagreement.mean()
    else:
        midpoint = d0
    gate = torch.sigmoid(k * (disagreement - midpoint))
    return gate * lambda_max


def soft_akd_loss(target_net, output_net, anchor_target, anchor_net,
                  labels, anchor_labels, sigma, lambda_soft, opt, eps=1e-7,
                  return_stats=False, scaler=None):
    """
    Soft Anchor-based Knowledge Distillation loss.

    Extends AKD by blending each of the three teacher similarity matrices
    with a precomputed class-level similarity matrix sigma, injecting global
    semantic structure into all three KL divergence terms.

        sigma_sharp = sigma_subset ^ (1 / sigma_temp)   (temperature sharpening)
        S_teacher_soft = (1 - lambda) * S_teacher + lambda * sigma_sharp

    Args:
        target_net:    teacher pooled features for the batch          [B, D]
        output_net:    student pooled features for the batch          [B, D]
        anchor_target: teacher pooled features for anchor images      [A, D]
        anchor_net:    student pooled features for anchor images      [A, D]
        labels:        batch class labels                             [B]
        anchor_labels: anchor class labels                            [A]
        sigma:         precomputed class similarity matrix             [C, C]
        lambda_soft:   max sigma weight (0.0 = pure teacher, 1.0 = pure sigma)
        opt:           options containing l_1, l_2, and sigma_temp hyperparameters
        eps:           small constant for numerical stability
    """
    # Normalize each vector
    anchor_target = F.normalize(anchor_target, p=2, dim=1)
    output_net = F.normalize(output_net, p=2, dim=1)
    target_net = F.normalize(target_net, p=2, dim=1)
    anchor_net = F.normalize(anchor_net, p=2, dim=1)

    # Cosine similarity scaled to [0, 1]
    a_student_sim = (torch.mm(output_net, torch.t(anchor_net)) + 1) / 2        # [B, A]
    a_teacher_sim = (torch.mm(target_net, torch.t(anchor_target)) + 1) / 2     # [B, A]
    a_teacher_sim_t, a_student_sim_t = torch.t(a_teacher_sim), torch.t(a_student_sim)  # [A, B]

    b_student_sim = (torch.mm(output_net, torch.t(output_net)) + 1) / 2        # [B, B]
    b_teacher_sim = (torch.mm(target_net, torch.t(target_net)) + 1) / 2        # [B, B]

    # ===== SOFTENING: Blend teacher similarities with class-level sigma =====
    sigma_bb = sigma[labels][:, labels]              # [B, B]
    sigma_ba = sigma[labels][:, anchor_labels]       # [B, A]
    sigma_ab = sigma[anchor_labels][:, labels]       # [A, B]

    # Temperature sharpening: sigma ^ (1/T), T < 1 sharpens contrast
    sigma_temp = getattr(opt, 'sigma_temp', 1.0)
    if sigma_temp != 1.0:
        exponent = 1.0 / sigma_temp
        sigma_bb = sigma_bb ** exponent
        sigma_ba = sigma_ba ** exponent
        sigma_ab = sigma_ab ** exponent

    lambda_k    = getattr(opt, 'lambda_k',    10.0)
    lambda_d0   = getattr(opt, 'lambda_d0',   0.3)
    lambda_mode = getattr(opt, 'lambda_mode', 'rw')

    stats = None
    if lambda_mode == 'rw':
        d_L1 = pearson_distance_rows(b_teacher_sim, b_student_sim)      # [B]
        d_L2 = pearson_distance_rows(a_teacher_sim, a_student_sim)      # [B]
        d_L3 = pearson_distance_rows(a_teacher_sim_t, a_student_sim_t)  # [A]

        if scaler is not None:
            # Power-function path: running-max normalization
            scaler.update(d_L1.detach(), d_L2.detach(), d_L3.detach())
            lambda_alpha = getattr(opt, 'lambda_alpha', 1.0)
            lam_L1_flat = compute_lambda_power(d_L1, scaler.d_max['L1'], lambda_alpha, lambda_soft)  # [B]
            lam_L2_flat = compute_lambda_power(d_L2, scaler.d_max['L2'], lambda_alpha, lambda_soft)  # [B]
            lam_L3_flat = compute_lambda_power(d_L3, scaler.d_max['L3'], lambda_alpha, lambda_soft)  # [A]
            if return_stats:
                stats = {'raw_L1': d_L1.detach(), 'raw_L2': d_L2.detach(), 'raw_L3': d_L3.detach(),
                         'lam_L1': lam_L1_flat.detach(), 'lam_L2': lam_L2_flat.detach(), 'lam_L3': lam_L3_flat.detach()}
            lam_L1 = lam_L1_flat.unsqueeze(1)  # [B, 1]
            lam_L2 = lam_L2_flat.unsqueeze(1)  # [B, 1]
            lam_L3 = lam_L3_flat.unsqueeze(1)  # [A, 1]
        elif return_stats:
            # Sigmoid path with stats collection
            k_mode = getattr(opt, 'lambda_k_mode', 'static')
            k_scale = getattr(opt, 'lambda_k_scale', 1.0)
            d0s = {k: scaler.d_mean[k] for k in ('L1', 'L2', 'L3')} if scaler is not None else {}
            k0s = {k: k_scale / (scaler.d_std[k] + 1e-7) for k in ('L1', 'L2', 'L3')} if (scaler is not None and k_mode == 'auto') else {}
            lam_L1, raw_L1, gate_L1 = adaptive_lambda(d_L1, lambda_soft, lambda_k, lambda_d0, return_stats=True, d0_override=d0s.get('L1'), k_override=k0s.get('L1'))
            lam_L2, raw_L2, gate_L2 = adaptive_lambda(d_L2, lambda_soft, lambda_k, lambda_d0, return_stats=True, d0_override=d0s.get('L2'), k_override=k0s.get('L2'))
            lam_L3, raw_L3, gate_L3 = adaptive_lambda(d_L3, lambda_soft, lambda_k, lambda_d0, return_stats=True, d0_override=d0s.get('L3'), k_override=k0s.get('L3'))
            stats = {'raw_L1': raw_L1, 'raw_L2': raw_L2, 'raw_L3': raw_L3,
                     'gate_L1': gate_L1, 'gate_L2': gate_L2, 'gate_L3': gate_L3}
            lam_L1 = lam_L1.unsqueeze(1)  # [B, 1]
            lam_L2 = lam_L2.unsqueeze(1)  # [B, 1]
            lam_L3 = lam_L3.unsqueeze(1)  # [A, 1]
        else:
            # Sigmoid path, normal
            k_mode = getattr(opt, 'lambda_k_mode', 'static')
            k_scale = getattr(opt, 'lambda_k_scale', 1.0)
            d0s = {k: scaler.d_mean[k] for k in ('L1', 'L2', 'L3')} if scaler is not None else {}
            k0s = {k: k_scale / (scaler.d_std[k] + 1e-7) for k in ('L1', 'L2', 'L3')} if (scaler is not None and k_mode == 'auto') else {}
            lam_L1 = adaptive_lambda(d_L1, lambda_soft, lambda_k, lambda_d0, d0_override=d0s.get('L1'), k_override=k0s.get('L1')).unsqueeze(1)  # [B, 1]
            lam_L2 = adaptive_lambda(d_L2, lambda_soft, lambda_k, lambda_d0, d0_override=d0s.get('L2'), k_override=k0s.get('L2')).unsqueeze(1)  # [B, 1]
            lam_L3 = adaptive_lambda(d_L3, lambda_soft, lambda_k, lambda_d0, d0_override=d0s.get('L3'), k_override=k0s.get('L3')).unsqueeze(1)  # [A, 1]
    else:  # 'cw'
        lam_L1 = cell_wise_lambda(b_teacher_sim, b_student_sim,
                                  lambda_soft, lambda_k, lambda_d0)   # [B, B]
        lam_L2 = cell_wise_lambda(a_teacher_sim, a_student_sim,
                                  lambda_soft, lambda_k, lambda_d0)   # [B, A]
        lam_L3 = cell_wise_lambda(a_teacher_sim_t, a_student_sim_t,
                                  lambda_soft, lambda_k, lambda_d0)   # [A, B]

    b_teacher_sim   = (1 - lam_L1) * b_teacher_sim   + lam_L1 * sigma_bb
    a_teacher_sim   = (1 - lam_L2) * a_teacher_sim   + lam_L2 * sigma_ba
    a_teacher_sim_t = (1 - lam_L3) * a_teacher_sim_t + lam_L3 * sigma_ab

    # Normalize to probability distributions
    a_student_sim = a_student_sim / torch.sum(a_student_sim, dim=1, keepdim=True)
    a_teacher_sim = a_teacher_sim / torch.sum(a_teacher_sim, dim=1, keepdim=True)
    a_teacher_sim_t = a_teacher_sim_t / torch.sum(a_teacher_sim_t, dim=1, keepdim=True)
    a_student_sim_t = a_student_sim_t / torch.sum(a_student_sim_t, dim=1, keepdim=True)
    b_student_sim = b_student_sim / torch.sum(b_student_sim, dim=1, keepdim=True)
    b_teacher_sim = b_teacher_sim / torch.sum(b_teacher_sim, dim=1, keepdim=True)

    # KL divergence components
    L_1 = torch.sum(b_teacher_sim * torch.log((b_teacher_sim + eps) / (b_student_sim + eps)))
    L_2 = torch.sum(a_teacher_sim * torch.log((a_teacher_sim + eps) / (a_student_sim + eps)))
    L_3 = torch.sum(a_teacher_sim_t * torch.log((a_teacher_sim_t + eps) / (a_student_sim_t + eps)))

    AKD_loss = opt.l_1 * L_1 + L_2 * (1 - opt.l_2) + L_3 * opt.l_2
    return AKD_loss, stats
