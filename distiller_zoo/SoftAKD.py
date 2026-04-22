from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisagreementScaler:
    """Tracks per-term Pearson disagreement statistics (mean, std, max) via EMA.

    Supports two update frequencies:
      per_batch=False (default): EMA updated once per epoch from epoch aggregates.
      per_batch=True:            EMA updated every mini-batch using batch statistics.
    After lock_epoch: all stats are frozen, no further updates.

    lock_std_only=True: only d_std is locked at lock_epoch (for k computation).
      d_mean continues to update. d_std keeps tracking for display but k uses
      the snapshotted d_std_locked value from lock_epoch onward.

    dynamic_lock=True: instead of a fixed lock_epoch, freezes both k (d_std) and
      d_mean when the epoch-over-epoch delta_d_std falls below lock_std_delta for
      all three terms. Call freeze(epoch) from the training loop when this condition
      is met. lock_epoch still acts as a hard-deadline fallback.
    """
    def __init__(self, init=0.1, momentum=0.99, lock_epoch=240,
                 per_batch=False, batch_momentum=0.999, lock_std_only=False,
                 dynamic_lock=False, lock_std_delta=1e-4, save_folder=None):
        self.d_max          = {'L1': init, 'L2': init, 'L3': init}
        self.d_mean         = {'L1': init, 'L2': init, 'L3': init}
        self.d_std          = {'L1': init, 'L2': init, 'L3': init}
        self.d_std_locked   = None   # set at lock_epoch when lock_std_only=True, or by freeze()
        self.d_mean_locked  = None   # set by freeze() — both k and d0 are frozen together
        self.momentum       = momentum
        self.lock_epoch     = lock_epoch
        self.lock_std_only  = lock_std_only
        self.dynamic_lock   = dynamic_lock
        self.lock_std_delta = lock_std_delta
        self.save_folder    = save_folder
        self._locked        = False   # True after freeze() is called
        self.per_batch      = per_batch
        self.batch_momentum = batch_momentum
        self._current_epoch = 0
        self._epoch_max     = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}
        self._epoch_sum     = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}
        self._epoch_n       = {'L1': 0,    'L2': 0,    'L3': 0}
        self._epoch_sq      = {'L1': 0.0,  'L2': 0.0,  'L3': 0.0}

    def freeze(self, epoch):
        """Snapshot d_std and d_mean and freeze all further EMA updates.

        Called from the training loop when dynamic_lock=True and the
        epoch-over-epoch delta_d_std drops below lock_std_delta for all terms.
        """
        if self._locked:
            return
        self.d_std_locked  = {k: v for k, v in self.d_std.items()}
        self.d_mean_locked = {k: v for k, v in self.d_mean.items()}
        self._locked = True
        std_str  = ', '.join(f'{k}={self.d_std_locked[k]:.6f}'  for k in ('L1', 'L2', 'L3'))
        mean_str = ', '.join(f'{k}={self.d_mean_locked[k]:.6f}' for k in ('L1', 'L2', 'L3'))
        msg = (f'[SoftAKD] d_std converged at epoch {epoch}, freezing k and d_mean.\n'
               f'  std={{{std_str}}}\n'
               f'  mean={{{mean_str}}}')
        print(msg)
        from distiller_zoo.SoftAKD2 import _log_to_file
        _log_to_file(msg, self.save_folder)

    def update(self, d_L1, d_L2, d_L3):
        """Call once per batch. Updates stats either per-batch or accumulates for epoch."""
        for key, d in [('L1', d_L1), ('L2', d_L2), ('L3', d_L3)]:
            batch_max = d.max().item()
            if batch_max > self._epoch_max[key]:
                self._epoch_max[key] = batch_max
            self._epoch_sum[key] += d.sum().item()
            self._epoch_n[key]   += d.numel()
            self._epoch_sq[key]  += (d ** 2).sum().item()

            if self.per_batch and not self._locked:
                update_mean = self._current_epoch <= self.lock_epoch or self.lock_std_only
                update_std  = self._current_epoch <= self.lock_epoch or self.lock_std_only
                if update_mean or update_std or self._current_epoch <= self.lock_epoch:
                    m = self.batch_momentum
                    batch_mean = d.mean().item()
                    batch_var  = (d ** 2).mean().item() - batch_mean ** 2
                    batch_std  = max(batch_var, 0.0) ** 0.5
                    if update_mean:
                        self.d_mean[key] = m * self.d_mean[key] + (1 - m) * batch_mean
                    if update_std:
                        self.d_std[key]  = m * self.d_std[key]  + (1 - m) * batch_std
                    if self._current_epoch <= self.lock_epoch and batch_max > self.d_max[key]:
                        self.d_max[key] = m * self.d_max[key] + (1 - m) * batch_max

    def step_epoch(self, epoch):
        """Call once per epoch. Handles epoch-level EMA (if per_batch=False) and lock.

        lock_std_only=True: d_mean updates every epoch; d_std also continues tracking
        for monitoring, but d_std_locked is snapshotted at lock_epoch for k computation.
        """
        self._current_epoch = epoch
        if not self.per_batch:
            if epoch == 1:
                for key in self.d_max:
                    self.d_max[key]  = self._epoch_max[key]
                    self.d_mean[key] = self._epoch_sum[key] / self._epoch_n[key]
                    self.d_std[key]  = (self._epoch_sq[key] / self._epoch_n[key] - self.d_mean[key] ** 2) ** 0.5
            elif not self._locked:
                m = 0.9 if epoch <= 20 else self.momentum
                for key in self.d_max:
                    epoch_mean = self._epoch_sum[key] / self._epoch_n[key]
                    epoch_std  = (self._epoch_sq[key] / self._epoch_n[key] - epoch_mean ** 2) ** 0.5
                    # d_max: always lock at lock_epoch
                    if epoch <= self.lock_epoch:
                        self.d_max[key]  = m * self.d_max[key]  + (1 - m) * self._epoch_max[key]
                    # d_mean: update always if lock_std_only, else until lock_epoch
                    if self.lock_std_only or epoch <= self.lock_epoch:
                        self.d_mean[key] = m * self.d_mean[key] + (1 - m) * epoch_mean
                    # d_std: update always for tracking (k uses d_std_locked after lock)
                    if self.lock_std_only or epoch <= self.lock_epoch:
                        self.d_std[key]  = m * self.d_std[key]  + (1 - m) * epoch_std
            # Snapshot d_std at lock_epoch for k computation (lock_std_only hard deadline)
            if self.lock_std_only and epoch == self.lock_epoch and self.d_std_locked is None and not self._locked:
                self.d_std_locked = {k: v for k, v in self.d_std.items()}
        self._epoch_max = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}
        self._epoch_sum = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}
        self._epoch_n   = {'L1': 0,   'L2': 0,   'L3': 0}
        self._epoch_sq  = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}




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
                          labels, anchor_labels, gcn, alpha=0.1, eps=1e-7):
    """
    Adaptively soften precomputed sigma using a learnable GCN.

    Implements ASL-style dual optimization:
    1. Σ_t_hat = α * GCN(sigma) + (1-α) * sigma
    2. loss_G = MSE(Σ_t_hat, Σ_s) where Σ_s = student batch-to-batch similarity [B, B] → [C, C]
    """
    num_classes = sigma.size(0)
    device = sigma.device
    C = num_classes

    output_net = F.normalize(output_net, p=2, dim=1)   # [B, D]

    # Batch-to-batch student similarity
    sim = (torch.mm(output_net, output_net.t()) + 1) / 2
    row_labels = labels   # [B]
    col_labels = labels   # [B]

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
    else:
        disagreement = disagreement / (disagreement.max() + eps)
        midpoint = d0
    k_eff = k_override if k_override is not None else k
    gate = torch.sigmoid(k_eff * (disagreement - midpoint))
    if return_stats:
        return gate * lambda_max, raw_dis, gate.detach()
    return gate * lambda_max  # [N]




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

    # ===== SOFTENING: Blend teacher similarities with sigma ==========
    if sigma is None:
        # Student mode: use student's own batch similarities as the blend target
        sigma_bb = b_student_sim.detach()    # [B, B]
        sigma_ba = a_student_sim.detach()    # [B, A]
        sigma_ab = a_student_sim_t.detach()  # [A, B]
    else:
        sigma_bb = sigma[labels][:, labels]              # [B, B]
        sigma_ba = sigma[labels][:, anchor_labels]       # [B, A]
        sigma_ab = sigma[anchor_labels][:, labels]       # [A, B]

        sigma_temp = getattr(opt, 'sigma_temp', 1.0)
        if sigma_temp != 1.0:
            exponent = 1.0 / sigma_temp
            sigma_bb = sigma_bb ** exponent
            sigma_ba = sigma_ba ** exponent
            sigma_ab = sigma_ab ** exponent

    lambda_d0 = getattr(opt, 'lambda_d0', 0.3)

    d_L1 = pearson_distance_rows(b_teacher_sim, b_student_sim)      # [B]
    d_L2 = pearson_distance_rows(a_teacher_sim, a_student_sim)      # [B]
    d_L3 = pearson_distance_rows(a_teacher_sim_t, a_student_sim_t)  # [A]

    if scaler is not None:
        scaler.update(d_L1.detach(), d_L2.detach(), d_L3.detach())

    if scaler is not None:
        d_mean_src = scaler.d_mean_locked if scaler.d_mean_locked is not None else scaler.d_mean
        d0s = {k: d_mean_src[k] for k in ('L1', 'L2', 'L3')}
    else:
        d0s = {}
    if scaler is not None:
        d_std_for_k = scaler.d_std_locked if scaler.d_std_locked is not None else scaler.d_std
        k0s = {k: 1.0 / (d_std_for_k[k] + 1e-7) for k in ('L1', 'L2', 'L3')}
    else:
        k0s = {}

    stats = None
    if return_stats:
        lam_L1, raw_L1, gate_L1 = adaptive_lambda(d_L1, lambda_soft, d0_override=d0s.get('L1'), k_override=k0s.get('L1'), return_stats=True)
        lam_L2, raw_L2, gate_L2 = adaptive_lambda(d_L2, lambda_soft, d0_override=d0s.get('L2'), k_override=k0s.get('L2'), return_stats=True)
        lam_L3, raw_L3, gate_L3 = adaptive_lambda(d_L3, lambda_soft, d0_override=d0s.get('L3'), k_override=k0s.get('L3'), return_stats=True)
        stats = {'raw_L1': raw_L1, 'raw_L2': raw_L2, 'raw_L3': raw_L3,
                 'gate_L1': gate_L1, 'gate_L2': gate_L2, 'gate_L3': gate_L3}
        lam_L1 = lam_L1.unsqueeze(1)  # [B, 1]
        lam_L2 = lam_L2.unsqueeze(1)  # [B, 1]
        lam_L3 = lam_L3.unsqueeze(1)  # [A, 1]
    else:
        lam_L1 = adaptive_lambda(d_L1, lambda_soft, d0_override=d0s.get('L1'), k_override=k0s.get('L1')).unsqueeze(1)  # [B, 1]
        lam_L2 = adaptive_lambda(d_L2, lambda_soft, d0_override=d0s.get('L2'), k_override=k0s.get('L2')).unsqueeze(1)  # [B, 1]
        lam_L3 = adaptive_lambda(d_L3, lambda_soft, d0_override=d0s.get('L3'), k_override=k0s.get('L3')).unsqueeze(1)  # [A, 1]

    # Blend in similarity space, then normalize to probabilities
    b_teacher_sim   = (1 - lam_L1) * b_teacher_sim   + lam_L1 * sigma_bb
    a_teacher_sim   = (1 - lam_L2) * a_teacher_sim   + lam_L2 * sigma_ba
    a_teacher_sim_t = (1 - lam_L3) * a_teacher_sim_t + lam_L3 * sigma_ab

    a_student_sim   = a_student_sim   / torch.sum(a_student_sim,   dim=1, keepdim=True)
    a_teacher_sim   = a_teacher_sim   / torch.sum(a_teacher_sim,   dim=1, keepdim=True)
    a_teacher_sim_t = a_teacher_sim_t / torch.sum(a_teacher_sim_t, dim=1, keepdim=True)
    a_student_sim_t = a_student_sim_t / torch.sum(a_student_sim_t, dim=1, keepdim=True)
    b_student_sim   = b_student_sim   / torch.sum(b_student_sim,   dim=1, keepdim=True)
    b_teacher_sim   = b_teacher_sim   / torch.sum(b_teacher_sim,   dim=1, keepdim=True)

    # KL divergence
    L_1 = torch.sum(b_teacher_sim * torch.log((b_teacher_sim + eps) / (b_student_sim + eps)))
    L_2 = torch.sum(a_teacher_sim * torch.log((a_teacher_sim + eps) / (a_student_sim + eps)))
    L_3 = torch.sum(a_teacher_sim_t * torch.log((a_teacher_sim_t + eps) / (a_student_sim_t + eps)))

    AKD_loss = opt.l_1 * L_1 + L_2 * (1 - opt.l_2) + L_3 * opt.l_2
    return AKD_loss, stats
