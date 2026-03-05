from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        'ab' (default) — batch-to-anchor: sim(batch_class_i, anchor_class_j) [B, A] → [C, C]
        'aa'           — anchor-to-anchor: sim(anchor_i, anchor_j)           [A, A] → [C, C]
        'bb'           — batch-to-batch:   sim(batch_class_i, batch_class_j) [B, B] → [C, C]
    """
    num_classes = sigma.size(0)
    device = sigma.device
    C = num_classes

    output_net = F.normalize(output_net, p=2, dim=1)   # [B, D]
    anchor_net = F.normalize(anchor_net, p=2, dim=1)   # [A, D]

    if sigma_s_mode == 'ab':
        # Similarity between batch samples and anchors [B, A]
        sim = (torch.mm(output_net, anchor_net.t()) + 1) / 2
        row_labels = labels          # [B]
        col_labels = anchor_labels   # [A]

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


def adaptive_lambda(disagreement, lambda_soft, k=10.0, d0=0.5, eps=1e-7):
    """
    Per-sample lambda via sigmoid gate on normalised Pearson disagreement.

    When teacher and student agree (low disagreement) → lambda ≈ 1.0 (pure AKD).
    When they disagree strongly → lambda → lambda_soft (blend in more sigma).

        gate = sigmoid(k * (d_norm - d0))
        lambda = 1.0 - gate * (1.0 - lambda_soft)   ∈ [lambda_soft, 1.0]

    Args:
        disagreement: [N] Pearson distances (unnormalized)
        lambda_soft:  scalar floor value
        k:            sigmoid steepness (default 10.0)
        d0:           sigmoid midpoint on normalized scale (default 0.5)
    Returns:
        [N] per-sample lambda values
    """
    disagreement_norm = disagreement / (disagreement.max() + eps)
    gate = torch.sigmoid(k * (disagreement_norm - d0))
    return 1.0 - gate * (1.0 - lambda_soft)  # [N]


def soft_akd_loss(target_net, output_net, anchor_target, anchor_net,
                  labels, anchor_labels, sigma, lambda_soft, opt, eps=1e-7):
    """
    Soft Anchor-based Knowledge Distillation loss.

    Extends AKD by blending each of the three teacher similarity matrices
    with a precomputed class-level similarity matrix sigma, injecting global
    semantic structure into all three KL divergence terms.

        sigma_sharp = sigma_subset ^ (1 / sigma_temp)   (temperature sharpening)
        S_teacher_soft = lambda * S_teacher + (1 - lambda) * sigma_sharp

    Args:
        target_net:    teacher pooled features for the batch          [B, D]
        output_net:    student pooled features for the batch          [B, D]
        anchor_target: teacher pooled features for anchor images      [A, D]
        anchor_net:    student pooled features for anchor images      [A, D]
        labels:        batch class labels                             [B]
        anchor_labels: anchor class labels                            [A]
        sigma:         precomputed class similarity matrix             [C, C]
        lambda_soft:   blending weight (1.0 = pure AKD, 0.0 = pure class-level)
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

    lambda_k  = getattr(opt, 'lambda_k',  10.0)
    lambda_d0 = getattr(opt, 'lambda_d0', 0.5)

    lam_L1 = adaptive_lambda(pearson_distance_rows(b_teacher_sim, b_student_sim),
                             lambda_soft, lambda_k, lambda_d0).unsqueeze(1)   # [B, 1]
    lam_L2 = adaptive_lambda(pearson_distance_rows(a_teacher_sim, a_student_sim),
                             lambda_soft, lambda_k, lambda_d0).unsqueeze(1)   # [B, 1]
    lam_L3 = adaptive_lambda(pearson_distance_rows(a_teacher_sim_t, a_student_sim_t),
                             lambda_soft, lambda_k, lambda_d0).unsqueeze(1)   # [A, 1]

    b_teacher_sim   = lam_L1 * b_teacher_sim   + (1 - lam_L1) * sigma_bb
    a_teacher_sim   = lam_L2 * a_teacher_sim   + (1 - lam_L2) * sigma_ba
    a_teacher_sim_t = lam_L3 * a_teacher_sim_t + (1 - lam_L3) * sigma_ab

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
    return AKD_loss
