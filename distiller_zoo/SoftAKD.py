from __future__ import print_function, division

import csv
import math
import os

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
    if anchor_net is not None:
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
    else:
        disagreement = disagreement / (disagreement.max() + eps)
        midpoint = d0
    k_eff = k_override if k_override is not None else k
    gate = torch.sigmoid(k_eff * (disagreement - midpoint))
    if return_stats:
        return gate * lambda_max, raw_dis, gate.detach()
    return gate * lambda_max  # [N]



# ---------------------------------------------------------------------------
# Typicality helpers (pure Python / small torch; no grad needed)
# ---------------------------------------------------------------------------

def _mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')

def _std(lst):
    if len(lst) < 2:
        return float('nan')
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

def _pearson_lists(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return float('nan')
    xs, ys = list(xs)[:n], list(ys)[:n]
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-7 or sy < 1e-7:
        return float('nan')
    return cov / (sx * sy)

def _fmt(v):
    try:
        return f'{float(v):.6f}' if not math.isnan(float(v)) else 'nan'
    except (TypeError, ValueError):
        return 'nan'


def compute_typicality(sim_matrix, sigma_full, labels, eps=1e-7):
    """Measure how well the batch's class-level similarity structure matches sigma.

    Aggregates the B×B raw similarity matrix into a C_b×C_b class-level matrix
    (where C_b = unique classes in batch), then correlates with the corresponding
    C_b×C_b submatrix of sigma_full via Pearson correlation.

    Args:
        sim_matrix: [B, B] raw cosine similarity (teacher or student), values in [0,1]
        sigma_full: [C_full, C_full] precomputed class similarity matrix
        labels:     [B] integer class labels (indices into sigma_full rows/cols)
    Returns:
        typ_global:         scalar Pearson r on upper triangle (excl. diagonal)
        typ_row_values:     list[float] per-class Pearson r (nan if < 3 off-diag elements)
        num_unique_classes: int
    """
    device = sim_matrix.device
    B = sim_matrix.size(0)

    unique_classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    C_b = len(unique_classes)

    counts = torch.zeros(C_b, device=device)
    counts.scatter_add_(0, inverse, torch.ones(B, device=device))

    # Row-aggregate B×B → C_b×B
    row_agg = torch.zeros(C_b, B, device=device)
    row_agg.scatter_add_(0, inverse.unsqueeze(1).expand(-1, B), sim_matrix)
    row_agg = row_agg / counts.unsqueeze(1).clamp(min=1)

    # Col-aggregate C_b×B → C_b×C_b
    class_sim = torch.zeros(C_b, C_b, device=device)
    class_sim.scatter_add_(1, inverse.unsqueeze(0).expand(C_b, -1), row_agg)
    class_sim = class_sim / counts.unsqueeze(0).clamp(min=1)

    sigma_sub = sigma_full[unique_classes][:, unique_classes]  # [C_b, C_b]

    # Global: Pearson on upper triangle (excl. diagonal)
    triu_mask = torch.triu(torch.ones(C_b, C_b, dtype=torch.bool, device=device), diagonal=1)
    x = class_sim[triu_mask]
    y = sigma_sub[triu_mask]
    if x.numel() < 2:
        typ_global = float('nan')
    else:
        x_c = x - x.mean()
        y_c = y - y.mean()
        typ_global = ((x_c * y_c).sum() / (x_c.norm() * y_c.norm() + eps)).item()

    # Row-wise: vectorized Pearson over all C_b rows simultaneously
    if C_b >= 4:
        off_diag   = ~torch.eye(C_b, dtype=torch.bool, device=device)  # [C_b, C_b]
        off_diag_f = off_diag.float()
        n_off      = float(C_b - 1)
        x_sum  = (class_sim * off_diag_f).sum(dim=1, keepdim=True)     # [C_b, 1]
        y_sum  = (sigma_sub * off_diag_f).sum(dim=1, keepdim=True)
        x_c    = (class_sim - x_sum / n_off) * off_diag_f              # [C_b, C_b]
        y_c    = (sigma_sub - y_sum / n_off) * off_diag_f
        cov    = (x_c * y_c).sum(dim=1)                                 # [C_b]
        sx     = (x_c ** 2).sum(dim=1).sqrt()
        sy     = (y_c ** 2).sum(dim=1).sqrt()
        denom  = sx * sy
        row_r  = torch.where(denom < eps,
                             torch.full_like(cov, float('nan')),
                             cov / (denom + eps))
        typ_row_values = row_r.tolist()
    else:
        typ_row_values = [float('nan')] * C_b

    return typ_global, typ_row_values, C_b


def compute_row_typicality(class_sim_matrix, sigma_full, eps=1e-7):
    """Row-wise Pearson correlation between each class row and the corresponding sigma row.

    Used to precompute teacher typicality once before training (from the teacher's C×C
    anchor similarity matrix), and per-batch for student typicality.

    Args:
        class_sim_matrix: [C, C] class-level similarity matrix (class order matches sigma_full)
        sigma_full:       [C, C] precomputed class similarity matrix
    Returns:
        typ_row: tensor [C] per-class Pearson r (nan for rows with < 3 off-diagonal elements)
    """
    C = class_sim_matrix.size(0)
    device = class_sim_matrix.device
    if C < 4:
        return torch.full((C,), float('nan'), device=device)
    # Vectorized: compute all C row-wise Pearson correlations simultaneously
    off_diag   = ~torch.eye(C, dtype=torch.bool, device=device)   # [C, C]
    off_diag_f = off_diag.float()
    n_off      = float(C - 1)
    x_sum  = (class_sim_matrix * off_diag_f).sum(dim=1, keepdim=True)
    y_sum  = (sigma_full * off_diag_f).sum(dim=1, keepdim=True)
    x_c    = (class_sim_matrix - x_sum / n_off) * off_diag_f
    y_c    = (sigma_full       - y_sum / n_off) * off_diag_f
    cov    = (x_c * y_c).sum(dim=1)
    sx     = (x_c ** 2).sum(dim=1).sqrt()
    sy     = (y_c ** 2).sum(dim=1).sqrt()
    denom  = sx * sy
    return torch.where(denom < eps,
                       torch.full_like(cov, float('nan')),
                       cov / (denom + eps))



class TypicalityLogger:
    """Logs per-batch and per-epoch typicality statistics to two CSV files.

    Per-batch CSV (every LOG_EVERY_N_BATCHES batches):
        epoch, batch_idx, typ_t/s globals, row mean/std/min/max,
        num_unique_classes, mean_lambda, mean_difficulty

    Per-epoch CSV (every epoch):
        epoch, typ_t/s global mean±std, row mean-of-means, row mean-of-stds,
        corr(typ_t, difficulty), corr(typ_s, difficulty), corr(typ_t, typ_s)

    Stdout per epoch:
        [Typicality] Epoch N: typ_t=mean±std, typ_s=mean±std, corr(t,s)=val
    """

    LOG_EVERY_N_BATCHES = 260

    def __init__(self, save_folder):
        from datetime import datetime
        self.save_folder = save_folder
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._batch_path = os.path.join(save_folder, f'typicality_batch_{ts}.csv')
        self._epoch_path = os.path.join(save_folder, f'typicality_epoch_{ts}.csv')

        _batch_cols = [
            'epoch', 'batch_idx',
            'typ_t_global', 'typ_s_global',
            'typ_t_row_mean', 'typ_t_row_std',
            'typ_s_row_mean', 'typ_s_row_std',
            'typ_t_row_min', 'typ_t_row_max',
            'typ_s_row_min', 'typ_s_row_max',
            'num_unique_classes', 'mean_lambda', 'mean_difficulty',
            'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
        ]
        _epoch_cols = [
            'epoch',
            'typ_t_global_mean', 'typ_t_global_std',
            'typ_s_global_mean', 'typ_s_global_std',
            'typ_t_row_mean_of_means', 'typ_t_row_mean_of_stds',
            'typ_s_row_mean_of_means', 'typ_s_row_mean_of_stds',
            'corr_typ_t_difficulty', 'corr_typ_s_difficulty', 'corr_typ_t_typ_s',
            'alpha_mean', 'alpha_std', 'ema_s_mean', 'ema_s_std',
        ]
        with open(self._batch_path, 'w', newline='') as f:
            csv.writer(f).writerow(_batch_cols)
        with open(self._epoch_path, 'w', newline='') as f:
            csv.writer(f).writerow(_epoch_cols)

        self._reset_epoch()

    def _reset_epoch(self):
        self._ep_typ_t_global  = []
        self._ep_typ_s_global  = []
        self._ep_t_row_means   = []
        self._ep_t_row_stds    = []
        self._ep_s_row_means   = []
        self._ep_s_row_stds    = []
        self._ep_difficulty    = []
        self._ep_alpha_means   = []
        self._ep_alpha_stds    = []

    def log_batch(self, epoch, batch_idx,
                  typ_t_global, typ_t_row_values,
                  typ_s_global, typ_s_row_values,
                  num_unique_classes, mean_lambda, mean_difficulty,
                  alpha_stats=None):
        # Always accumulate for epoch summary
        if not math.isnan(float(typ_t_global)):
            self._ep_typ_t_global.append(typ_t_global)
        if not math.isnan(float(typ_s_global)):
            self._ep_typ_s_global.append(typ_s_global)
        self._ep_difficulty.append(mean_difficulty)

        t_valid = [v for v in typ_t_row_values if not math.isnan(float(v))]
        s_valid = [v for v in typ_s_row_values if not math.isnan(float(v))]
        t_row_mean = _mean(t_valid)
        t_row_std  = _std(t_valid)
        s_row_mean = _mean(s_valid)
        s_row_std  = _std(s_valid)
        if not math.isnan(float(t_row_mean)):
            self._ep_t_row_means.append(t_row_mean)
            self._ep_t_row_stds.append(t_row_std if not math.isnan(float(t_row_std)) else 0.0)
        if not math.isnan(float(s_row_mean)):
            self._ep_s_row_means.append(s_row_mean)
            self._ep_s_row_stds.append(s_row_std if not math.isnan(float(s_row_std)) else 0.0)
        if alpha_stats is not None:
            self._ep_alpha_means.append(alpha_stats['mean'])
            self._ep_alpha_stds.append(alpha_stats['std'])

        if batch_idx % self.LOG_EVERY_N_BATCHES != 0:
            return

        a_mean = _fmt(alpha_stats['mean']) if alpha_stats else 'nan'
        a_std  = _fmt(alpha_stats['std'])  if alpha_stats else 'nan'
        a_min  = _fmt(alpha_stats['min'])  if alpha_stats else 'nan'
        a_max  = _fmt(alpha_stats['max'])  if alpha_stats else 'nan'
        row = [
            epoch, batch_idx,
            _fmt(typ_t_global), _fmt(typ_s_global),
            _fmt(t_row_mean), _fmt(t_row_std),
            _fmt(s_row_mean), _fmt(s_row_std),
            _fmt(min(t_valid) if t_valid else float('nan')),
            _fmt(max(t_valid) if t_valid else float('nan')),
            _fmt(min(s_valid) if s_valid else float('nan')),
            _fmt(max(s_valid) if s_valid else float('nan')),
            num_unique_classes,
            _fmt(mean_lambda), _fmt(mean_difficulty),
            a_mean, a_std, a_min, a_max,
        ]
        with open(self._batch_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def end_epoch(self, epoch, typ_scaler=None):
        tg = self._ep_typ_t_global
        sg = self._ep_typ_s_global
        d  = self._ep_difficulty

        tg_mean = _mean(tg);  tg_std = _std(tg)
        sg_mean = _mean(sg);  sg_std = _std(sg)
        t_rm    = _mean(self._ep_t_row_means)
        t_rs    = _mean(self._ep_t_row_stds)
        s_rm    = _mean(self._ep_s_row_means)
        s_rs    = _mean(self._ep_s_row_stds)

        n = min(len(tg), len(d))
        corr_t_d = _pearson_lists(tg[:n], d[:n])
        n = min(len(sg), len(d))
        corr_s_d = _pearson_lists(sg[:n], d[:n])
        n = min(len(tg), len(sg))
        corr_t_s = _pearson_lists(tg[:n], sg[:n])

        a_mean = _mean(self._ep_alpha_means)
        a_std  = _mean(self._ep_alpha_stds)
        ema_s_mean = float('nan')
        ema_s_std  = float('nan')

        row = [
            epoch,
            _fmt(tg_mean), _fmt(tg_std),
            _fmt(sg_mean), _fmt(sg_std),
            _fmt(t_rm), _fmt(t_rs),
            _fmt(s_rm), _fmt(s_rs),
            _fmt(corr_t_d), _fmt(corr_s_d), _fmt(corr_t_s),
            _fmt(a_mean), _fmt(a_std), _fmt(ema_s_mean), _fmt(ema_s_std),
        ]
        with open(self._epoch_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        sg_mean_p = sg_mean if not math.isnan(float(sg_mean)) else float('nan')
        sg_std_p  = sg_std  if not math.isnan(float(sg_std))  else 0.0
        a_mean_p  = a_mean  if not math.isnan(float(a_mean))  else float('nan')
        a_std_p   = a_std   if not math.isnan(float(a_std))   else 0.0
        print(f'[Typicality] Epoch {epoch}: '
              f'α_mean={a_mean_p:.4f}±{a_std_p:.4f}, '
              f'typ_s={sg_mean_p:.4f}±{sg_std_p:.4f}, '
              f'ema_s_mean={ema_s_mean:.4f}, ema_s_std={ema_s_std:.4f}')

        self._reset_epoch()


def soft_akd_loss(target_net, output_net, anchor_target, anchor_net,
                  labels, anchor_labels, sigma, lambda_soft, opt, eps=1e-7,
                  return_stats=False, scaler=None,
                  typicality_logger=None, batch_idx=0, epoch=0,
                  typ_t_precomputed=None):
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

    # Typicality snapshot — raw sims before any blending or GCN softening.
    # Computed every LOG_STRIDE batches (not every batch) to avoid Python overhead.
    _LOG_STRIDE = TypicalityLogger.LOG_EVERY_N_BATCHES // 10  # every ~26 batches
    _typ_t_global = _typ_s_global = _typ_t_rows = _typ_s_rows = _typ_n_cls = None
    if typicality_logger is not None and sigma is not None and batch_idx % _LOG_STRIDE == 0:
        with torch.no_grad():
            _typ_t_global, _typ_t_rows, _typ_n_cls = compute_typicality(b_teacher_sim, sigma, labels)
            _typ_s_global, _typ_s_rows, _         = compute_typicality(b_student_sim,  sigma, labels)

    # ===== Per-class dynamic power scaling α(c) = exp(z(c)) =====================
    # Computed under no_grad from student anchor typicality + precomputed teacher typicality.
    # σ_temp (opt.sigma_temp) is superseded by this mechanism and is no longer applied.
    alpha_full = None
    if typ_t_precomputed is not None and sigma is not None:
        with torch.no_grad():
            # Student typicality: anchor-anchor sim [A, A], A=C since anchors span all classes
            a_anchor_sim = (torch.mm(anchor_net, anchor_net.t()) + 1) / 2     # [A, A]
            typ_s_full   = compute_row_typicality(a_anchor_sim, sigma)         # [C]
            typ_t_dev    = typ_t_precomputed.to(sigma.device)
            # Replace NaNs with 0, clamp to ignore negative correlations
            typ_t_safe   = torch.nan_to_num(typ_t_dev,  nan=0.0).clamp(min=0.0)
            typ_s_safe   = torch.nan_to_num(typ_s_full, nan=0.0).clamp(min=0.0)
            # Ratio: high when teacher is typical but student is not
            alpha_full   = (typ_t_safe + eps) / (typ_s_safe + eps)            # [C]

    # ===== SOFTENING: Blend teacher similarities with power-scaled sigma ==========
    if sigma is None:
        # Student mode: use student's own batch similarities as the blend target
        sigma_bb = b_student_sim.detach()    # [B, B]
        sigma_ba = a_student_sim.detach()    # [B, A]
        sigma_ab = a_student_sim_t.detach()  # [A, B]
    else:
        sigma_bb = sigma[labels][:, labels]              # [B, B]
        sigma_ba = sigma[labels][:, anchor_labels]       # [B, A]
        sigma_ab = sigma[anchor_labels][:, labels]       # [A, B]

        # Per-class dynamic power scaling: α(c) = exp(z(c)) from typicality.
        # L1/L2 rows are batch samples → α indexed by batch labels.
        # L3 rows are anchor samples  → α indexed by anchor_labels.
        # Falls back to static sigma_temp when typ infrastructure is not provided.
        if alpha_full is not None:
            alpha_bb = alpha_full[labels]                                       # [B]
            alpha_ba = alpha_full[labels]                                       # [B]
            alpha_ab = alpha_full[anchor_labels]                                # [A]
            sigma_bb = sigma_bb.clamp(min=1e-6) ** alpha_bb.unsqueeze(1)       # [B, B]
            sigma_ba = sigma_ba.clamp(min=1e-6) ** alpha_ba.unsqueeze(1)       # [B, A]
            sigma_ab = sigma_ab.clamp(min=1e-6) ** alpha_ab.unsqueeze(1)       # [A, B]
        else:
            sigma_temp = getattr(opt, 'sigma_temp', None)
            if sigma_temp is not None and sigma_temp != 1.0:
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

    # Logging
    if typicality_logger is not None and _typ_t_global is not None:
        alpha_stats = None
        if alpha_full is not None:
            with torch.no_grad():
                alpha_stats = {
                    'mean': alpha_full.mean().item(),
                    'std':  alpha_full.std(unbiased=False).item(),
                    'min':  alpha_full.min().item(),
                    'max':  alpha_full.max().item(),
                }
        typicality_logger.log_batch(
            epoch, batch_idx,
            _typ_t_global, _typ_t_rows,
            _typ_s_global, _typ_s_rows,
            _typ_n_cls,
            lam_L1.mean().item(),
            d_L1.mean().item(),
            alpha_stats=alpha_stats,
        )

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
