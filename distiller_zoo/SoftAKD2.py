from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """General Class Network - learnable confusion matrix M.

    Initialized as identity + small noise so that softmax rows start
    peaked at the diagonal (each class most similar to itself).
    During training, M learns class-level confusion patterns by watching
    what the student actually produces.

    Similar to the class uncertainty network in the ASL paper.
    """

    def __init__(self, num_classes):
        super(GCN, self).__init__()
        identity = torch.eye(num_classes)
        noise = torch.randn(num_classes, num_classes) * 1e-4
        self.M = nn.Parameter(identity + noise)

    def forward(self, sigma_matrix):

        return sigma_matrix @ self.M


def soften_sigma_with_gcn(sigma, feat_teacher, feat_student,
                          a_feat_t, a_feat_s,
                          labels, anchor_labels, gcn, alpha=0.1, eps=1e-7):
    """Soften a precomputed class similarity matrix using GCN.

    Produces a GCN-modulated sigma for use in soft_akd_loss, plus a
    separate loss_G to train the GCN.

    Steps:
        1. Sigma_g = gcn(sigma)           — GCN transforms sigma via sigma @ M
        2. Sigma_t_hat = alpha * Sigma_g + (1 - alpha) * sigma
        3. Sigma_s_class = student's actual class-level confusion (detached)
        4. loss_G = MSE(Sigma_t_hat, Sigma_s_class)

    Gradient isolation:
        - sigma_soft is returned DETACHED so the distillation loss does not
          send gradients to GCN.
        - Student similarities are DETACHED so loss_G does not send
          gradients to the student.

    Args:
        sigma:         Precomputed class similarity matrix  [C, C]
        feat_teacher:  Teacher batch features  [B, D]  (detached)
        feat_student:  Student batch features  [B, D]
        a_feat_t:      Teacher anchor features [A, D]  (detached)
        a_feat_s:      Student anchor features [A, D]
        labels:        Batch class labels  [B]
        anchor_labels: Anchor class labels [A]
        gcn:           GCN module (learnable M)
        alpha:         Blending weight (0 = original sigma, 1 = full GCN)
        eps:           Numerical-stability constant

    Returns:
        sigma_soft: Softened sigma [C, C] (detached from GCN graph)
        loss_G:     GCN loss (trains M only)
    """
    num_classes = sigma.size(0)
    device = sigma.device

    # 1. GCN transforms sigma
    Sigma_g = gcn(sigma)                                         # [C, C]
    Sigma_t_hat = alpha * Sigma_g + (1 - alpha) * sigma          # [C, C]

    # 2. Compute student's actual class-level confusion (all detached)
    feat_s_norm = F.normalize(feat_student.detach(), p=2, dim=1)
    a_feat_s_norm = F.normalize(a_feat_s.detach(), p=2, dim=1)
    a_student_sim = (torch.mm(feat_s_norm, a_feat_s_norm.t()) + 1) / 2  # [B, A]

    # Aggregate to class level: Sigma_s_class[i, j] = mean student similarity
    # between batch samples of class i and anchors of class j
    Sigma_s_class = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, num_classes, device=device)

    for c in range(num_classes):
        mask_batch = (labels == c)
        if not mask_batch.any():
            continue
        sim_from_c = a_student_sim[mask_batch].mean(dim=0)       # [A]

        for c2 in range(num_classes):
            mask_anchor = (anchor_labels == c2)
            if not mask_anchor.any():
                continue
            Sigma_s_class[c, c2] = sim_from_c[mask_anchor].mean()
            counts[c, c2] = 1.0

    # 3. GCN loss: push Sigma_t_hat toward the student's actual confusion
    valid = counts > 0
    if valid.any():
        loss_G = F.mse_loss(Sigma_t_hat[valid], Sigma_s_class[valid])
    else:
        loss_G = torch.tensor(0.0, device=device)

    # 4. Return detached sigma for distill loss, and loss_G for GCN training
    return Sigma_t_hat.detach(), loss_G


SNAPSHOT_EPOCHS = {1, 150, 180, 210, 240}


def monitor_gcn(gcn, epoch, num_classes, sigma=None, alpha=0.1, snapshot_store=None):
    """Print GCN/M matrix statistics every 5 epochs and capture snapshots.

    Args:
        gcn:            GCN module
        epoch:          Current epoch number
        num_classes:    Total number of classes
        sigma:          Precomputed class similarity matrix [C, C] (optional)
        alpha:          Blending weight used in training (for correct display)
        snapshot_store: Dict to accumulate matrix snapshots for later plotting
    """
    with torch.no_grad():
        M = gcn.M
        off_diag_mask = ~torch.eye(num_classes, dtype=torch.bool, device=M.device)

        if epoch % 5 == 0:
            diag_mean = M.diag().mean().item()
            off_diag_mean = M[off_diag_mask].mean().item()
            matrix_var = M.var().item()

            print("\n" + "=" * 60)
            print(f"GCN/M Matrix Monitoring - Epoch {epoch}")
            print("=" * 60)
            print("Raw M statistics:")
            print(f"  Diagonal mean:     {diag_mean:.4f}  (init ~1.0)")
            print(f"  Off-diagonal mean: {off_diag_mean:.4f}  (init ~0.0)")
            print(f"  Matrix variance:   {matrix_var:.6f}  (init ~1e-8)")

            if sigma is not None:
                Sigma_g = gcn(sigma)
                Sigma_t_hat = alpha * Sigma_g + (1 - alpha) * sigma

                print("Sigma_g (GCN output = sigma @ M):")
                print(f"  Diagonal mean:     {Sigma_g.diag().mean().item():.4f}")
                print(f"  Off-diagonal mean: {Sigma_g[off_diag_mask].mean().item():.6f}")
                print(f"Sigma_t_hat (blended, alpha={alpha}):")
                print(f"  Diagonal mean:     {Sigma_t_hat.diag().mean().item():.4f}")
                print(f"  Off-diagonal mean: {Sigma_t_hat[off_diag_mask].mean().item():.4f}")
            print("=" * 60 + "\n")

        # Capture snapshot for plotting (independent of print frequency)
        if snapshot_store is not None and epoch in SNAPSHOT_EPOCHS and sigma is not None:
            Sigma_g = gcn(sigma)
            Sigma_t_hat = alpha * Sigma_g + (1 - alpha) * sigma
            diff = Sigma_t_hat - sigma
            snapshot_store[epoch] = {
                'Sigma_g':     Sigma_g[:10, :10].cpu().numpy(),
                'Sigma_t_hat': Sigma_t_hat[:10, :10].cpu().numpy(),
                'diff':        diff[:10, :10].cpu().numpy(),
            }


def save_gcn_plots(snapshot_store, loss_G_history, akd_loss_history, save_folder, alpha):
    """Save GCN diagnostic plots to save_folder at end of training.

    Produces:
        gcn_matrices.png  – 3 rows (Sigma_g, Sigma_t_hat, Diff) x N epoch cols
        gcn_losses.png    – loss_G and AKD_loss curves over all epochs
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    epochs_present = sorted(snapshot_store.keys())
    n_epochs = len(epochs_present)

    if n_epochs > 0:
        fig, axes = plt.subplots(3, n_epochs, figsize=(4 * n_epochs, 11))
        if n_epochs == 1:
            axes = axes[:, None]

        row_labels = [
            f'Sigma_g  (sigma @ M)',
            f'Sigma_t_hat  (blended, alpha={alpha})',
            'Diff  (Sigma_t_hat - sigma)',
        ]
        keys = ['Sigma_g', 'Sigma_t_hat', 'diff']

        for col, ep in enumerate(epochs_present):
            snap = snapshot_store[ep]
            for row, (key, row_label) in enumerate(zip(keys, row_labels)):
                ax = axes[row, col]
                mat = snap[key]
                vmax = max(abs(mat.min()), abs(mat.max()))
                cmap = 'RdBu_r' if key == 'diff' else 'viridis'
                vmin = -vmax if key == 'diff' else mat.min()
                im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
                ax.set_title(f'Epoch {ep}', fontsize=9)
                if col == 0:
                    ax.set_ylabel(row_label, fontsize=8)
                ax.set_xlabel('Class', fontsize=7)
                ax.set_ylabel(ax.get_ylabel(), fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle('GCN Matrix Evolution (classes 0-9)', fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, 'gcn_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved gcn_matrices.png to {save_folder}')

        # Save raw matrices as .npy for exact value inspection
        for ep in epochs_present:
            snap = snapshot_store[ep]
            for key in ['Sigma_g', 'Sigma_t_hat', 'diff']:
                npy_path = os.path.join(save_folder, f'gcn_{key}_epoch{ep}.npy')
                np.save(npy_path, snap[key])
        print(f'Saved .npy matrix files to {save_folder}')

    if loss_G_history or akd_loss_history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if loss_G_history:
            axes[0].plot(loss_G_history)
            axes[0].set_title('GCN Loss (loss_G) per Epoch')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True)

        if akd_loss_history:
            axes[1].plot(akd_loss_history, color='orange')
            axes[1].set_title('AKD Loss per Epoch')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, 'gcn_losses.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved gcn_losses.png to {save_folder}')