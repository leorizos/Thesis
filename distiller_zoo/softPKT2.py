from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import os


class SoftPKT2(nn.Module):
    """
    Soft Probabilistic Knowledge Transfer (Soft PKT v2)

    Extends PKT by blending instance-level and class-level similarity:
        S_t_soft = λ * S_t_instance + (1 - λ) * σ_class

    Where:
        - S_t_instance: Batch-level teacher similarity (original PKT)
        - σ_class: Pre-computed global class similarity matrix
        - λ: Blending weight (0 = pure class-level, 1 = pure instance-level)

    This addresses PKT's limitation of only seeing batch-level relationships
    by incorporating global semantic class structure.
    """

    def __init__(self, sigma_path='save/class_similarity_matrix.npy', lambda_soft=0.5):
        """
        Args:
            sigma_path: Path to pre-computed class similarity matrix [100, 100]
            lambda_soft: Blending weight λ ∈ [0, 1]
                        λ = 1.0: Pure instance-level (original PKT)
                        λ = 0.0: Pure class-level
                        λ = 0.5: Balanced (default)
        """
        super(SoftPKT2, self).__init__()

        self.lambda_soft = lambda_soft

        # Load pre-computed class similarity matrix
        if not os.path.exists(sigma_path):
            raise FileNotFoundError(
                f"Class similarity matrix not found at {sigma_path}. "
                f"Please run compute_class_similarity_matrix.py first."
            )

        sigma = np.load(sigma_path)

        # Verify shape and properties
        assert sigma.shape == (100, 100), f"Expected shape (100, 100), got {sigma.shape}"
        assert sigma.min() >= 0.0 and sigma.max() <= 1.0, \
            f"Sigma values should be in [0, 1], got range [{sigma.min()}, {sigma.max()}]"

        # Register as buffer (not a trainable parameter)
        self.register_buffer('sigma', torch.from_numpy(sigma).float())

        print(f'==> Soft PKT v2 initialized')
        print(f'    Lambda (λ): {self.lambda_soft}')
        print(f'    Sigma matrix loaded from: {sigma_path}')
        print(f'    Sigma stats: mean={sigma.mean():.4f}, std={sigma.std():.4f}')
        print(f'    Blending: {self.lambda_soft*100:.1f}% instance + {(1-self.lambda_soft)*100:.1f}% class')

    def forward(self, f_s, f_t, labels):
        """
        Args:
            f_s: Student features [B, D_s]
            f_t: Teacher features [B, D_t]
            labels: Class labels [B]

        Returns:
            loss: Scalar tensor (soft PKT loss)
        """
        loss = self.soft_cosine_similarity_loss(f_s, f_t, labels,
                                                 self.sigma, self.lambda_soft)
        return loss

    @staticmethod
    def soft_cosine_similarity_loss(output_net, target_net, labels, sigma, lambda_soft, eps=0.0000001):
        """
        Compute soft PKT loss with class-level similarity blending.

        Args:
            output_net: Student features [B, D_s]
            target_net: Teacher features [B, D_t]
            labels: Class labels [B]
            sigma: Class similarity matrix [100, 100]
            lambda_soft: Blending weight λ
            eps: Small constant for numerical stability

        Returns:
            loss: Scalar KL divergence loss
        """

        # ============================================================
        # SECTION 1: NORMALIZE STUDENT FEATURES (same as PKT)
        # ============================================================
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0  # Handle NaN

        # ============================================================
        # SECTION 2: NORMALIZE TEACHER FEATURES (same as PKT)
        # ============================================================
        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0  # Handle NaN

        # ============================================================
        # SECTION 3: COMPUTE SIMILARITY MATRICES (same as PKT)
        # ============================================================
        # Student similarity: [B, B]
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))

        # Teacher similarity: [B, B]
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # ============================================================
        # SECTION 4: SCALE TO [0, 1] (same as PKT)
        # ============================================================
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # ============================================================
        # SECTION 5: SOFT PKT - EXTRACT CLASS-LEVEL SIMILARITIES
        # ============================================================
        # Extract relevant class similarities for this batch
        # sigma_batch[i, j] = sigma[labels[i], labels[j]]

        batch_size = labels.size(0)
        sigma_batch = sigma[labels][:, labels]  # [B, B]

        # Ensure sigma_batch is on the same device as target_similarity
        sigma_batch = sigma_batch.to(target_similarity.device)

        # ============================================================
        # SECTION 6: SOFT PKT - BLEND INSTANCE + CLASS SIMILARITIES
        # ============================================================
        # S_t_soft = λ * S_t_instance + (1 - λ) * σ_class
        target_similarity_soft = lambda_soft * target_similarity + (1 - lambda_soft) * sigma_batch

        # ============================================================
        # SECTION 7: NORMALIZE TO PROBABILITIES (same as PKT)
        # ============================================================
        # Row-wise normalization to create probability distributions
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity_soft = target_similarity_soft / torch.sum(target_similarity_soft, dim=1, keepdim=True)

        # ============================================================
        # SECTION 8: COMPUTE KL DIVERGENCE (same as PKT)
        # ============================================================
        # KL(P_teacher_soft || P_student)
        loss = torch.mean(target_similarity_soft * torch.log((target_similarity_soft + eps) / (model_similarity + eps)))

        return loss