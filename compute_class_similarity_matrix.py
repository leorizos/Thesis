"""
Compute Class Similarity Matrix for CIFAR100
This script computes a 100x100 similarity matrix where each entry (i,j) represents
the average similarity between all samples from class i and all samples from class j.
Similarity is computed using the same method as PKT: cosine similarity after L2 normalization,
scaled to [0, 1].
"""

from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns

from models import model_dict


def parse_args():
    parser = argparse.ArgumentParser('Compute class similarity matrix')
    parser.add_argument('--teacher_path', type=str,
                        default='save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth',
                        help='path to teacher model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet32x4',
                        help='teacher model architecture')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of data loading workers')
    parser.add_argument('--output_path', type=str,
                        default='save/class_similarity_matrix.npy',
                        help='path to save the similarity matrix')
    parser.add_argument('--data_folder', type=str, default='../data/',
                        help='path to dataset')
    parser.add_argument('--visualize', action='store_true',
                        help='create and save visualization heatmap')

    args = parser.parse_args()
    return args


def get_cifar100_train_loader(data_folder, batch_size=128, num_workers=2):
    """
    Get CIFAR100 training data loader WITHOUT augmentation for feature extraction.
    """
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    # Use test transform (no augmentation) for consistent feature extraction
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=transform)

    train_loader = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=False,  # Keep order for consistent class grouping
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, train_set


def load_teacher_model(model_path, model_name, num_classes=100, device='cuda'):
    """
    Load the pre-trained teacher model.
    """
    print(f'==> Loading teacher model: {model_name}')
    print(f'    Checkpoint: {model_path}')

    model = model_dict[model_name](num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print('==> Teacher model loaded successfully')
    return model


@torch.no_grad()
def extract_features(model, data_loader, device='cuda'):
    """
    Extract features from all samples in the dataset.
    Returns:
        features: tensor of shape [N, D] where N is number of samples
        labels: tensor of shape [N] with class labels
    """
    print('\n==> Extracting features from all samples...')

    all_features = []
    all_labels = []
    total_samples = len(data_loader.dataset)
    processed_samples = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)

        # Forward pass with feature extraction
        # is_feat=True returns [f0, f1, f2, f3, f4], logits
        # f4 is the pooled feature before FC layer (same as used in PKT)
        feats, _ = model(images, is_feat=True)
        features = feats[-1]  # Use final pooled features (f4)

        all_features.append(features.cpu())
        all_labels.append(targets)

        processed_samples += images.size(0)

        # Checkpoint every 1000 samples
        if (batch_idx + 1) % 10 == 0 or processed_samples >= total_samples:
            print(f'    Progress: {processed_samples}/{total_samples} samples processed')

    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f'==> Feature extraction complete!')
    print(f'    Features shape: {all_features.shape}')
    print(f'    Labels shape: {all_labels.shape}')

    return all_features, all_labels


def organize_features_by_class(features, labels, num_classes=100):
    """
    Organize features by class.
    Returns:
        class_features: list of 100 tensors, each of shape [N_i, D]
    """
    print('\n==> Organizing features by class...')

    class_features = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        class_feat = features[class_mask]
        class_features.append(class_feat)

        if (class_idx + 1) % 20 == 0:
            print(f'    Progress: {class_idx + 1}/{num_classes} classes organized')

    # Verify all classes have samples
    for class_idx, feat in enumerate(class_features):
        assert feat.size(0) > 0, f"Class {class_idx} has no samples!"

    print(f'==> Organization complete!')
    print(f'    Samples per class (first 10): {[feat.size(0) for feat in class_features[:10]]}')

    return class_features


def compute_class_similarity_matrix(class_features, num_classes=100, device='cuda'):
    """
    Compute the 100x100 class similarity matrix.
    For each pair of classes (i, j):
        1. Compute all pairwise cosine similarities between samples
        2. Scale to [0, 1]: (cosine + 1) / 2
        3. Average all pairwise similarities

    This follows the exact same similarity computation as PKT.
    """
    print('\n==> Computing class similarity matrix...')
    print('    This may take a few minutes...')

    sigma = torch.zeros(num_classes, num_classes)

    for i in range(num_classes):
        # Process class i
        features_i = class_features[i].to(device)

        # L2 normalize (same as PKT does)
        features_i_norm = F.normalize(features_i, p=2, dim=1)

        for j in range(num_classes):
            features_j = class_features[j].to(device)

            # L2 normalize
            features_j_norm = F.normalize(features_j, p=2, dim=1)

            # Compute all pairwise cosine similarities: [N_i, D] @ [D, N_j] = [N_i, N_j]
            cosine_sims = torch.mm(features_i_norm, features_j_norm.T)

            # Scale to [0, 1] (same as PKT: (cosine + 1) / 2)
            scaled_sims = (cosine_sims + 1.0) / 2.0

            # Average all pairwise similarities
            sigma[i, j] = scaled_sims.mean().item()

        # Checkpoint every 10 classes
        if (i + 1) % 10 == 0:
            print(f'    Progress: {i + 1}/{num_classes} classes completed')
            print(f'        Current diagonal value σ[{i},{i}] = {sigma[i, i]:.4f} (should be close to 1.0)')

    print(f'\n==> Class similarity matrix computation complete!')

    return sigma


def verify_matrix(sigma, num_classes=100):
    """
    Verify the computed similarity matrix has expected properties.
    """
    print('\n==> Verifying similarity matrix properties...')

    # Check shape
    assert sigma.shape == (num_classes, num_classes), f"Wrong shape: {sigma.shape}"
    print(f'    ✓ Shape is correct: {sigma.shape}')

    # Check range [0, 1]
    min_val = sigma.min().item()
    max_val = sigma.max().item()
    assert 0.0 <= min_val <= 1.0, f"Min value {min_val} out of range [0, 1]"
    assert 0.0 <= max_val <= 1.0, f"Max value {max_val} out of range [0, 1]"
    print(f'    ✓ Values in range [0, 1]: min={min_val:.4f}, max={max_val:.4f}')

    # Check diagonal (intra-class similarity should be high, close to 1.0)
    diagonal_vals = torch.diagonal(sigma)
    diag_mean = diagonal_vals.mean().item()
    diag_min = diagonal_vals.min().item()
    print(f'    ✓ Diagonal (intra-class) similarity: mean={diag_mean:.4f}, min={diag_min:.4f}')

    # Check symmetry (should be symmetric since similarity is symmetric)
    symmetry_error = (sigma - sigma.T).abs().max().item()
    print(f'    ✓ Symmetry error: {symmetry_error:.6f} (should be ~0)')

    # Statistics
    mean_val = sigma.mean().item()
    std_val = sigma.std().item()
    print(f'    • Overall statistics: mean={mean_val:.4f}, std={std_val:.4f}')

    # Off-diagonal statistics (inter-class similarity)
    mask = ~torch.eye(num_classes, dtype=bool)
    off_diag = sigma[mask]
    off_diag_mean = off_diag.mean().item()
    off_diag_std = off_diag.std().item()
    print(f'    • Off-diagonal (inter-class): mean={off_diag_mean:.4f}, std={off_diag_std:.4f}')

    print('==> Verification complete!')


def visualize_similarity_matrix(sigma, output_path):
    """
    Create and save a heatmap visualization of the similarity matrix.
    """
    print('\n==> Creating visualization...')

    plt.figure(figsize=(12, 10))
    sns.heatmap(sigma.numpy(),
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
                cbar_kws={'label': 'Similarity'},
                xticklabels=10,  # Show every 10th label
                yticklabels=10)

    plt.title('CIFAR100 Class Similarity Matrix (100x100)', fontsize=14)
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Class Index', fontsize=12)
    plt.tight_layout()

    # Save with .png extension
    viz_path = output_path.replace('.npy', '_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f'    Visualization saved to: {viz_path}')
    plt.close()


def save_matrix(sigma, output_path):
    """
    Save the similarity matrix to disk in multiple formats.
    """
    print('\n==> Saving similarity matrix...')

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Save as numpy array (.npy)
    np.save(output_path, sigma.numpy())
    print(f'    Saved as NumPy array: {output_path}')

    # Save as PyTorch tensor (.pt)
    pt_path = output_path.replace('.npy', '.pt')
    torch.save(sigma, pt_path)
    print(f'    Saved as PyTorch tensor: {pt_path}')

    print('==> Saving complete!')


def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'Computing CIFAR100 Class Similarity Matrix')
    print(f'{"="*60}')
    print(f'Device: {device}')
    print(f'Teacher model: {args.model_name}')
    print(f'Checkpoint: {args.teacher_path}')
    print(f'Output: {args.output_path}')
    print(f'{"="*60}\n')

    # Load data
    train_loader, train_set = get_cifar100_train_loader(
        args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f'Dataset: CIFAR100 Training Set')
    print(f'Total samples: {len(train_set)}')
    print(f'Batch size: {args.batch_size}')

    # Load teacher model
    model = load_teacher_model(args.teacher_path, args.model_name, device=device)

    # Extract features
    features, labels = extract_features(model, train_loader, device=device)

    # Organize by class
    class_features = organize_features_by_class(features, labels, num_classes=100)

    # Compute similarity matrix
    sigma = compute_class_similarity_matrix(class_features, num_classes=100, device=device)

    # Verify matrix properties
    verify_matrix(sigma, num_classes=100)

    # Save matrix
    save_matrix(sigma, args.output_path)

    # Visualize if requested
    if args.visualize:
        visualize_similarity_matrix(sigma, args.output_path)

    print(f'\n{"="*60}')
    print(f'Class similarity matrix computation completed successfully!')
    print(f'Matrix saved to: {args.output_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
