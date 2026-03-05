"""
OOD Detection Evaluation Script
Evaluates trained student models on Out-of-Distribution detection tasks
Uses AUROC and FPR95 metrics with Maximum Softmax Probability (MSP) as the OOD score
"""

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from models import model_dict

split_symbol = '~' if os.name == 'nt' else ':'


class ImageFolderDataset(Dataset):
    """Custom dataset to load images from a folder of PNG files"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter out hidden files (starting with .) and macOS metadata files (._)
        self.images = sorted([f for f in os.listdir(root_dir)
                            if f.endswith('.png') and not f.startswith('._') and not f.startswith('.')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return dummy label (0) since we only care about images for OOD detection
        return image, 0


def parse_option():
    parser = argparse.ArgumentParser('OOD Detection Evaluation')

    # Model and checkpoint
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained student model checkpoint')
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        help='Student model architecture')

    # Dataset selection
    parser.add_argument('--ood_dataset', type=str, required=True,
                        choices=['cifar10', 'tiny-imagenet', 'human-detection', 'dtd', 'svhn', 'places', 'lsun'],
                        help='OOD dataset to evaluate on')
    parser.add_argument('--in_dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet'],
                        help='In-distribution dataset (what model was trained on)')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')

    # Data paths
    parser.add_argument('--data_folder', type=str, default='../data',
                        help='Path to data folder')

    # OOD detection method
    parser.add_argument('--score_func', type=str, default='msp',
                        choices=['msp', 'energy', 'odin'],
                        help='Scoring function for OOD detection')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling for ODIN')

    # Output settings
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output (only model name, accuracy, and metrics)')

    opt = parser.parse_args()

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return opt


def load_student_model(model_path, model_name, num_classes, device, quiet=False):
    """Load trained student model from checkpoint"""
    if not quiet:
        print(f'==> Loading student model from {model_path}')

    model = model_dict[model_name](num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    best_acc = None
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint.get('best_acc', None)
        if not quiet:
            print(f"==> Loaded model with best accuracy: {best_acc if best_acc else 'N/A'}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    if not quiet:
        print('==> Student model loaded successfully')

    return model, best_acc


def get_cifar100_test_loader(data_folder, batch_size, num_workers):
    """Get CIFAR-100 test dataloader (in-distribution)"""
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.CIFAR100(
        root=data_folder,
        train=False,
        transform=transform,
        download=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def get_cifar10_loader(data_folder, batch_size, num_workers):
    """Get CIFAR-10 dataloader (OOD for CIFAR-100 trained models)"""
    # Use same normalization as CIFAR-100 for consistency
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    ood_dataset = datasets.CIFAR10(
        root=data_folder,
        train=False,
        transform=transform,
        download=True
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_tiny_imagenet_loader(data_folder, batch_size, num_workers):
    """Get Tiny-ImageNet-200 dataloader (OOD)"""
    # Resize to 32x32 to match CIFAR-100 input size
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    val_dir = os.path.join(data_folder, 'tiny-imagenet-200', 'val', 'images')

    if not os.path.exists(val_dir):
        raise ValueError(f"Tiny-ImageNet validation directory not found: {val_dir}")

    ood_dataset = datasets.ImageFolder(
        root=os.path.join(data_folder, 'tiny-imagenet-200', 'val'),
        transform=transform
    )

    # If ImageFolder doesn't work (flat structure), use custom approach
    if len(ood_dataset) == 0:
        ood_dataset = TinyImageNetDataset(val_dir, transform)

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Custom dataset for Tiny-ImageNet validation set with flat structure"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.JPEG') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Dummy label


class HumanDetectionDataset(torch.utils.data.Dataset):
    """Custom dataset for Human Detection dataset"""
    def __init__(self, data_folder, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images from both classes
        for label in [0, 1]:
            class_dir = os.path.join(data_folder, str(label))
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class PlacesDataset(torch.utils.data.Dataset):
    """Custom dataset for Places365 test set with flat structure"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Dummy label


def get_human_detection_loader(data_folder, batch_size, num_workers):
    """Get Human Detection dataset dataloader (OOD)"""
    # Resize to 32x32 to match CIFAR-100
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_path = os.path.join(data_folder, 'human detection dataset')

    if not os.path.exists(dataset_path):
        raise ValueError(f"Human Detection dataset not found: {dataset_path}")

    ood_dataset = HumanDetectionDataset(dataset_path, transform)

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_dtd_loader(data_folder, batch_size, num_workers):
    """Get DTD (Describable Textures Dataset) dataloader (OOD)"""
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_path = os.path.join(data_folder, 'DTD', 'dtd', 'images')

    if not os.path.exists(dataset_path):
        raise ValueError(f"DTD dataset not found: {dataset_path}")

    ood_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_svhn_loader(data_folder, batch_size, num_workers):
    """Get SVHN (Street View House Numbers) dataloader (OOD)"""
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    ood_dataset = datasets.SVHN(
        root=data_folder,
        split='test',
        transform=transform,
        download=True
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_places_loader(data_folder, batch_size, num_workers):
    """Get Places dataset dataloader (OOD)"""
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_path = os.path.join(data_folder, 'Places', 'test_256')

    if not os.path.exists(dataset_path):
        raise ValueError(f"Places dataset not found: {dataset_path}")

    ood_dataset = PlacesDataset(dataset_path, transform)

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_lsun_loader(data_folder, batch_size, num_workers):
    """Get LSUN dataset dataloader (OOD)"""
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    # Look for LSUN_pil folder first (PNG files), then fall back to LSUN (LMDB format)
    lsun_pil_path = os.path.join(data_folder, 'LSUN', 'LSUN_pil')
    lsun_lmdb_path = os.path.join(data_folder, 'LSUN')

    if os.path.exists(lsun_pil_path):
        # Use custom dataset for PNG files
        ood_dataset = ImageFolderDataset(
            root_dir=lsun_pil_path,
            transform=transform
        )
    elif os.path.exists(lsun_lmdb_path):
        # Use torchvision LSUN dataset for LMDB format
        ood_dataset = datasets.LSUN(
            root=lsun_lmdb_path,
            classes='test',
            transform=transform
        )
    else:
        raise ValueError(f"LSUN dataset not found in either {lsun_pil_path} or {lsun_lmdb_path}")

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return ood_loader


def get_ood_loader(ood_dataset, data_folder, batch_size, num_workers):
    """Get appropriate OOD dataloader based on dataset choice"""
    if ood_dataset == 'cifar10':
        return get_cifar10_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'tiny-imagenet':
        return get_tiny_imagenet_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'human-detection':
        return get_human_detection_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'dtd':
        return get_dtd_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'svhn':
        return get_svhn_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'places':
        return get_places_loader(data_folder, batch_size, num_workers)
    elif ood_dataset == 'lsun':
        return get_lsun_loader(data_folder, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown OOD dataset: {ood_dataset}")


def compute_msp_scores(model, data_loader, device):
    """
    Compute Maximum Softmax Probability (MSP) scores
    Higher MSP = more confident = likely in-distribution
    Lower MSP = less confident = likely OOD
    """
    all_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_scores.extend(max_probs.cpu().numpy())

    return np.array(all_scores)


def compute_energy_scores(model, data_loader, device, temperature=1.0):
    """
    Compute Energy scores
    Energy = -T * log(sum(exp(logits/T)))
    Lower energy = more confident = likely in-distribution
    Higher energy = less confident = likely OOD
    """
    all_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            all_scores.extend(energy.cpu().numpy())

    return np.array(all_scores)


def compute_odin_scores(model, data_loader, device, temperature=1000.0, epsilon=0.0014):
    """
    Compute ODIN scores with input preprocessing and temperature scaling
    Note: This is simplified ODIN without input perturbation
    """
    all_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits / temperature, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_scores.extend(max_probs.cpu().numpy())

    return np.array(all_scores)


def calculate_auroc(in_scores, out_scores):
    """Calculate AUROC score"""
    # Create labels: 1 for in-distribution, 0 for OOD
    y_true = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])

    auroc = roc_auc_score(y_true, y_score)
    return auroc


def calculate_fpr95(in_scores, out_scores):
    """Calculate FPR at 95% TPR"""
    # Create labels: 1 for in-distribution, 0 for OOD
    y_true = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find FPR at 95% TPR
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return fpr95


def evaluate_ood_detection(opt):
    """Main OOD detection evaluation function"""

    if not opt.quiet:
        print("=" * 80)
        print("OOD DETECTION EVALUATION")
        print("=" * 80)
        print(f"Model: {opt.model_s}")
        print(f"Checkpoint: {opt.model_path}")
        print(f"In-distribution dataset: {opt.in_dataset}")
        print(f"OOD dataset: {opt.ood_dataset}")
        print(f"Scoring function: {opt.score_func}")
        print(f"Device: {opt.device}")
        print("=" * 80)

    # Determine number of classes based on in-distribution dataset
    if opt.in_dataset == 'cifar100':
        num_classes = 100
    elif opt.in_dataset == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError(f"Unknown in-distribution dataset: {opt.in_dataset}")

    # Load model
    model, best_acc = load_student_model(opt.model_path, opt.model_s, num_classes, opt.device, quiet=opt.quiet)

    # Get in-distribution test loader
    if not opt.quiet:
        print("\n==> Loading in-distribution (ID) test data...")
    if opt.in_dataset == 'cifar100':
        id_loader = get_cifar100_test_loader(opt.data_folder, opt.batch_size, opt.num_workers)
    else:
        raise NotImplementedError(f"In-distribution dataset {opt.in_dataset} not yet implemented")

    if not opt.quiet:
        print(f"ID dataset loaded: {len(id_loader.dataset)} samples")

    # Get OOD loader
    if not opt.quiet:
        print(f"\n==> Loading out-of-distribution (OOD) data: {opt.ood_dataset}...")
    ood_loader = get_ood_loader(opt.ood_dataset, opt.data_folder, opt.batch_size, opt.num_workers)
    if not opt.quiet:
        print(f"OOD dataset loaded: {len(ood_loader.dataset)} samples")

    # Compute scores
    if not opt.quiet:
        print(f"\n==> Computing {opt.score_func.upper()} scores for ID data...")
    if opt.score_func == 'msp':
        id_scores = compute_msp_scores(model, id_loader, opt.device)
    elif opt.score_func == 'energy':
        id_scores = compute_energy_scores(model, id_loader, opt.device, opt.temperature)
    elif opt.score_func == 'odin':
        id_scores = compute_odin_scores(model, id_loader, opt.device, opt.temperature)
    else:
        raise ValueError(f"Unknown scoring function: {opt.score_func}")

    if not opt.quiet:
        print(f"==> Computing {opt.score_func.upper()} scores for OOD data...")
    if opt.score_func == 'msp':
        ood_scores = compute_msp_scores(model, ood_loader, opt.device)
    elif opt.score_func == 'energy':
        ood_scores = compute_energy_scores(model, ood_loader, opt.device, opt.temperature)
    elif opt.score_func == 'odin':
        ood_scores = compute_odin_scores(model, ood_loader, opt.device, opt.temperature)

    # Calculate metrics
    if not opt.quiet:
        print("\n==> Calculating OOD detection metrics...")
    auroc = calculate_auroc(id_scores, ood_scores)
    fpr95 = calculate_fpr95(id_scores, ood_scores)

    # Print results
    if opt.quiet:
        # Minimal output: only essential info
        print(f"{opt.ood_dataset:<20} AUROC: {auroc * 100:6.2f}%  FPR95: {fpr95 * 100:6.2f}%")
    else:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"AUROC: {auroc * 100:.2f}%")
        print(f"FPR95: {fpr95 * 100:.2f}%")
        print("=" * 80)

    # Save results (only in non-quiet mode)
    if not opt.quiet:
        save_results(opt, auroc, fpr95)

    return auroc, fpr95, best_acc


def save_results(opt, auroc, fpr95):
    """Save evaluation results to file"""
    # Create results directory
    results_dir = os.path.join(os.path.dirname(opt.model_path), 'ood_results')
    os.makedirs(results_dir, exist_ok=True)

    # Create results filename
    results_file = os.path.join(results_dir,
                                f'ood_{opt.ood_dataset}_{opt.score_func}.txt')

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OOD DETECTION EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {opt.model_s}\n")
        f.write(f"Checkpoint: {opt.model_path}\n")
        f.write(f"In-distribution dataset: {opt.in_dataset}\n")
        f.write(f"OOD dataset: {opt.ood_dataset}\n")
        f.write(f"Scoring function: {opt.score_func}\n")
        f.write(f"Temperature: {opt.temperature}\n")
        f.write("=" * 80 + "\n\n")

        f.write("METRICS:\n")
        f.write(f"AUROC: {auroc * 100:.2f}%\n")
        f.write(f"FPR95: {fpr95 * 100:.2f}%\n\n")

    print(f"\n==> Results saved to: {results_file}")


if __name__ == '__main__':
    opt = parse_option()
    evaluate_ood_detection(opt)
