# Out-of-Distribution (OOD) Detection Evaluation - Detailed Explanation

## Table of Contents
1. [Overview](#overview)
2. [OOD Detection Concept](#ood-detection-concept)
3. [Preprocessing Details by Dataset](#preprocessing-details-by-dataset)
4. [Scoring Functions](#scoring-functions)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Implementation Details](#implementation-details)
7. [Usage Examples](#usage-examples)

---

## Overview

**File**: `eval_ood.py`

This script evaluates trained student models on Out-of-Distribution (OOD) detection tasks. It measures how well a model can distinguish between:
- **In-Distribution (ID)**: Data from the same distribution as training (e.g., CIFAR-100)
- **Out-of-Distribution (OOD)**: Data from different distributions (e.g., LSUN, SVHN, DTD)

**Key Metrics**:
- **AUROC** (Area Under ROC Curve): Overall detection performance (0-100%)
- **FPR95** (False Positive Rate at 95% TPR): Error rate when detecting 95% of ID samples

---

## OOD Detection Concept

### What is OOD Detection?

A model trained on CIFAR-100 should:
- Produce **high confidence** on CIFAR-100 test images (in-distribution)
- Produce **low confidence** on unfamiliar images like textures, scenes, or other datasets (out-of-distribution)

### Why is it Important?

In real-world deployments, models encounter data they weren't trained on. Good OOD detection prevents:
- Overconfident predictions on unfamiliar inputs
- Deployment failures in safety-critical applications
- Silently wrong predictions

### How Does it Work?

1. **Compute confidence scores** for both ID and OOD data
2. **Set a threshold**: Below threshold = OOD, above = ID
3. **Measure performance**: How well does this threshold separate the two distributions?

---

## Preprocessing Details by Dataset

All datasets are preprocessed to match the **CIFAR-100 training format**:
- **Target size**: 32×32 pixels
- **Normalization**: Mean=[0.5071, 0.4867, 0.4408], Std=[0.2675, 0.2565, 0.2761]

### 1. CIFAR-100 (In-Distribution)

**Purpose**: Baseline test set (what the model was trained on)

**Location**: `../data/` (auto-downloaded)

**Preprocessing**:
```python
transforms.Compose([
    transforms.ToTensor(),           # Convert PIL Image to tensor [0,1]
    transforms.Normalize(            # Standardize to match training
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: 32×32 (native)
- **No resizing needed**
- Uses CIFAR-100 test split (10,000 samples)
- 100 classes of objects

---

### 2. CIFAR-10 (OOD)

**Purpose**: Similar visual domain but different classes

**Location**: `../data/` (auto-downloaded)

**Preprocessing**:
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 stats
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: 32×32 (native)
- **No resizing needed**
- Uses CIFAR-10 test split (10,000 samples)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Note**: Uses CIFAR-100 normalization (not CIFAR-10's) for consistency

**Why OOD?** Different class distribution (10 vs 100 classes), some overlap but not identical

---

### 3. SVHN (Street View House Numbers) (OOD)

**Purpose**: Real-world digit images (very different domain)

**Location**: `../data/` (auto-downloaded)

**Preprocessing**:
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: 32×32 (native)
- **No resizing needed**
- Uses SVHN test split (~26,032 samples)
- Contains house numbers from Google Street View
- 10 digit classes (0-9)

**Why OOD?** Completely different domain (digits vs objects), different image characteristics (real photos vs object images)

---

### 4. Tiny-ImageNet-200 (OOD)

**Purpose**: Natural images with higher resolution (downsampled)

**Location**: `../data/tiny-imagenet-200/val/`

**Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(32),           # Resize from 64×64
    transforms.CenterCrop(32),       # Ensure exact 32×32
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: 64×64
- **Resizing**: Bilinear interpolation to 32×32
- **Center crop**: Ensures exactly 32×32 if aspect ratio varies
- 200 classes (subset of ImageNet)
- ~10,000 validation images

**Why OOD?** Different classes, different original resolution, subset of ImageNet

**File Structure**:
- Option 1: `tiny-imagenet-200/val/` with class subdirectories (ImageFolder)
- Option 2: Flat structure with all `.JPEG` files (TinyImageNetDataset)

---

### 5. LSUN (Large-scale Scene Understanding) (OOD)

**Purpose**: Scene images (very different from objects)

**Location**:
- `../data/LSUN/LSUN_pil/` (PNG files) **[Primary]**
- `../data/LSUN/` (LMDB format) [Fallback]

**Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(32),           # Resize from variable sizes
    transforms.CenterCrop(32),       # Ensure exact 32×32
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: Variable (often 256×256 or larger)
- **Resizing**: Scales to 32 on shorter side
- **Center crop**: Takes center 32×32 region
- Scene categories: bedroom, living room, church, etc.
- **Custom modification**: Supports PNG files (filters out macOS metadata files `._*`)

**Why OOD?** Scenes vs objects, very different visual characteristics

**File Formats**:
1. **PNG files** (custom): Individual `.png` files in `LSUN_pil/`
2. **LMDB** (original): Database format with `test_lmdb`

---

### 6. DTD (Describable Textures Dataset) (OOD)

**Purpose**: Texture images (minimal semantic content)

**Location**: `../data/DTD/dtd/images/`

**Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
})
```

**Details**:
- **Original size**: Variable (typically larger)
- **Resizing**: Scales to 32 on shorter side
- **Center crop**: Takes center 32×32 region
- 47 texture categories: banded, blotchy, braided, bubbly, bumpy, chequered, etc.
- ~5,640 images total

**Why OOD?** Pure textures without objects, fundamentally different visual features

**File Structure**: ImageFolder format with subdirectories per texture class

---

### 7. Places (OOD)

**Purpose**: Scene recognition (overlaps with LSUN but larger)

**Location**: `../data/Places/test_256/`

**Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: 256×256
- **Resizing**: Scales down to 32×32
- 365 scene categories
- Large-scale scene dataset

**Why OOD?** Scene-centric (not object-centric), different visual statistics

**File Structure**: Flat directory with all `.jpg` files

---

### 8. Human Detection Dataset (OOD)

**Purpose**: Binary classification dataset (human presence)

**Location**: `../data/human detection dataset/`

**Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])
```

**Details**:
- **Original size**: Variable
- **Resizing**: Scales to 32×32
- 2 classes: 0 (no human), 1 (human present)

**Why OOD?** Different task (detection vs classification), different domain

**File Structure**: Two subdirectories `0/` and `1/` with images

---

## Preprocessing Summary Table

| Dataset | Original Size | Resize? | Center Crop? | Normalization | Notes |
|---------|--------------|---------|--------------|---------------|-------|
| **CIFAR-100** (ID) | 32×32 | ❌ | ❌ | CIFAR-100 | Native format |
| **CIFAR-10** | 32×32 | ❌ | ❌ | CIFAR-100 | Uses CIFAR-100 stats! |
| **SVHN** | 32×32 | ❌ | ❌ | CIFAR-100 | Native format |
| **Tiny-ImageNet** | 64×64 | ✅ → 32 | ✅ | CIFAR-100 | Downsampled |
| **LSUN** | Variable | ✅ → 32 | ✅ | CIFAR-100 | Supports PNG/LMDB |
| **DTD** | Variable | ✅ → 32 | ✅ | CIFAR-100 | Textures |
| **Places** | 256×256 | ✅ → 32 | ✅ | CIFAR-100 | Large scenes |
| **Human Detection** | Variable | ✅ → 32 | ✅ | CIFAR-100 | Binary task |

**Key Points**:
- All datasets use **CIFAR-100 normalization** for consistency
- Resizing uses **bilinear interpolation** (PyTorch default)
- Center crop ensures **exact 32×32** output
- Normalization formula: `output = (input - mean) / std`

---

## Scoring Functions

Three methods to quantify model confidence:

### 1. MSP (Maximum Softmax Probability) [Default]

**Formula**:
```
MSP(x) = max(softmax(f(x)))
```

**How it works**:
1. Forward pass: Get logits `z = f(x)`
2. Apply softmax: `p = softmax(z) = exp(z_i) / Σ exp(z_j)`
3. Take maximum: `MSP = max(p)`

**Interpretation**:
- **High MSP** (close to 1.0): Model is confident → Likely **ID**
- **Low MSP** (close to 0.0): Model is uncertain → Likely **OOD**

**Example**:
```python
Logits: [2.5, 1.2, 0.8, ...]
Softmax: [0.82, 0.13, 0.05, ...]
MSP: 0.82  # Model is 82% confident in top class
```

**Implementation** (`eval_ood.py:465-481`):
```python
def compute_msp_scores(model, data_loader, device):
    all_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)      # Softmax normalization
            max_probs, _ = torch.max(probs, dim=1)  # Maximum probability
            all_scores.extend(max_probs.cpu().numpy())
    return np.array(all_scores)
```

**Pros**: Simple, fast, widely used baseline

**Cons**: Can be overconfident on OOD data

---

### 2. Energy Score

**Formula**:
```
Energy(x) = -T * log(Σ exp(f_i(x) / T))
```

where `T` is temperature (default=1.0)

**How it works**:
1. Forward pass: Get logits `z = f(x)`
2. Scale by temperature: `z / T`
3. Compute log-sum-exp: `-T * log(Σ exp(z_i / T))`

**Interpretation**:
- **Low energy** (negative, large magnitude): High confidence → Likely **ID**
- **High energy** (closer to 0): Low confidence → Likely **OOD**

**Example**:
```python
Logits: [2.5, 1.2, 0.8, ...]
Energy: -2.67  # Low energy = confident = ID

Logits: [0.1, 0.05, 0.08, ...]
Energy: -0.15  # High energy = uncertain = OOD
```

**Implementation** (`eval_ood.py:484-500`):
```python
def compute_energy_scores(model, data_loader, device, temperature=1.0):
    all_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            # Energy = -T * logsumexp(logits/T)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            all_scores.extend(energy.cpu().numpy())
    return np.array(all_scores)
```

**Pros**: More theoretically grounded, often better than MSP

**Cons**: Requires tuning temperature for best performance

---

### 3. ODIN (Out-of-DIstributioN detector)

**Formula**:
```
ODIN(x) = max(softmax(f(x) / T))
```

with temperature scaling `T` (default=1000.0)

**How it works**:
1. Forward pass: Get logits `z = f(x)`
2. **Temperature scaling**: Divide by large T (e.g., 1000)
3. Apply softmax and take max

**Interpretation**:
- Temperature scaling **sharpens** the difference between ID and OOD
- Makes ID predictions more confident, OOD less confident

**Example**:
```python
Logits: [2.5, 1.2, 0.8]

Without temperature (T=1):
Softmax: [0.82, 0.13, 0.05]
MSP: 0.82

With temperature (T=1000):
Softmax: [0.334, 0.333, 0.333]  # Much flatter!
MSP: 0.334  # Lower confidence
```

**Implementation** (`eval_ood.py:503-518`):
```python
def compute_odin_scores(model, data_loader, device, temperature=1000.0):
    all_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            # Temperature-scaled softmax
            probs = F.softmax(logits / temperature, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_scores.extend(max_probs.cpu().numpy())
    return np.array(all_scores)
```

**Note**: This is simplified ODIN without input perturbation (full ODIN adds gradient-based noise)

**Pros**: Better separation with temperature tuning

**Cons**: Requires careful temperature selection, slower if using full perturbation

---

## Evaluation Metrics

### 1. AUROC (Area Under the Receiver Operating Characteristic Curve)

**Range**: 0% to 100% (higher is better)

**What it measures**: Overall ability to distinguish ID from OOD across all thresholds

**Perfect score**: 100% (perfect separation)
**Random guessing**: 50%

**Interpretation**:
- **90-100%**: Excellent detection
- **80-90%**: Good detection
- **70-80%**: Fair detection
- **50-70%**: Poor detection
- **< 50%**: Worse than random

**Calculation**:
```python
def calculate_auroc(in_scores, out_scores):
    # Labels: 1=ID, 0=OOD
    y_true = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])

    auroc = roc_auc_score(y_true, y_score)
    return auroc
```

**Visual**:
```
ROC Curve plots TPR vs FPR for all thresholds
AUROC = Area under this curve

     TPR
      ^
   1.0|    ___---
      |  _/
      | /
      |/
   0.0+---------> FPR
      0.0      1.0

Larger area = better OOD detection
```

---

### 2. FPR95 (False Positive Rate at 95% True Positive Rate)

**Range**: 0% to 100% (lower is better)

**What it measures**: How many OOD samples are incorrectly classified as ID when we detect 95% of ID samples

**Perfect score**: 0% (no false alarms)
**Worst score**: 100% (all OOD classified as ID)

**Interpretation**:
- **0-10%**: Excellent (< 10% false alarms)
- **10-30%**: Good
- **30-50%**: Fair
- **> 50%**: Poor (too many false alarms)

**Calculation**:
```python
def calculate_fpr95(in_scores, out_scores):
    y_true = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find FPR when TPR = 0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return fpr95
```

**Why TPR=95%?**: Industry standard - we want to catch 95% of ID samples while minimizing false alarms

**Example**:
```
Setting: Detect 95% of CIFAR-100 test images correctly
FPR95 = 20% means:
- We correctly flag 9,500 / 10,000 CIFAR-100 images as ID
- But we also incorrectly flag 20% of OOD images as ID
```

---

### Metric Comparison

| Metric | Better Value | What it Measures | Use Case |
|--------|-------------|------------------|----------|
| **AUROC** | Higher (→100%) | Overall separation | General performance |
| **FPR95** | Lower (→0%) | Practical error rate | Real-world deployment |

**Which to use?**
- **AUROC**: Compare different methods overall
- **FPR95**: Evaluate real-world deployment readiness

---

## Implementation Details

### Custom Datasets

#### 1. ImageFolderDataset
Loads PNG files from a flat directory (used for LSUN)

```python
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter out macOS metadata files
        self.images = sorted([f for f in os.listdir(root_dir)
                            if f.endswith('.png')
                            and not f.startswith('._')
                            and not f.startswith('.')])
```

**Key features**:
- Filters hidden files (`.` prefix)
- Filters macOS metadata (`._` prefix)
- Returns dummy label 0 (labels not needed for OOD)

#### 2. TinyImageNetDataset
Handles Tiny-ImageNet's flat validation structure

```python
class TinyImageNetDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.images = [f for f in os.listdir(img_dir)
                      if f.endswith('.JPEG') or f.endswith('.jpg')]
```

#### 3. PlacesDataset
Handles Places365 test set

```python
class PlacesDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.images = [f for f in os.listdir(img_dir)
                      if f.endswith('.jpg') or f.endswith('.png')]
```

#### 4. HumanDetectionDataset
Loads from binary class subdirectories

```python
class HumanDetectionDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        # Load from subdirectories: 0/ and 1/
        for label in [0, 1]:
            class_dir = os.path.join(data_folder, str(label))
            # ... load images
```

---

### File Structure Requirements

```
../data/
├── cifar-100-python/           # Auto-downloaded
├── cifar-10-batches-py/        # Auto-downloaded
├── SVHN/                       # Auto-downloaded
├── tiny-imagenet-200/
│   └── val/
│       ├── images/             # Flat structure
│       └── [class folders]/    # OR class subdirectories
├── LSUN/
│   ├── LSUN_pil/              # PNG files (custom)
│   │   ├── correct_resize_0.png
│   │   ├── correct_resize_1.png
│   │   └── ...
│   └── test_lmdb/             # OR LMDB format (original)
├── DTD/
│   └── dtd/
│       └── images/
│           ├── banded/
│           ├── blotchy/
│           └── ... (47 texture classes)
├── Places/
│   └── test_256/
│       ├── image_00001.jpg
│       └── ...
└── human detection dataset/
    ├── 0/                      # No human
    │   └── *.png
    └── 1/                      # Human present
        └── *.png
```

---

## Usage Examples

### Basic Usage

```bash
python eval_ood.py \
    --model_path save/students/models/S_resnet8x4_T_resnet32x4_cifar100_kd/resnet8x4_best.pth \
    --model_s resnet8x4 \
    --ood_dataset lsun
```

### All Parameters

```bash
python eval_ood.py \
    --model_path <path_to_checkpoint.pth> \
    --model_s resnet8x4 \
    --ood_dataset lsun \
    --in_dataset cifar100 \
    --score_func msp \
    --batch_size 128 \
    --num_workers 4 \
    --gpu_id 0 \
    --data_folder ../data
```

### Using Different Scoring Functions

#### MSP (default)
```bash
python eval_ood.py --model_path <path> --model_s resnet8x4 --ood_dataset svhn --score_func msp
```

#### Energy Score
```bash
python eval_ood.py --model_path <path> --model_s resnet8x4 --ood_dataset svhn --score_func energy
```

#### ODIN with Custom Temperature
```bash
python eval_ood.py --model_path <path> --model_s resnet8x4 --ood_dataset svhn \
    --score_func odin --temperature 1000
```

### Batch Evaluation (All Datasets)

**Windows**:
```batch
eval_all_ood.bat save\students\models\S~resnet8x4_T~resnet32x4_cifar100_kd\resnet8x4_best.pth
```

**Linux/Mac**:
```bash
./eval_all_ood.sh save/students/models/S:resnet8x4_T:resnet32x4_cifar100_kd/resnet8x4_best.pth
```

### Programmatic Usage

```python
from eval_ood import evaluate_ood_detection, parse_option
import argparse

# Setup options
parser = argparse.ArgumentParser()
# ... add arguments ...
opt = parser.parse_args([
    '--model_path', 'save/students/models/.../best.pth',
    '--model_s', 'resnet8x4',
    '--ood_dataset', 'lsun'
])

# Run evaluation
auroc, fpr95 = evaluate_ood_detection(opt)
print(f"AUROC: {auroc*100:.2f}%, FPR95: {fpr95*100:.2f}%")
```

---

## Output Format

### Console Output

```
================================================================================
OOD DETECTION EVALUATION
================================================================================
Model: resnet8x4
Checkpoint: save/students/models/S_resnet8x4_T_resnet32x4_cifar100_kd/resnet8x4_best.pth
In-distribution dataset: cifar100
OOD dataset: lsun
Scoring function: msp
Device: cuda
================================================================================

==> Loading student model from ...
==> Loaded model with best accuracy: 73.09
==> Student model loaded successfully

==> Loading in-distribution (ID) test data...
ID dataset loaded: 10000 samples

==> Loading out-of-distribution (OOD) data: lsun...
Loading LSUN from PNG files: ../data\LSUN\LSUN_pil
OOD dataset loaded: 10000 samples

==> Computing MSP scores for ID data...
==> Computing MSP scores for OOD data...

==> Calculating OOD detection metrics...

================================================================================
RESULTS
================================================================================
AUROC: 67.98%
FPR95: 90.37%
================================================================================

Score Statistics:
ID scores  - Mean: 0.8014, Std: 0.2317, Min: 0.0797, Max: 1.0000
OOD scores - Mean: 0.6682, Std: 0.2386, Min: 0.1039, Max: 1.0000

==> Results saved to: save/students/models/.../ood_results/ood_lsun_msp.txt
```

### Saved Files

**Text results**: `ood_results/ood_lsun_msp.txt`
```
================================================================================
OOD DETECTION EVALUATION RESULTS
================================================================================
Model: resnet8x4
Checkpoint: save/students/models/.../resnet8x4_best.pth
In-distribution dataset: cifar100
OOD dataset: lsun
Scoring function: msp
Temperature: 1.0
================================================================================

METRICS:
AUROC: 67.98%
FPR95: 90.37%

SCORE STATISTICS:
ID scores  - Mean: 0.8014, Std: 0.2317
OOD scores - Mean: 0.6682, Std: 0.2386
```

**NumPy arrays** (for further analysis):
- `ood_results/id_scores_lsun_msp.npy`
- `ood_results/ood_scores_lsun_msp.npy`

Load with: `scores = np.load('id_scores_lsun_msp.npy')`

---

## Interpreting Results

### Good OOD Detection Example

```
AUROC: 95.32%
FPR95: 8.45%

ID scores  - Mean: 0.9234, Std: 0.0821
OOD scores - Mean: 0.4512, Std: 0.1932
```

**Interpretation**:
- ✅ **AUROC 95.32%**: Excellent separation
- ✅ **FPR95 8.45%**: Only 8.45% false alarms
- ✅ **Score gap**: ID mean (0.92) >> OOD mean (0.45)
- ✅ **Low ID std**: Model consistently confident on ID
- **Conclusion**: Model reliably detects OOD samples

### Poor OOD Detection Example

```
AUROC: 58.23%
FPR95: 87.32%

ID scores  - Mean: 0.7123, Std: 0.2456
OOD scores - Mean: 0.6834, Std: 0.2301
```

**Interpretation**:
- ❌ **AUROC 58.23%**: Barely better than random (50%)
- ❌ **FPR95 87.32%**: Most OOD samples misclassified as ID
- ❌ **Score overlap**: ID mean (0.71) ≈ OOD mean (0.68)
- ❌ **High std both**: Inconsistent confidence
- **Conclusion**: Model cannot distinguish ID from OOD

### Typical Results by Dataset (ResNet8x4 on CIFAR-100)

| OOD Dataset | Expected AUROC | Expected FPR95 | Difficulty |
|-------------|---------------|----------------|------------|
| **CIFAR-10** | 70-80% | 50-70% | Medium (similar domain) |
| **SVHN** | 85-95% | 10-30% | Easy (very different) |
| **Tiny-ImageNet** | 75-85% | 30-50% | Medium |
| **LSUN** | 65-75% | 60-90% | Hard (scenes vs objects) |
| **DTD** | 80-90% | 20-40% | Easy (pure textures) |
| **Places** | 65-75% | 60-85% | Hard (scenes) |

---

## Common Issues and Solutions

### 1. Dataset Not Found

**Error**: `ValueError: LSUN dataset not found: ../data/LSUN`

**Solutions**:
- Check dataset path matches expected structure
- Verify folder name (case-sensitive on Linux/Mac)
- For LSUN: ensure either `LSUN_pil/` or `test_lmdb/` exists

### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python eval_ood.py ... --batch_size 64

# Reduce number of workers
python eval_ood.py ... --num_workers 2
```

### 3. macOS Metadata Files

**Error**: `PIL.UnidentifiedImageError: cannot identify image file '._image.png'`

**Solution**: Already handled! Code filters out `._` files automatically.

### 4. Wrong Image Format

**Error**: Images not loading from Tiny-ImageNet

**Solution**: Code tries both ImageFolder and flat structure automatically.

### 5. Normalization Mismatch

**Issue**: Poor performance due to wrong normalization

**Solution**: All datasets use CIFAR-100 stats:
```python
mean=[0.5071, 0.4867, 0.4408]
std=[0.2675, 0.2565, 0.2761]
```

---

## Advanced Topics

### Custom OOD Dataset

To add a new OOD dataset:

1. **Create loader function**:
```python
def get_my_dataset_loader(data_folder, batch_size, num_workers):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(
        root=os.path.join(data_folder, 'MyDataset'),
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, ...)
```

2. **Register in dispatcher**:
```python
def get_ood_loader(ood_dataset, ...):
    if ood_dataset == 'my-dataset':
        return get_my_dataset_loader(...)
    # ... existing cases
```

3. **Add to argument choices**:
```python
parser.add_argument('--ood_dataset', choices=[..., 'my-dataset'])
```

### Ensemble Scoring

Combine multiple scoring functions:

```python
# Compute all scores
msp_scores = compute_msp_scores(model, loader, device)
energy_scores = compute_energy_scores(model, loader, device)

# Normalize to [0,1]
msp_norm = (msp_scores - msp_scores.min()) / (msp_scores.max() - msp_scores.min())
energy_norm = (energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min())

# Ensemble: average or max
ensemble_scores = (msp_norm + energy_norm) / 2
```

### Threshold Selection

Find optimal threshold for deployment:

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Maximize TPR - FPR
idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"TPR: {tpr[idx]:.2%}, FPR: {fpr[idx]:.2%}")
```

---

## References

### Papers

1. **MSP**: Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017

2. **ODIN**: Liang et al., "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks", ICLR 2018

3. **Energy**: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020

### Datasets

- **CIFAR-10/100**: Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009
- **SVHN**: Netzer et al., "Reading Digits in Natural Images with Unsupervised Feature Learning", NIPS 2011
- **Tiny-ImageNet**: Stanford CS231n course
- **LSUN**: Yu et al., "LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop", arXiv 2015
- **DTD**: Cimpoi et al., "Describing Textures in the Wild", CVPR 2014
- **Places**: Zhou et al., "Places: A 10 million Image Database for Scene Recognition", TPAMI 2017

---

## Summary

This evaluation script provides:

✅ **Comprehensive OOD detection** across 7 OOD datasets
✅ **Standardized preprocessing** (all to 32×32, CIFAR-100 normalization)
✅ **Multiple scoring functions** (MSP, Energy, ODIN)
✅ **Standard metrics** (AUROC, FPR95)
✅ **Flexible data formats** (ImageFolder, flat directories, LMDB, PNG)
✅ **Batch evaluation** scripts for convenience
✅ **Detailed logging** and result saving

**Key Insight**: All OOD datasets are preprocessed identically (resize to 32×32, CIFAR-100 normalization) to ensure fair comparison and match the model's training distribution.
