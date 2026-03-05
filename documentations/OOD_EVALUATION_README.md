# OOD Detection Evaluation for SimKD

This guide explains how to evaluate trained student models on Out-of-Distribution (OOD) detection tasks.

## Overview

After training a student model with any distillation method (PKT, soft_pkt, SimKD, etc.), you can evaluate its OOD detection performance using three different datasets:

1. **CIFAR-10** - 10 classes, 32x32 images
2. **Tiny-ImageNet-200** - 200 classes, 64x64 images (resized to 32x32)
3. **Human Detection Dataset** - Binary classification, various sizes (resized to 32x32)

## Metrics

The evaluation computes two key metrics:

- **AUROC (Area Under ROC Curve)**: Higher is better (100% is perfect)
  - Measures the model's ability to distinguish between in-distribution and OOD samples
  - Values typically range from 50% (random) to 100% (perfect separation)

- **FPR95 (False Positive Rate at 95% True Positive Rate)**: Lower is better (0% is perfect)
  - Measures the percentage of OOD samples incorrectly classified as in-distribution
  - When the model correctly identifies 95% of in-distribution samples

## OOD Detection Methods

The script supports three scoring functions:

1. **MSP (Maximum Softmax Probability)** - Default
   - Uses the maximum probability from the softmax output
   - Simple and effective baseline

2. **Energy**
   - Energy-based OOD detection
   - Uses: `-T * log(sum(exp(logits/T)))`

3. **ODIN**
   - Temperature-scaled softmax with input preprocessing
   - Requires `--temperature` parameter

## Usage

### Basic Command

```bash
cd SimKD
python eval_ood.py \
    --model_path <path_to_trained_model> \
    --model_s <student_architecture> \
    --ood_dataset <cifar10|tiny-imagenet|human-detection> \
    --batch_size 128 \
    --num_workers 12
```

### Examples

#### Example 1: Evaluate PKT model on CIFAR-10

```bash
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset cifar10 \
    --batch_size 128 \
    --num_workers 12
```

#### Example 2: Evaluate soft_pkt model on Tiny-ImageNet

```bash
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset tiny-imagenet \
    --batch_size 128 \
    --num_workers 12
```

#### Example 3: Evaluate on Human Detection dataset

```bash
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_simkd_r~0.0_a~0.0_b~1.0_0/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset human-detection \
    --batch_size 128 \
    --num_workers 12
```

#### Example 4: Using Energy scoring function

```bash
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset cifar10 \
    --score_func energy \
    --temperature 1.0
```

### Using Convenience Scripts

#### Windows (PowerShell/CMD):

```batch
run_ood_eval.bat "save\students\models\S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0\resnet8x4_best.pth" cifar10
```

#### Linux/Mac:

```bash
chmod +x run_ood_eval.sh
./run_ood_eval.sh save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth cifar10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **Required** | Path to trained model checkpoint (.pth file) |
| `--model_s` | str | `resnet8x4` | Student model architecture |
| `--ood_dataset` | str | **Required** | OOD dataset: `cifar10`, `tiny-imagenet`, or `human-detection` |
| `--in_dataset` | str | `cifar100` | In-distribution dataset (what model was trained on) |
| `--batch_size` | int | `128` | Batch size for evaluation |
| `--num_workers` | int | `4` | Number of data loading workers |
| `--gpu_id` | str | `0` | GPU device ID |
| `--data_folder` | str | `../data` | Path to data folder containing datasets |
| `--score_func` | str | `msp` | Scoring function: `msp`, `energy`, or `odin` |
| `--temperature` | float | `1.0` | Temperature scaling for ODIN/Energy |

## Typical Workflow

### 1. Train a student model

```bash
python train_student.py \
    --num_workers 12 \
    --distill soft_pkt \
    --model_s resnet8x4 \
    --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth \
    --dataset cifar100 \
    --trial 1 \
    -r 1.0 -a 0.0 -b 5.0
```

### 2. Run OOD evaluation on all three datasets

```bash
# CIFAR-10
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset cifar10

# Tiny-ImageNet
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset tiny-imagenet

# Human Detection
python eval_ood.py \
    --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth" \
    --model_s resnet8x4 \
    --ood_dataset human-detection
```

## Output

The evaluation will:

1. **Print results to console**:
   - AUROC score
   - FPR95 score
   - Score statistics (mean, std, min, max)

2. **Save results to files** in the model's directory under `ood_results/`:
   - `ood_<dataset>_<score_func>.txt` - Text file with all results
   - `id_scores_<dataset>_<score_func>.npy` - NumPy array of in-distribution scores
   - `ood_scores_<dataset>_<score_func>.npy` - NumPy array of OOD scores

Example output location:
```
save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/
├── resnet8x4_best.pth
├── parameters.json
├── test_best_metrics.json
└── ood_results/
    ├── ood_cifar10_msp.txt
    ├── id_scores_cifar10_msp.npy
    └── ood_scores_cifar10_msp.npy
```

## Expected Results

Good OOD detection performance typically shows:
- **AUROC > 70%** (higher is better)
- **FPR95 < 50%** (lower is better)

The exact values depend on:
- The distillation method used
- The model architecture
- The OOD dataset difficulty
- The in-distribution dataset

## Troubleshooting

### Dataset not found error

Make sure your datasets are in the correct location:
```
data/
├── cifar-10-batches-py/
├── cifar-100-python/
├── tiny-imagenet-200/
│   ├── train/
│   ├── val/
│   │   ├── images/
│   │   └── val_annotations.txt
│   └── wnids.txt
└── human detection dataset/
    ├── 0/  (non-human images)
    └── 1/  (human images)
```

Run the verification script to check:
```bash
python ../verify_ood_datasets.py
```

### Model loading error

Ensure the model path points to a `.pth` file containing the trained student model:
```bash
ls -l save/students/models/*/resnet8x4_best.pth
```

### Out of memory error

Reduce the batch size:
```bash
python eval_ood.py ... --batch_size 64
```

## Comparing Multiple Models

To compare OOD detection performance across different distillation methods, you can run the evaluation on all trained models:

```bash
# Evaluate all models on CIFAR-10
for model_dir in save/students/models/*/; do
    model_file="${model_dir}resnet8x4_best.pth"
    if [ -f "$model_file" ]; then
        echo "Evaluating: $model_file"
        python eval_ood.py --model_path "$model_file" --model_s resnet8x4 --ood_dataset cifar10
    fi
done
```

Then aggregate results from all `ood_results/` directories for analysis.

## Notes

- The evaluation uses the test set of CIFAR-100 as the in-distribution dataset
- All OOD datasets are preprocessed with the same normalization as CIFAR-100
- Images are resized to 32x32 to match CIFAR-100's input size
- The evaluation is performed in inference mode (no gradients)
- Results are deterministic (no data augmentation during evaluation)
