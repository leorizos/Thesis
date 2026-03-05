# Quick Start Guide: OOD Detection Evaluation

## Step 1: Train Your Model

Train a student model using any distillation method:

```bash
python train_student.py --num_workers 12 --distill soft_pkt --model_s resnet8x4 \
  --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth \
  --dataset cifar100 --trial 1 -r 1.0 -a 0.0 -b 5.0
```

The trained model will be saved in:
```
save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth
```

## Step 2: Evaluate OOD Detection

### Option A: Evaluate on a single OOD dataset

```bash
python eval_ood.py \
  --model_path "save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth" \
  --model_s resnet8x4 \
  --ood_dataset cifar10 \
  --batch_size 128 \
  --num_workers 12
```

**Available OOD datasets:**
- `cifar10` - CIFAR-10 (10 classes, 10,000 test images)
- `tiny-imagenet` - Tiny-ImageNet-200 (200 classes, 10,000 validation images)
- `human-detection` - Human Detection Dataset (2 classes, 921 images)

### Option B: Evaluate on all three OOD datasets at once

**Windows:**
```batch
eval_all_ood.bat "save\students\models\S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1\resnet8x4_best.pth"
```

**Linux/Mac:**
```bash
chmod +x eval_all_ood.sh
./eval_all_ood.sh save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/resnet8x4_best.pth
```

## Step 3: Check Results

Results are saved in the model directory under `ood_results/`:

```
save/students/models/S~resnet8x4_T~resnet32x4_cifar100_soft_pkt_r~1.0_a~0.0_b~5.0_1/
└── ood_results/
    ├── ood_cifar10_msp.txt
    ├── ood_tiny-imagenet_msp.txt
    ├── ood_human-detection_msp.txt
    ├── id_scores_*.npy
    └── ood_scores_*.npy
```

### Understanding the Metrics

**AUROC (Area Under ROC Curve):**
- Range: 50% (random) to 100% (perfect)
- **Higher is better**
- Measures ability to distinguish ID from OOD
- Good performance: > 70%

**FPR95 (False Positive Rate at 95% TPR):**
- Range: 0% (perfect) to 100% (worst)
- **Lower is better**
- % of OOD samples misclassified as ID
- Good performance: < 50%

### Example Output

```
================================================================================
RESULTS
================================================================================
AUROC: 75.23%
FPR95: 45.67%
================================================================================

Score Statistics:
ID scores  - Mean: 0.5234, Std: 0.2156, Min: 0.0612, Max: 0.9987
OOD scores - Mean: 0.3421, Std: 0.1889, Min: 0.0523, Max: 0.9234
```

## Complete Example Workflow

```bash
# 1. Train with PKT
python train_student.py --num_workers 12 --distill pkt --model_s resnet8x4 \
  --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth \
  -r 1.0 -a 1.0 -b 1.0

# 2. Train with soft_pkt
python train_student.py --num_workers 12 --distill soft_pkt --model_s resnet8x4 \
  --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth \
  -r 1.0 -a 0.0 -b 5.0 --trial 1

# 3. Train with SimKD
python train_student.py --num_workers 12 --distill simkd --model_s resnet8x4 \
  --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth \
  -r 0.0 -a 0.0 -b 1.0

# 4. Evaluate all models on all OOD datasets
for model_dir in save/students/models/*/; do
  model_file="${model_dir}resnet8x4_best.pth"
  if [ -f "$model_file" ]; then
    echo "Evaluating: $model_file"
    ./eval_all_ood.sh "$model_file"
  fi
done
```

## Advanced Options

### Using Different Scoring Functions

**Energy-based scoring:**
```bash
python eval_ood.py --model_path <path> --ood_dataset cifar10 --score_func energy --temperature 1.0
```

**ODIN scoring:**
```bash
python eval_ood.py --model_path <path> --ood_dataset cifar10 --score_func odin --temperature 1000.0
```

### Custom Data Path

```bash
python eval_ood.py --model_path <path> --ood_dataset cifar10 --data_folder /path/to/data
```

### Different GPU

```bash
python eval_ood.py --model_path <path> --ood_dataset cifar10 --gpu_id 1
```

## Troubleshooting

**Q: Dataset not found error**
- Verify datasets with: `python ../verify_ood_datasets.py`
- Check data folder structure matches requirements

**Q: Out of memory**
- Reduce batch size: `--batch_size 64` or `--batch_size 32`

**Q: Slow evaluation**
- Increase num_workers: `--num_workers 12`
- Use GPU: Check `--gpu_id` is set correctly

## For More Details

See the complete documentation in `OOD_EVALUATION_README.md`
