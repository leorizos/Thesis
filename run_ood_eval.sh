#!/bin/bash
# Convenience script for running OOD detection evaluation
# Usage: ./run_ood_eval.sh <model_path> <ood_dataset>
# Example: ./run_ood_eval.sh save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth cifar10

MODEL_PATH=$1
OOD_DATASET=$2

if [ -z "$MODEL_PATH" ] || [ -z "$OOD_DATASET" ]; then
    echo "Usage: $0 <model_path> <ood_dataset>"
    echo "OOD datasets: cifar10, tiny-imagenet, human-detection"
    exit 1
fi

# Extract model architecture from path (assumes resnet8x4 by default)
MODEL_S="resnet8x4"

echo "Running OOD evaluation..."
echo "Model: $MODEL_PATH"
echo "OOD Dataset: $OOD_DATASET"
echo ""

python eval_ood.py \
    --model_path "$MODEL_PATH" \
    --model_s "$MODEL_S" \
    --ood_dataset "$OOD_DATASET" \
    --batch_size 128 \
    --num_workers 12 \
    --score_func msp
