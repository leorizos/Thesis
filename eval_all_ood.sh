#!/bin/bash
# Script to evaluate OOD detection on all datasets
# Usage: ./eval_all_ood.sh <model_path>

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

MODEL_S="resnet8x4"
BATCH_SIZE=128
NUM_WORKERS=12

# Extract model directory
MODEL_DIR=$(dirname "$MODEL_PATH")
OUTPUT_FILE="$MODEL_DIR/ood_results/all_ood_results.txt"

# Create ood_results directory if it doesn't exist
mkdir -p "$MODEL_DIR/ood_results"

# Extract best accuracy from test_best_metrics.json
BEST_ACC="N/A"
METRICS_FILE="$MODEL_DIR/test_best_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    BEST_ACC=$(python -c "import json; d=json.load(open('$METRICS_FILE')); print(d['test_acc'])")
fi

# Write header
echo "Model: $MODEL_S" > "$OUTPUT_FILE"
echo "Best Accuracy: $BEST_ACC" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "OOD Detection Results:" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

echo "[1/7] Evaluating on CIFAR-10..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset cifar10 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[2/7] Evaluating on Tiny-ImageNet-200..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset tiny-imagenet --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[3/7] Evaluating on Human Detection Dataset..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset human-detection --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[4/7] Evaluating on DTD..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset dtd --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[5/7] Evaluating on SVHN..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset svhn --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[6/7] Evaluating on Places..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset places --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo "[7/7] Evaluating on LSUN..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset lsun --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --quiet >> "$OUTPUT_FILE" 2>&1

echo ""
echo "================================================================================"
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "================================================================================"
