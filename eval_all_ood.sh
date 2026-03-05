#!/bin/bash
# Script to evaluate OOD detection on all datasets
# Usage: ./eval_all_ood.sh <model_path>
# Example: ./eval_all_ood.sh save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 save/students/models/S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0/resnet8x4_best.pth"
    exit 1
fi

MODEL_S="resnet8x4"
BATCH_SIZE=128
NUM_WORKERS=12

echo "================================================================================"
echo "Evaluating OOD Detection on All Datasets"
echo "================================================================================"
echo "Model: $MODEL_PATH"
echo ""

echo "[1/7] Evaluating on CIFAR-10..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset cifar10 --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[2/7] Evaluating on Tiny-ImageNet-200..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset tiny-imagenet --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[3/7] Evaluating on Human Detection Dataset..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset human-detection --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[4/7] Evaluating on DTD..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset dtd --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[5/7] Evaluating on SVHN..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset svhn --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[6/7] Evaluating on Places..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset places --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "[7/7] Evaluating on LSUN..."
python eval_ood.py --model_path "$MODEL_PATH" --model_s $MODEL_S --ood_dataset lsun --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS
echo ""

echo "================================================================================"
echo "All evaluations completed!"
echo "Check the ood_results folder in your model directory for detailed results."
echo "================================================================================"
