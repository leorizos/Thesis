#!/bin/bash
# Evaluate OOD detection for one or all trials of a model.
#
# Usage:
#   ./eval_all_ood_multi.sh "<pattern>" [trial]
#
# <pattern>  Path with * where the trial number goes, e.g.:
#   "save/students/models/S:resnet8x4_..._*/resnet8x4_best.pth"
# [trial]    Optional: evaluate only this trial number (1-6).
#            If omitted, evaluates all trials 1-6.
#
# Example (single trial):
#   ./eval_all_ood_multi.sh "save/students/models/..._*/resnet8x4_best.pth" 3
# Example (all trials):
#   ./eval_all_ood_multi.sh "save/students/models/..._*/resnet8x4_best.pth"

PATTERN=$1
TRIAL=$2

if [ -z "$PATTERN" ]; then
    echo "Usage: $0 <pattern> [trial]"
    echo "  pattern: model path with * where the trial number goes"
    echo "  trial:   optional trial number to evaluate (default: all 1-6)"
    exit 1
fi

if [ -n "$TRIAL" ]; then
    TRIALS=($TRIAL)
else
    TRIALS=(1 2 3 4 5 6)
fi

for t in "${TRIALS[@]}"; do
    MODEL_PATH="${PATTERN/\*/$t}"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Skipping trial $t — file not found: $MODEL_PATH"
        continue
    fi
    echo "================================================================================"
    echo "Evaluating trial $t: $MODEL_PATH"
    echo "================================================================================"
    bash eval_all_ood.sh "$MODEL_PATH"
done
