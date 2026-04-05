#!/usr/bin/env python3
"""
Aggregate OOD results across multiple seeds of the same method and output LaTeX lines.

Usage:
    python aggregate_ood.py <glob_pattern> <method_label>

The glob pattern should use * where the trial/seed number appears (right after b_10.0_).

Examples:
    # Directory-based results (new format with ood_results/all_ood_results.txt):
    python aggregate_ood.py "save/students/models/S_resnet8x4_T_resnet32x4_cifar100_soft_akd_r_1.0_a_0.0_b_10.0_*_l_0.7_a_0.1_glr_0.0001_t_0.2" "SoftAKD $\\lambda_0$=0.7 T=0.2"

    # Standalone .txt files sitting directly in models/:
    python aggregate_ood.py "save/students/models/Sresnet8x4_Tresnet32x4_cifar100_soft_akd_r1.0_a0.0_b10.0_*_l_0.7_a_0.1_glr_0.0001_t_0.5" "SoftAKD $\\lambda_0$=0.7 T=0.5"
"""

import sys
import glob
import json
import math
import os
import re


DATASETS = ['cifar10', 'tiny-imagenet', 'human-detection', 'dtd', 'svhn', 'places', 'lsun']


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    if n == 1:
        return round(m, 2), None
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return round(m, 2), round(math.sqrt(var), 2)


def fmt(m, s):
    if m is None:
        return '--'
    if s is None:
        return f"{m}"
    return f"{m} $\\pm$ {s}"


def parse_ood_txt(filepath):
    auroc, fpr95 = {}, {}
    best_acc = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Best Accuracy:'):
                try:
                    best_acc = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            for d in DATASETS:
                if line.lower().startswith(d):
                    a = re.search(r'AUROC:\s+([\d.]+)%', line)
                    fp = re.search(r'FPR95:\s+([\d.]+)%', line)
                    if a:
                        auroc[d] = float(a.group(1))
                    if fp:
                        fpr95[d] = float(fp.group(1))
    return auroc, fpr95, best_acc


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pattern = sys.argv[1]
    label = sys.argv[2]

    # Find matches: directories or .txt files
    matches = sorted(glob.glob(pattern))
    if not matches:
        matches = sorted(glob.glob(pattern + '.txt'))
    if not matches:
        print(f"No matches found for: {pattern}")
        sys.exit(1)

    print(f"Found {len(matches)} trial(s):")
    for m in matches:
        print(f"  {m}")
    print()

    all_auroc = {d: [] for d in DATASETS}
    all_fpr   = {d: [] for d in DATASETS}
    all_top1, all_top5 = [], []
    per_seed_auroc_avg = []
    per_seed_fpr_avg   = []

    for match in matches:
        result_file = None
        json_file   = None

        if os.path.isdir(match):
            # Support both: pattern points to model dir, or directly to ood_results dir
            for candidate in [
                os.path.join(match, 'all_ood_results.txt'),
                os.path.join(match, 'ood_results', 'all_ood_results.txt'),
            ]:
                if os.path.exists(candidate):
                    result_file = candidate
                    break
            model_dir = os.path.dirname(match) if os.path.basename(match) == 'ood_results' else match
            json_candidate = os.path.join(model_dir, 'test_best_metrics.json')
            if os.path.exists(json_candidate):
                json_file = json_candidate
        elif os.path.isfile(match):
            result_file = match

        if result_file is None:
            print(f"  WARNING: no result file found for {match}, skipping.")
            continue

        auroc, fpr95, best_acc = parse_ood_txt(result_file)

        # Prefer json for accuracy (also has top5)
        if json_file:
            with open(json_file) as f:
                d = json.load(f)
                best_acc = d.get('test_acc', best_acc)
                top5 = d.get('test_acc_top5')
                if top5 is not None:
                    all_top5.append(top5)

        for d in DATASETS:
            if d in auroc:
                all_auroc[d].append(auroc[d])
            if d in fpr95:
                all_fpr[d].append(fpr95[d])

        if best_acc is not None:
            all_top1.append(best_acc)

        # Per-seed average across datasets (for the Average column)
        seed_auroc_vals = [auroc[d] for d in DATASETS if d in auroc]
        seed_fpr_vals   = [fpr95[d] for d in DATASETS if d in fpr95]
        if seed_auroc_vals:
            per_seed_auroc_avg.append(sum(seed_auroc_vals) / len(seed_auroc_vals))
        if seed_fpr_vals:
            per_seed_fpr_avg.append(sum(seed_fpr_vals) / len(seed_fpr_vals))

    n = len(all_top1)
    print(f"Aggregating {n} trial(s)\n")

    # Per-dataset stats
    auroc_means, fpr_means = [], []
    auroc_cells, fpr_cells = [], []

    for d in DATASETS:
        am, as_ = mean_std(all_auroc[d])
        fm, fs  = mean_std(all_fpr[d])
        auroc_means.append(am)
        fpr_means.append(fm)
        auroc_cells.append(fmt(am, as_))
        fpr_cells.append(fmt(fm, fs))

    # Average column: mean and std of per-seed averages across datasets
    avg_am, avg_as = mean_std(per_seed_auroc_avg)
    avg_fm, avg_fs = mean_std(per_seed_fpr_avg)

    # Accuracy
    top1_m, top1_s = mean_std(all_top1)
    top5_m, top5_s = mean_std(all_top5) if all_top5 else (None, None)

    auroc_line = f"{label} & {' & '.join(auroc_cells)} & {fmt(avg_am, avg_as)} \\\\"
    fpr_line   = f"{label} & {' & '.join(fpr_cells)}   & {fmt(avg_fm, avg_fs)} \\\\"
    acc_line   = f"{label} & {n} & {fmt(top1_m, top1_s)} & {fmt(top5_m, top5_s)} \\\\"

    print("=" * 70)
    print("AUROC table:")
    print(auroc_line)
    print("\nFPR95 table:")
    print(fpr_line)
    print("\nAccuracy table:")
    print(acc_line)
    print("=" * 70)


if __name__ == '__main__':
    main()
