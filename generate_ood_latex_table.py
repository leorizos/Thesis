"""
Generate LaTeX tables from OOD detection results across multiple models
"""

import os
import re
import numpy as np
from pathlib import Path

# Define models directory
MODELS_DIR = Path("save/students/models")

# OOD datasets in order
OOD_DATASETS = ['cifar10', 'tiny-imagenet', 'human-detection', 'dtd', 'svhn', 'places', 'lsun']

def parse_all_ood_results(filepath):
    """Parse the consolidated all_ood_results.txt file"""
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse each dataset's results
    for dataset in OOD_DATASETS:
        # Pattern: dataset_name  AUROC: XX.XX%  FPR95: XX.XX%
        pattern = rf'{dataset}\s+AUROC:\s+([\d.]+)%\s+FPR95:\s+([\d.]+)%'
        match = re.search(pattern, content)

        if match:
            auroc = float(match.group(1))
            fpr95 = float(match.group(2))
            results[dataset] = {'auroc': auroc, 'fpr95': fpr95}

    return results


def parse_individual_ood_files(model_dir):
    """Parse individual ood_<dataset>_msp.txt files"""
    results = {}
    ood_dir = model_dir / 'ood_results'

    if not ood_dir.exists():
        return results

    for dataset in OOD_DATASETS:
        ood_file = ood_dir / f'ood_{dataset}_msp.txt'

        if not ood_file.exists():
            continue

        with open(ood_file, 'r') as f:
            content = f.read()

        # Extract AUROC and FPR95
        auroc_match = re.search(r'AUROC:\s+([\d.]+)%', content)
        fpr95_match = re.search(r'FPR95:\s+([\d.]+)%', content)

        if auroc_match and fpr95_match:
            auroc = float(auroc_match.group(1))
            fpr95 = float(fpr95_match.group(1))
            results[dataset] = {'auroc': auroc, 'fpr95': fpr95}

    return results


def get_ood_results(model_dir):
    """Get OOD results from either format"""
    all_results_file = model_dir / 'ood_results' / 'all_ood_results.txt'

    if all_results_file.exists():
        return parse_all_ood_results(all_results_file)
    else:
        return parse_individual_ood_files(model_dir)


def collect_all_results():
    """Collect results from all models, grouped by methodology"""

    # Group models by methodology
    kd_models = []
    pkt_models = []
    soft_pkt2_models = []

    # Find all model directories
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Categorize by methodology
        if '_kd_' in model_name and 'soft_pkt' not in model_name:
            kd_models.append(model_dir)
        elif '_pkt_' in model_name and 'soft_pkt' not in model_name:
            pkt_models.append(model_dir)
        elif 'soft_pkt2' in model_name:
            soft_pkt2_models.append(model_dir)

    print(f"Found {len(kd_models)} KD models")
    print(f"Found {len(pkt_models)} PKT models")
    print(f"Found {len(soft_pkt2_models)} Soft PKT v2 models")

    # Collect results for each group
    def collect_group_results(model_list):
        group_results = {dataset: {'auroc': [], 'fpr95': []} for dataset in OOD_DATASETS}

        for model_dir in model_list:
            results = get_ood_results(model_dir)
            print(f"  - {model_dir.name}: {len(results)} datasets")

            for dataset, metrics in results.items():
                if dataset in group_results:
                    group_results[dataset]['auroc'].append(metrics['auroc'])
                    group_results[dataset]['fpr95'].append(metrics['fpr95'])

        return group_results

    print("\nCollecting KD results...")
    kd_results = collect_group_results(kd_models)

    print("\nCollecting PKT results...")
    pkt_results = collect_group_results(pkt_models)

    print("\nCollecting Soft PKT v2 results...")
    soft_pkt2_results = collect_group_results(soft_pkt2_models)

    return {
        'KD': kd_results,
        'PKT': pkt_results,
        'Soft PKT v2': soft_pkt2_results
    }


def format_mean_std(values):
    """Format values as mean ± std"""
    if len(values) == 0:
        return "---"
    elif len(values) == 1:
        return f"{values[0]:.2f}"
    else:
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        return f"{mean:.2f} $\\pm$ {std:.2f}"


def generate_latex_tables(all_results):
    """Generate LaTeX tables for AUROC and FPR95"""

    latex_output = []

    # Header
    latex_output.append("% OOD Detection Results - LaTeX Tables")
    latex_output.append("% Generated automatically from model evaluation results")
    latex_output.append("")

    # Generate AUROC table
    latex_output.append("% AUROC Table")
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{AUROC (\%) for OOD Detection across Different Datasets}")
    latex_output.append(r"\label{tab:ood_auroc}")
    latex_output.append(r"\begin{tabular}{l" + "c" * len(OOD_DATASETS) + "}")
    latex_output.append(r"\hline")

    # Header row
    header = "Method & " + " & ".join([ds.replace('-', ' ').title() for ds in OOD_DATASETS]) + r" \\"
    latex_output.append(header)
    latex_output.append(r"\hline")

    # Data rows for AUROC
    for method_name, results in all_results.items():
        row_values = []
        for dataset in OOD_DATASETS:
            auroc_values = results[dataset]['auroc']
            row_values.append(format_mean_std(auroc_values))

        row = method_name + " & " + " & ".join(row_values) + r" \\"
        latex_output.append(row)

    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    latex_output.append("")
    latex_output.append("")

    # Generate FPR95 table
    latex_output.append("% FPR95 Table")
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{FPR95 (\%) for OOD Detection across Different Datasets}")
    latex_output.append(r"\label{tab:ood_fpr95}")
    latex_output.append(r"\begin{tabular}{l" + "c" * len(OOD_DATASETS) + "}")
    latex_output.append(r"\hline")

    # Header row
    latex_output.append(header)
    latex_output.append(r"\hline")

    # Data rows for FPR95
    for method_name, results in all_results.items():
        row_values = []
        for dataset in OOD_DATASETS:
            fpr95_values = results[dataset]['fpr95']
            row_values.append(format_mean_std(fpr95_values))

        row = method_name + " & " + " & ".join(row_values) + r" \\"
        latex_output.append(row)

    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")

    return "\n".join(latex_output)


def main():
    print("="*80)
    print("Generating LaTeX Tables for OOD Detection Results")
    print("="*80)

    # Collect all results
    all_results = collect_all_results()

    # Generate LaTeX tables
    print("\n" + "="*80)
    print("Generating LaTeX tables...")
    latex_content = generate_latex_tables(all_results)

    # Save to file
    output_file = "ood_results_comparison.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX tables saved to: {output_file}")
    print("="*80)

    # Also print to console
    print("\n" + "="*80)
    print("LaTeX Content:")
    print("="*80)
    print(latex_content)
    print("="*80)


if __name__ == '__main__':
    main()
