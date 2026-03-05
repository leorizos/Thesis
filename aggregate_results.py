"""
Results Aggregation Script
Aggregates training and OOD detection results across multiple trials
Computes mean ± std for all metrics
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

split_symbol = '~' if os.name == 'nt' else ':'


def parse_model_name(model_dir_name):
    """Parse model directory name to extract configuration"""
    # Format: S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_1
    pattern = f'S{split_symbol}(.+)_T{split_symbol}(.+)_(.+)_(.+)_r{split_symbol}(.+)_a{split_symbol}(.+)_b{split_symbol}(.+)_(.+)'
    match = re.match(pattern, model_dir_name)

    if match:
        return {
            'student': match.group(1),
            'teacher': match.group(2),
            'dataset': match.group(3),
            'distill': match.group(4),
            'r': float(match.group(5)),
            'a': float(match.group(6)),
            'b': float(match.group(7)),
            'trial': int(match.group(8))
        }
    return None


def read_test_metrics(model_dir):
    """Read test metrics from test_best_metrics.json"""
    metrics_file = os.path.join(model_dir, 'test_best_metrics.json')

    if not os.path.exists(metrics_file):
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return {
            'test_acc': metrics.get('test_acc', None),
            'test_acc_top5': metrics.get('test_acc_top5', None),
            'test_loss': metrics.get('test_loss', None),
            'epoch': metrics.get('epoch', None)
        }
    except Exception as e:
        print(f"Warning: Could not read {metrics_file}: {e}")
        return None


def read_ood_metrics(model_dir, ood_dataset):
    """Read OOD detection metrics for a specific dataset"""
    ood_results_dir = os.path.join(model_dir, 'ood_results')

    if not os.path.exists(ood_results_dir):
        return None

    # Try different score functions
    for score_func in ['msp', 'energy', 'odin']:
        results_file = os.path.join(ood_results_dir, f'ood_{ood_dataset}_{score_func}.txt')

        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    content = f.read()

                # Extract AUROC and FPR95
                auroc_match = re.search(r'AUROC:\s+([\d.]+)%', content)
                fpr95_match = re.search(r'FPR95:\s+([\d.]+)%', content)

                if auroc_match and fpr95_match:
                    return {
                        'auroc': float(auroc_match.group(1)),
                        'fpr95': float(fpr95_match.group(1)),
                        'score_func': score_func
                    }
            except Exception as e:
                print(f"Warning: Could not parse {results_file}: {e}")

    return None


def collect_results(models_dir):
    """Collect all results from trained models"""
    results = []

    for model_dir_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_dir_name)

        if not os.path.isdir(model_dir):
            continue

        # Parse model name
        config = parse_model_name(model_dir_name)
        if not config:
            continue

        # Read test metrics
        test_metrics = read_test_metrics(model_dir)
        if not test_metrics:
            continue

        # Read OOD metrics for each dataset
        ood_datasets = ['cifar10', 'tiny-imagenet', 'human-detection']
        ood_metrics = {}

        for ood_dataset in ood_datasets:
            metrics = read_ood_metrics(model_dir, ood_dataset)
            if metrics:
                ood_metrics[ood_dataset] = metrics

        # Combine all information
        result = {
            'model_dir': model_dir_name,
            **config,
            **test_metrics,
            'ood_metrics': ood_metrics
        }

        results.append(result)

    return results


def group_by_configuration(results):
    """Group results by configuration (excluding trial number)"""
    grouped = defaultdict(list)

    for result in results:
        # Create key from configuration
        key = (
            result['student'],
            result['teacher'],
            result['dataset'],
            result['distill'],
            result['r'],
            result['a'],
            result['b']
        )

        grouped[key].append(result)

    return grouped


def compute_statistics(values):
    """Compute mean and std for a list of values"""
    if not values:
        return None, None

    values = [v for v in values if v is not None]
    if not values:
        return None, None

    return np.mean(values), np.std(values)


def aggregate_configuration(results_list):
    """Aggregate results for a single configuration across trials"""
    if not results_list:
        return None

    # Extract configuration from first result
    config = results_list[0]
    num_trials = len(results_list)

    # Aggregate test metrics
    test_accs = [r['test_acc'] for r in results_list if r['test_acc'] is not None]
    test_acc_top5s = [r['test_acc_top5'] for r in results_list if r['test_acc_top5'] is not None]
    test_losses = [r['test_loss'] for r in results_list if r['test_loss'] is not None]

    test_acc_mean, test_acc_std = compute_statistics(test_accs)
    test_acc_top5_mean, test_acc_top5_std = compute_statistics(test_acc_top5s)
    test_loss_mean, test_loss_std = compute_statistics(test_losses)

    # Aggregate OOD metrics
    ood_datasets = ['cifar10', 'tiny-imagenet', 'human-detection']
    ood_aggregated = {}

    for ood_dataset in ood_datasets:
        aurocs = []
        fpr95s = []

        for result in results_list:
            if ood_dataset in result['ood_metrics']:
                aurocs.append(result['ood_metrics'][ood_dataset]['auroc'])
                fpr95s.append(result['ood_metrics'][ood_dataset]['fpr95'])

        auroc_mean, auroc_std = compute_statistics(aurocs)
        fpr95_mean, fpr95_std = compute_statistics(fpr95s)

        if auroc_mean is not None:
            ood_aggregated[ood_dataset] = {
                'auroc_mean': auroc_mean,
                'auroc_std': auroc_std,
                'fpr95_mean': fpr95_mean,
                'fpr95_std': fpr95_std,
                'num_trials': len(aurocs)
            }

    return {
        'student': config['student'],
        'teacher': config['teacher'],
        'dataset': config['dataset'],
        'distill': config['distill'],
        'r': config['r'],
        'a': config['a'],
        'b': config['b'],
        'num_trials': num_trials,
        'test_acc_mean': test_acc_mean,
        'test_acc_std': test_acc_std,
        'test_acc_top5_mean': test_acc_top5_mean,
        'test_acc_top5_std': test_acc_top5_std,
        'test_loss_mean': test_loss_mean,
        'test_loss_std': test_loss_std,
        'ood_metrics': ood_aggregated
    }


def save_aggregated_results(aggregated_results, output_dir):
    """Save aggregated results in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_file = os.path.join(output_dir, 'aggregated_results.json')
    with open(json_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"Saved JSON results to: {json_file}")

    # Create DataFrame for CSV export
    rows = []
    for agg in aggregated_results:
        base_row = {
            'Method': agg['distill'],
            'r': agg['r'],
            'a': agg['a'],
            'b': agg['b'],
            'Trials': agg['num_trials'],
            'Test_Acc_Mean': agg['test_acc_mean'],
            'Test_Acc_Std': agg['test_acc_std'],
            'Test_Loss_Mean': agg['test_loss_mean'],
            'Test_Loss_Std': agg['test_loss_std']
        }

        # Add OOD metrics
        for ood_dataset, metrics in agg['ood_metrics'].items():
            base_row[f'{ood_dataset}_AUROC_Mean'] = metrics['auroc_mean']
            base_row[f'{ood_dataset}_AUROC_Std'] = metrics['auroc_std']
            base_row[f'{ood_dataset}_FPR95_Mean'] = metrics['fpr95_mean']
            base_row[f'{ood_dataset}_FPR95_Std'] = metrics['fpr95_std']

        rows.append(base_row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_file = os.path.join(output_dir, 'aggregated_results.csv')
    df.to_csv(csv_file, index=False, float_format='%.2f')
    print(f"Saved CSV results to: {csv_file}")

    # Create formatted text summary
    txt_file = os.path.join(output_dir, 'aggregated_results.txt')
    with open(txt_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("AGGREGATED RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        for agg in aggregated_results:
            f.write(f"Method: {agg['distill']}\n")
            f.write(f"Parameters: r={agg['r']}, a={agg['a']}, b={agg['b']}\n")
            f.write(f"Trials: {agg['num_trials']}\n")
            f.write("-" * 100 + "\n")

            # Test accuracy
            if agg['test_acc_mean'] is not None:
                f.write(f"Test Accuracy: {agg['test_acc_mean']:.2f} ± {agg['test_acc_std']:.2f}%\n")

            if agg['test_acc_top5_mean'] is not None:
                f.write(f"Test Top-5 Accuracy: {agg['test_acc_top5_mean']:.2f} ± {agg['test_acc_top5_std']:.2f}%\n")

            # OOD metrics
            f.write("\nOOD Detection Performance:\n")
            for ood_dataset, metrics in agg['ood_metrics'].items():
                f.write(f"  {ood_dataset}:\n")
                f.write(f"    AUROC: {metrics['auroc_mean']:.2f} ± {metrics['auroc_std']:.2f}%\n")
                f.write(f"    FPR95: {metrics['fpr95_mean']:.2f} ± {metrics['fpr95_std']:.2f}%\n")

            f.write("\n" + "=" * 100 + "\n\n")

    print(f"Saved text summary to: {txt_file}")


def print_summary_table(aggregated_results):
    """Print a formatted summary table to console"""
    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 120)

    for agg in aggregated_results:
        print(f"\n{'Method:':<20} {agg['distill']}")
        print(f"{'Parameters:':<20} r={agg['r']}, a={agg['a']}, b={agg['b']}")
        print(f"{'Trials:':<20} {agg['num_trials']}")
        print("-" * 120)

        if agg['test_acc_mean'] is not None:
            print(f"{'Test Accuracy:':<20} {agg['test_acc_mean']:.2f} ± {agg['test_acc_std']:.2f}%")

        print(f"\n{'OOD Detection Performance:':}")
        print(f"{'Dataset':<25} {'AUROC (Mean ± Std)':<30} {'FPR95 (Mean ± Std)':<30}")
        print("-" * 120)

        for ood_dataset, metrics in sorted(agg['ood_metrics'].items()):
            auroc_str = f"{metrics['auroc_mean']:.2f} ± {metrics['auroc_std']:.2f}%"
            fpr95_str = f"{metrics['fpr95_mean']:.2f} ± {metrics['fpr95_std']:.2f}%"
            print(f"{ood_dataset:<25} {auroc_str:<30} {fpr95_str:<30}")

        print("=" * 120)


def main():
    parser = argparse.ArgumentParser('Aggregate experimental results across trials')
    parser.add_argument('--models_dir', type=str, default='save/students/models',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save aggregated results')
    parser.add_argument('--method', type=str, default=None,
                        help='Filter by specific distillation method')

    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory not found: {args.models_dir}")
        sys.exit(1)

    print("Collecting results from trained models...")
    results = collect_results(args.models_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found {len(results)} trained models")

    # Filter by method if specified
    if args.method:
        results = [r for r in results if r['distill'] == args.method]
        print(f"Filtered to {len(results)} models with method '{args.method}'")

    # Group by configuration
    grouped = group_by_configuration(results)
    print(f"Found {len(grouped)} unique configurations")

    # Aggregate each configuration
    aggregated_results = []
    for config_key, results_list in sorted(grouped.items()):
        agg = aggregate_configuration(results_list)
        if agg:
            aggregated_results.append(agg)

    # Save results
    save_aggregated_results(aggregated_results, args.output_dir)

    # Print summary
    print_summary_table(aggregated_results)

    print(f"\nAggregation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
