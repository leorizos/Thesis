"""
Experiment Tracking Script
Monitors progress of training and evaluation experiments
Shows which models are trained, evaluated, and missing
"""

import os
import sys
import json
import re
from collections import defaultdict
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

split_symbol = '~' if os.name == 'nt' else ':'


def parse_model_name(model_dir_name):
    """Parse model directory name to extract configuration"""
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


def check_model_status(model_dir):
    """Check training and evaluation status of a model"""
    status = {
        'model_exists': False,
        'test_metrics_exists': False,
        'ood_evaluated': {
            'cifar10': False,
            'tiny-imagenet': False,
            'human-detection': False
        },
        'test_acc': None
    }

    # Check if model file exists
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_best.pth')]
    status['model_exists'] = len(model_files) > 0

    # Check if test metrics exist
    test_metrics_file = os.path.join(model_dir, 'test_best_metrics.json')
    if os.path.exists(test_metrics_file):
        status['test_metrics_exists'] = True
        try:
            with open(test_metrics_file, 'r') as f:
                metrics = json.load(f)
                status['test_acc'] = metrics.get('test_acc', None)
        except:
            pass

    # Check OOD evaluation
    ood_results_dir = os.path.join(model_dir, 'ood_results')
    if os.path.exists(ood_results_dir):
        for ood_dataset in status['ood_evaluated'].keys():
            # Check for any score function
            for score_func in ['msp', 'energy', 'odin']:
                result_file = os.path.join(ood_results_dir, f'ood_{ood_dataset}_{score_func}.txt')
                if os.path.exists(result_file):
                    status['ood_evaluated'][ood_dataset] = True
                    break

    return status


def scan_experiments(models_dir):
    """Scan all experiments and their status"""
    experiments = defaultdict(list)

    if not os.path.exists(models_dir):
        return experiments

    for model_dir_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_dir_name)

        if not os.path.isdir(model_dir):
            continue

        # Parse model name
        config = parse_model_name(model_dir_name)
        if not config:
            continue

        # Check status
        status = check_model_status(model_dir)

        # Group by configuration (excluding trial)
        config_key = (
            config['distill'],
            config['r'],
            config['a'],
            config['b']
        )

        experiments[config_key].append({
            'trial': config['trial'],
            'model_dir': model_dir_name,
            'status': status
        })

    return experiments


def print_experiment_status(experiments):
    """Print formatted status of all experiments"""
    print("\n" + "=" * 120)
    print("EXPERIMENT TRACKING - STATUS OVERVIEW")
    print("=" * 120 + "\n")

    if not experiments:
        print("No experiments found!")
        return

    total_configs = len(experiments)
    total_models = sum(len(trials) for trials in experiments.values())

    print(f"Total configurations: {total_configs}")
    print(f"Total models: {total_models}\n")

    for config_key, trials in sorted(experiments.items()):
        distill, r, a, b = config_key

        print("=" * 120)
        print(f"Method: {distill:15} | Parameters: r={r}, a={a}, b={b}")
        print("=" * 120)

        # Count statuses
        num_trials = len(trials)
        num_trained = sum(1 for t in trials if t['status']['model_exists'])
        num_tested = sum(1 for t in trials if t['status']['test_metrics_exists'])

        ood_counts = {
            'cifar10': sum(1 for t in trials if t['status']['ood_evaluated']['cifar10']),
            'tiny-imagenet': sum(1 for t in trials if t['status']['ood_evaluated']['tiny-imagenet']),
            'human-detection': sum(1 for t in trials if t['status']['ood_evaluated']['human-detection'])
        }

        print(f"{'Progress:':<20} {num_trained}/{num_trials} trained, {num_tested}/{num_trials} tested")
        print(f"{'OOD Evaluated:':<20} CIFAR-10: {ood_counts['cifar10']}/{num_trials}, "
              f"Tiny-ImageNet: {ood_counts['tiny-imagenet']}/{num_trials}, "
              f"Human-Detection: {ood_counts['human-detection']}/{num_trials}")

        print("\nTrial Details:")
        print(f"{'Trial':<8} {'Trained':<10} {'Tested':<10} {'CIFAR-10':<12} {'Tiny-ImageNet':<15} {'Human-Det':<12} {'Test Acc':<10}")
        print("-" * 120)

        for trial_info in sorted(trials, key=lambda x: x['trial']):
            trial = trial_info['trial']
            status = trial_info['status']

            trained = "✓" if status['model_exists'] else "✗"
            tested = "✓" if status['test_metrics_exists'] else "✗"
            cifar10 = "✓" if status['ood_evaluated']['cifar10'] else "✗"
            tiny_imagenet = "✓" if status['ood_evaluated']['tiny-imagenet'] else "✗"
            human_det = "✓" if status['ood_evaluated']['human-detection'] else "✗"
            test_acc = f"{status['test_acc']:.2f}%" if status['test_acc'] is not None else "N/A"

            print(f"{trial:<8} {trained:<10} {tested:<10} {cifar10:<12} {tiny_imagenet:<15} {human_det:<12} {test_acc:<10}")

        print()

    print("=" * 120 + "\n")


def print_missing_experiments(experiments, expected_trials=5):
    """Print list of missing experiments and evaluations"""
    print("=" * 120)
    print("MISSING EXPERIMENTS AND EVALUATIONS")
    print("=" * 120 + "\n")

    missing_training = []
    missing_ood = []

    for config_key, trials in sorted(experiments.items()):
        distill, r, a, b = config_key
        config_name = f"{distill} (r={r}, a={a}, b={b})"

        # Check for missing trials
        existing_trials = {t['trial'] for t in trials}
        for trial in range(1, expected_trials + 1):
            if trial not in existing_trials:
                missing_training.append(f"  {config_name} - Trial {trial}")

        # Check for missing OOD evaluations
        for trial_info in trials:
            trial = trial_info['trial']
            status = trial_info['status']

            if status['model_exists']:
                for ood_dataset in ['cifar10', 'tiny-imagenet', 'human-detection']:
                    if not status['ood_evaluated'][ood_dataset]:
                        missing_ood.append(
                            f"  {config_name} - Trial {trial} - {ood_dataset}"
                        )

    if missing_training:
        print("Missing Training:")
        for item in missing_training:
            print(item)
        print()
    else:
        print("✓ All training experiments completed!\n")

    if missing_ood:
        print("Missing OOD Evaluations:")
        for item in missing_ood:
            print(item)
        print()
    else:
        print("✓ All OOD evaluations completed!\n")

    print("=" * 120 + "\n")


def print_completion_stats(experiments, expected_trials=5):
    """Print overall completion statistics"""
    print("=" * 120)
    print("COMPLETION STATISTICS")
    print("=" * 120 + "\n")

    total_expected = len(experiments) * expected_trials
    total_trained = sum(
        sum(1 for t in trials if t['status']['model_exists'])
        for trials in experiments.values()
    )

    ood_datasets = ['cifar10', 'tiny-imagenet', 'human-detection']
    total_ood_evaluations = {}

    for ood_dataset in ood_datasets:
        total_ood_evaluations[ood_dataset] = sum(
            sum(1 for t in trials if t['status']['ood_evaluated'][ood_dataset])
            for trials in experiments.values()
        )

    print(f"Training Progress: {total_trained}/{total_expected} ({100*total_trained/total_expected:.1f}%)")

    for ood_dataset in ood_datasets:
        count = total_ood_evaluations[ood_dataset]
        print(f"OOD Evaluation ({ood_dataset}): {count}/{total_trained} ({100*count/total_trained:.1f}% of trained models)")

    print("\n" + "=" * 120 + "\n")


def generate_commands_for_missing(experiments, expected_trials=5):
    """Generate commands to run missing experiments"""
    print("=" * 120)
    print("COMMANDS TO RUN MISSING EXPERIMENTS")
    print("=" * 120 + "\n")

    print("# Missing Training Commands:\n")

    for config_key, trials in sorted(experiments.items()):
        distill, r, a, b = config_key

        existing_trials = {t['trial'] for t in trials}
        for trial in range(1, expected_trials + 1):
            if trial not in existing_trials:
                print(f"python train_student.py --num_workers 12 --distill {distill} "
                      f"--model_s resnet8x4 --path_t save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth "
                      f"--dataset cifar100 --trial {trial} -r {r} -a {a} -b {b}")

    print("\n# Missing OOD Evaluation Commands:\n")

    for config_key, trials in sorted(experiments.items()):
        distill, r, a, b = config_key

        for trial_info in trials:
            trial = trial_info['trial']
            status = trial_info['status']
            model_dir = trial_info['model_dir']

            if status['model_exists']:
                model_path = f"save/students/models/{model_dir}/resnet8x4_best.pth"

                for ood_dataset in ['cifar10', 'tiny-imagenet', 'human-detection']:
                    if not status['ood_evaluated'][ood_dataset]:
                        print(f"python eval_ood.py --model_path \"{model_path}\" "
                              f"--model_s resnet8x4 --ood_dataset {ood_dataset}")

    print("\n" + "=" * 120 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser('Track experimental progress')
    parser.add_argument('--models_dir', type=str, default='save/students/models',
                        help='Directory containing trained models')
    parser.add_argument('--expected_trials', type=int, default=5,
                        help='Expected number of trials per configuration')
    parser.add_argument('--show_commands', action='store_true',
                        help='Show commands for missing experiments')

    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory not found: {args.models_dir}")
        sys.exit(1)

    # Scan experiments
    experiments = scan_experiments(args.models_dir)

    if not experiments:
        print("No experiments found!")
        sys.exit(0)

    # Print status
    print_experiment_status(experiments)
    print_completion_stats(experiments, args.expected_trials)
    print_missing_experiments(experiments, args.expected_trials)

    if args.show_commands:
        generate_commands_for_missing(experiments, args.expected_trials)


if __name__ == '__main__':
    main()
