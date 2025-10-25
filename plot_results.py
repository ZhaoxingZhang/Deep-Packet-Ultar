
import json
import matplotlib.pyplot as plt
import click
import os

@click.command()
@click.option('--result_path', required=True, type=click.Path(exists=True), help='Path to the open_set_curves.json file.')
@click.option('--output_path', required=True, type=click.Path(), help='Path to save the output plot image.')
def plot_curves(result_path, output_path):
    """Plots the accuracy-rejection curves from the evaluation results."""
    with open(result_path, 'r') as f:
        data = json.load(f)

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    strategies = {
        'baseline_softmax': 'Baseline (ResNet) Softmax',
        'moe_softmax': 'MoE Final Layer Softmax',
        'moe_gate': 'MoE Gating Network'
    }

    for key, label in strategies.items():
        if key in data:
            rejection_rates = [point['rejection_rate'] for point in data[key]]
            accuracies = [point['accuracy'] for point in data[key]]
            ax.plot(rejection_rates, accuracies, marker='.', linestyle='-', label=label)

    ax.set_title('Accuracy vs. Rejection Rate for Open-Set Recognition', fontsize=16)
    ax.set_xlabel('Rejection Rate', fontsize=12)
    ax.set_ylabel('Accuracy on Accepted Samples', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(min(accuracies) if 'accuracies' in locals() and accuracies else 0.8, 1.0)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_curves()
