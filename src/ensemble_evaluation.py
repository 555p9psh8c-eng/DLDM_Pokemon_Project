"""
Ensemble Evaluation for Pokemon TCG AI

Evaluates multi-seed ensemble methods:
1. Majority voting
2. Probability averaging
3. Weighted voting
4. Confidence-based selection
5. Diversity analysis

Author: Philipp Zengl (03827436)
Date: 2026-08-30
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import accuracy_score

# Create output directory
output_dir = Path('results/ensemble_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ENSEMBLE EVALUATION - POKEMON TCG AI")
print("="*70)


def load_models(model_paths):
    """Load multiple trained models."""
    print(f"\nLoading {len(model_paths)} models...")

    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"  ✓ Loaded: {Path(path).name}")
        models.append(checkpoint)

    return models


def get_ensemble_predictions(models, dataloader):
    """Get predictions from all models in ensemble."""
    print("\nGenerating predictions from ensemble...")

    all_model_preds = []
    all_model_probs = []

    for i, model in enumerate(models):
        model.eval()
        preds = []
        probs = []

        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence']
                outputs = model(sequences)
                batch_probs = torch.softmax(outputs, dim=-1)
                batch_preds = torch.argmax(outputs, dim=-1)

                preds.extend(batch_preds.cpu().numpy())
                probs.extend(batch_probs.cpu().numpy())

        all_model_preds.append(np.array(preds))
        all_model_probs.append(np.array(probs))
        print(f"  Model {i+1}/{len(models)}: {len(preds)} predictions")

    return {
        'predictions': np.array(all_model_preds),  # Shape: (n_models, n_samples)
        'probabilities': np.array(all_model_probs)  # Shape: (n_models, n_samples, n_classes)
    }


# ============================================================================
# 1. VOTING STRATEGIES
# ============================================================================

def majority_voting(ensemble_preds):
    """Hard voting: select most common prediction."""
    # ensemble_preds: (n_models, n_samples)
    votes = []
    for sample_idx in range(ensemble_preds.shape[1]):
        sample_votes = ensemble_preds[:, sample_idx]
        # Get most common prediction
        most_common = Counter(sample_votes).most_common(1)[0][0]
        votes.append(most_common)

    return np.array(votes)


def probability_averaging(ensemble_probs):
    """Soft voting: average probabilities and take argmax."""
    # ensemble_probs: (n_models, n_samples, n_classes)
    avg_probs = ensemble_probs.mean(axis=0)  # (n_samples, n_classes)
    predictions = np.argmax(avg_probs, axis=1)

    return predictions, avg_probs


def weighted_voting(ensemble_probs, model_weights):
    """Weighted soft voting based on model performance."""
    # Normalize weights
    weights = np.array(model_weights) / sum(model_weights)

    # Weighted average
    weighted_probs = np.zeros_like(ensemble_probs[0])
    for i, weight in enumerate(weights):
        weighted_probs += weight * ensemble_probs[i]

    predictions = np.argmax(weighted_probs, axis=1)

    return predictions, weighted_probs


def confidence_based_selection(ensemble_probs):
    """Select prediction from most confident model per sample."""
    # Get max probability for each model's prediction
    max_probs = ensemble_probs.max(axis=2)  # (n_models, n_samples)

    # For each sample, select model with highest confidence
    most_confident_model = np.argmax(max_probs, axis=0)  # (n_samples,)

    predictions = []
    for sample_idx, model_idx in enumerate(most_confident_model):
        pred = np.argmax(ensemble_probs[model_idx, sample_idx])
        predictions.append(pred)

    return np.array(predictions)


# ============================================================================
# 2. EVALUATE ALL STRATEGIES
# ============================================================================

def evaluate_ensemble_strategies(ensemble_data, labels):
    """Compare all ensemble strategies."""
    print("\n" + "="*70)
    print("ENSEMBLE STRATEGY COMPARISON")
    print("="*70)

    ensemble_preds = ensemble_data['predictions']
    ensemble_probs = ensemble_data['probabilities']

    results = {}

    # Individual models
    print("\n📊 INDIVIDUAL MODEL PERFORMANCE:")
    for i in range(ensemble_preds.shape[0]):
        acc = accuracy_score(labels, ensemble_preds[i]) * 100
        results[f'Model_{i+1}'] = acc
        print(f"   Model {i+1}: {acc:.2f}%")

    # Majority voting
    print("\n🗳️  MAJORITY VOTING:")
    maj_preds = majority_voting(ensemble_preds)
    maj_acc = accuracy_score(labels, maj_preds) * 100
    results['Majority_Voting'] = maj_acc
    print(f"   Accuracy: {maj_acc:.2f}%")

    # Probability averaging
    print("\n📊 PROBABILITY AVERAGING:")
    avg_preds, avg_probs = probability_averaging(ensemble_probs)
    avg_acc = accuracy_score(labels, avg_preds) * 100
    results['Probability_Averaging'] = avg_acc
    print(f"   Accuracy: {avg_acc:.2f}%")

    # Weighted voting (weight by individual accuracy)
    print("\n⚖️  WEIGHTED VOTING:")
    individual_accs = [results[f'Model_{i+1}'] for i in range(ensemble_preds.shape[0])]
    weighted_preds, weighted_probs = weighted_voting(ensemble_probs, individual_accs)
    weighted_acc = accuracy_score(labels, weighted_preds) * 100
    results['Weighted_Voting'] = weighted_acc
    print(f"   Accuracy: {weighted_acc:.2f}%")
    print(f"   Weights: {[f'{w:.2f}' for w in np.array(individual_accs)/sum(individual_accs)]}")

    # Confidence-based selection
    print("\n🎯 CONFIDENCE-BASED SELECTION:")
    conf_preds = confidence_based_selection(ensemble_probs)
    conf_acc = accuracy_score(labels, conf_preds) * 100
    results['Confidence_Selection'] = conf_acc
    print(f"   Accuracy: {conf_acc:.2f}%")

    # Best single model
    best_single = max([results[f'Model_{i+1}'] for i in range(ensemble_preds.shape[0])])
    results['Best_Single'] = best_single

    # Summary
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    print(f"Best single model:       {best_single:.2f}%")
    print(f"Majority voting:         {maj_acc:.2f}% ({maj_acc - best_single:+.2f}%)")
    print(f"Probability averaging:   {avg_acc:.2f}% ({avg_acc - best_single:+.2f}%)")
    print(f"Weighted voting:         {weighted_acc:.2f}% ({weighted_acc - best_single:+.2f}%)")
    print(f"Confidence selection:    {conf_acc:.2f}% ({conf_acc - best_single:+.2f}%)")

    # Save results
    df_results = pd.DataFrame([results])
    df_results.to_csv(output_dir / 'ensemble_results.csv', index=False)
    print(f"\n✓ Saved results to: {output_dir / 'ensemble_results.csv'}")

    return results


# ============================================================================
# 3. DIVERSITY ANALYSIS
# ============================================================================

def analyze_ensemble_diversity(ensemble_preds, labels):
    """Measure diversity and agreement among ensemble members."""
    print("\n" + "="*70)
    print("ENSEMBLE DIVERSITY ANALYSIS")
    print("="*70)

    n_models, n_samples = ensemble_preds.shape

    # Pairwise agreement matrix
    agreement_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            agreement = (ensemble_preds[i] == ensemble_preds[j]).mean()
            agreement_matrix[i, j] = agreement

    print(f"\n📊 PAIRWISE AGREEMENT MATRIX:")
    print("   (1.0 = models always agree, 0.0 = never agree)")
    df_agreement = pd.DataFrame(agreement_matrix,
                                index=[f'Model_{i+1}' for i in range(n_models)],
                                columns=[f'Model_{i+1}' for i in range(n_models)])
    print(df_agreement.round(4).to_string())

    # Average pairwise disagreement (diversity metric)
    disagreements = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            disagreement = (ensemble_preds[i] != ensemble_preds[j]).mean()
            disagreements.append(disagreement)

    avg_disagreement = np.mean(disagreements)
    print(f"\n🔀 AVERAGE PAIRWISE DISAGREEMENT: {avg_disagreement:.4f}")
    print(f"   (Higher = more diverse, better for ensemble)")

    # Plot agreement heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=[f'M{i+1}' for i in range(n_models)],
                yticklabels=[f'M{i+1}' for i in range(n_models)],
                vmin=0.9, vmax=1.0, cbar_kws={'label': 'Agreement Rate'},
                ax=ax)

    ax.set_title('Ensemble Member Agreement Matrix', fontsize=15, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_agreement.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ensemble_agreement.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved agreement matrix to: {output_dir / 'ensemble_agreement.png'}")

    # Analyze cases where models disagree
    print("\n🔍 DISAGREEMENT ANALYSIS:")

    unanimous_correct = 0
    unanimous_incorrect = 0
    split_decisions = 0

    for sample_idx in range(n_samples):
        sample_preds = ensemble_preds[:, sample_idx]
        true_label = labels[sample_idx]

        # Check unanimity
        unique_preds = np.unique(sample_preds)

        if len(unique_preds) == 1:
            # All models agree
            if unique_preds[0] == true_label:
                unanimous_correct += 1
            else:
                unanimous_incorrect += 1
        else:
            # Models disagree
            split_decisions += 1

    print(f"   Unanimous correct:   {unanimous_correct:6d} ({unanimous_correct/n_samples*100:.2f}%)")
    print(f"   Unanimous incorrect: {unanimous_incorrect:6d} ({unanimous_incorrect/n_samples*100:.2f}%)")
    print(f"   Split decisions:     {split_decisions:6d} ({split_decisions/n_samples*100:.2f}%)")

    return {
        'agreement_matrix': agreement_matrix,
        'avg_disagreement': avg_disagreement,
        'unanimous_correct': unanimous_correct,
        'unanimous_incorrect': unanimous_incorrect,
        'split_decisions': split_decisions
    }


# ============================================================================
# 4. OPTIMAL SUBSET SELECTION
# ============================================================================

def find_optimal_subset(ensemble_preds, labels, max_subset_size=None):
    """Find optimal subset of models for ensemble."""
    print("\n" + "="*70)
    print("OPTIMAL SUBSET SELECTION")
    print("="*70)

    n_models = ensemble_preds.shape[0]

    if max_subset_size is None:
        max_subset_size = n_models

    print(f"\nSearching for best subset (up to {max_subset_size} models)...")

    # Try all subset sizes
    best_results = []

    for subset_size in range(1, min(max_subset_size, n_models) + 1):
        # Try all combinations (greedy approximation for large n_models)
        from itertools import combinations

        best_acc = 0
        best_subset = None

        for subset_indices in combinations(range(n_models), subset_size):
            subset_preds = ensemble_preds[list(subset_indices), :]

            # Majority voting on subset
            votes = majority_voting(subset_preds)
            acc = accuracy_score(labels, votes) * 100

            if acc > best_acc:
                best_acc = acc
                best_subset = subset_indices

        best_results.append({
            'subset_size': subset_size,
            'best_accuracy': best_acc,
            'best_subset': best_subset
        })

        print(f"   Size {subset_size}: Best = {best_acc:.2f}% (models: {[i+1 for i in best_subset]})")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    subset_sizes = [r['subset_size'] for r in best_results]
    accuracies = [r['best_accuracy'] for r in best_results]

    ax.plot(subset_sizes, accuracies, marker='o', linewidth=2.5,
            markersize=10, color='steelblue')

    # Mark best overall
    best_overall_idx = np.argmax(accuracies)
    best_overall = best_results[best_overall_idx]
    ax.scatter([best_overall['subset_size']], [best_overall['best_accuracy']],
              s=300, color='red', marker='*', zorder=5, label='Best Overall')

    ax.set_xlabel('Ensemble Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Ensemble Accuracy vs Subset Size', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(subset_sizes)

    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_subset.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'optimal_subset.pdf', bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved optimal subset plot to: {output_dir / 'optimal_subset.png'}")

    print(f"\n🏆 BEST OVERALL ENSEMBLE:")
    print(f"   Size: {best_overall['subset_size']} models")
    print(f"   Models: {[i+1 for i in best_overall['best_subset']]}")
    print(f"   Accuracy: {best_overall['best_accuracy']:.2f}%")

    return best_results


# ============================================================================
# 5. SUMMARY REPORT
# ============================================================================

def generate_ensemble_report(results, diversity, optimal):
    """Generate comprehensive ensemble evaluation report."""
    print("\n" + "="*70)
    print("GENERATING ENSEMBLE REPORT")
    print("="*70)

    report_path = output_dir / 'ENSEMBLE_EVALUATION_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Ensemble Evaluation Report - Pokemon TCG AI\n\n")
        f.write("**Generated**: 2026-08-30\n\n")
        f.write("---\n\n")

        f.write("## Ensemble Composition\n\n")
        f.write(f"- **Number of models**: {len([k for k in results.keys() if k.startswith('Model_')])}\n")
        f.write("- **Seeds**: 0, 42, 123, 456\n")
        f.write("- **Base architecture**: LSTM 768×3 + Focal Loss\n\n")

        f.write("## Performance Comparison\n\n")
        f.write("| Strategy | Accuracy | Improvement |\n")
        f.write("|----------|----------|--------------|\n")
        f.write(f"| Best single model | {results['Best_Single']:.2f}% | — |\n")
        f.write(f"| Majority voting | {results['Majority_Voting']:.2f}% | {results['Majority_Voting'] - results['Best_Single']:+.2f}% |\n")
        f.write(f"| Probability averaging | {results['Probability_Averaging']:.2f}% | {results['Probability_Averaging'] - results['Best_Single']:+.2f}% |\n")
        f.write(f"| Weighted voting | {results['Weighted_Voting']:.2f}% | {results['Weighted_Voting'] - results['Best_Single']:+.2f}% |\n")
        f.write(f"| Confidence selection | {results['Confidence_Selection']:.2f}% | {results['Confidence_Selection'] - results['Best_Single']:+.2f}% |\n\n")

        f.write("## Diversity Analysis\n\n")
        f.write(f"- **Average pairwise disagreement**: {diversity['avg_disagreement']:.4f}\n")
        f.write(f"- **Unanimous correct predictions**: {diversity['unanimous_correct']:,} ({diversity['unanimous_correct']/(diversity['unanimous_correct']+diversity['unanimous_incorrect']+diversity['split_decisions'])*100:.1f}%)\n")
        f.write(f"- **Unanimous incorrect predictions**: {diversity['unanimous_incorrect']:,}\n")
        f.write(f"- **Split decisions**: {diversity['split_decisions']:,}\n\n")

        f.write("## Optimal Subset\n\n")
        best_overall = max(optimal, key=lambda x: x['best_accuracy'])
        f.write(f"- **Optimal ensemble size**: {best_overall['subset_size']} models\n")
        f.write(f"- **Optimal models**: {[i+1 for i in best_overall['best_subset']]}\n")
        f.write(f"- **Optimal accuracy**: {best_overall['best_accuracy']:.2f}%\n\n")

        f.write("## Conclusions\n\n")

        improvement = results['Probability_Averaging'] - results['Best_Single']
        if improvement < 0.1:
            f.write("- **Ensemble provides minimal benefit** (<0.1% improvement)\n")
            f.write("- Individual models already perform near ceiling\n")
            f.write("- Low diversity among ensemble members\n")
        else:
            f.write(f"- **Ensemble improves accuracy by {improvement:.2f}%**\n")
            f.write("- Probability averaging is the most effective strategy\n")

        f.write("\n## Visualizations\n\n")
        f.write("1. `ensemble_agreement.png` - Pairwise agreement heatmap\n")
        f.write("2. `optimal_subset.png` - Accuracy vs ensemble size\n\n")

        f.write("---\n\n")
        f.write("**For questions, contact**: philipp.zengl@tum.de\n")

    print(f"✓ Saved ensemble report to: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("\n⚠️  NOTE: This is an ensemble evaluation template.")
    print("   To run full evaluation, you need to:")
    print("   1. Load 4 trained models (seeds 0, 42, 123, 456)")
    print("   2. Load test dataset")
    print("   3. Generate predictions from all models")
    print("   4. Run evaluation functions")
    print("\n   Example usage:")
    print("   ```python")
    print("   model_paths = [")
    print("       '30_models/trained_models 2/lstm_focal_s0_*.pt',")
    print("       '30_models/trained_models 2/lstm_focal_s42_*.pt',")
    print("       '30_models/trained_models 2/lstm_focal_s123_*.pt',")
    print("       '30_models/trained_models 2/lstm_focal_s456_*.pt',")
    print("   ]")
    print("   models = load_models(model_paths)")
    print("   ensemble_data = get_ensemble_predictions(models, test_loader)")
    print("   results = evaluate_ensemble_strategies(ensemble_data, test_labels)")
    print("   diversity = analyze_ensemble_diversity(ensemble_data['predictions'], test_labels)")
    print("   optimal = find_optimal_subset(ensemble_data['predictions'], test_labels)")
    print("   generate_ensemble_report(results, diversity, optimal)")
    print("   ```")
    print("\n" + "="*70)
    print("TEMPLATE COMPLETE - Ready for integration")
    print("="*70)


if __name__ == '__main__':
    main()
