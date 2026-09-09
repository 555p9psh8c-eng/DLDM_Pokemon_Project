"""compare_models.py — Compare original and improved Transformer architectures.

This script loads both model checkpoints and provides a detailed comparison of:
- Architecture differences (layers, dimensions, parameters)
- Performance differences (test accuracy, top-3 accuracy)
- Training differences (epochs, convergence speed)

Usage:
    python compare_models.py
    python compare_models.py --original Transformer_baseline.pt --improved Transformer_improved.pt
"""

import argparse
import torch


def count_parameters(state_dict):
    """Count total parameters in a model state dict."""
    return sum(p.numel() for p in state_dict.values())


def analyze_checkpoint(path):
    """Load and analyze a model checkpoint."""
    print(f"\nLoading {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    state_dict = ckpt["state_dict"]

    # Extract architecture info
    info = {
        "path": path,
        "config": config,
        "n_params": count_parameters(state_dict),
        "test_acc": ckpt.get("test_acc", None),
        "test_top3": ckpt.get("test_top3", None),
        "val_acc": ckpt.get("val_acc", None),
    }

    # Architecture details
    info["d_model"] = config.get("d_model", "unknown")
    info["n_layers"] = config.get("n_layers", "unknown")
    info["n_heads"] = config.get("n_heads", "unknown")
    info["d_ff"] = config.get("d_ff", "unknown")
    info["dropout"] = config.get("dropout", "unknown")
    info["lr"] = config.get("lr", "unknown")
    info["epochs"] = config.get("epochs", "unknown")
    info["warmup"] = config.get("warmup_steps", config.get("warmup", "unknown"))

    return info


def print_comparison(original, improved):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("MODEL COMPARISON: Original vs Improved Transformer")
    print("="*100)

    print("\n### ARCHITECTURE ###")
    print(f"{'Metric':<20} {'Original':<25} {'Improved':<25} {'Change':<20}")
    print("-"*90)

    def format_change(orig, impr):
        if isinstance(orig, (int, float)) and isinstance(impr, (int, float)):
            pct = ((impr - orig) / orig * 100) if orig != 0 else 0
            return f"+{impr-orig:,} ({pct:+.0f}%)"
        return "N/A"

    metrics = [
        ("d_model", "d_model", "Model Dimension"),
        ("n_layers", "n_layers", "Layers"),
        ("n_heads", "n_heads", "Attention Heads"),
        ("d_ff", "d_ff", "FF Dimension"),
        ("n_params", "n_params", "Total Parameters"),
    ]

    for key, attr, label in metrics:
        orig_val = original[attr]
        impr_val = improved[attr]
        change = format_change(orig_val, impr_val)
        print(f"{label:<20} {str(orig_val):<25} {str(impr_val):<25} {change:<20}")

    print("\n### TRAINING CONFIG ###")
    print(f"{'Metric':<20} {'Original':<25} {'Improved':<25} {'Change':<20}")
    print("-"*90)

    training_metrics = [
        ("dropout", "dropout", "Dropout"),
        ("lr", "lr", "Learning Rate"),
        ("warmup", "warmup", "Warmup Steps"),
        ("epochs", "epochs", "Max Epochs"),
    ]

    for key, attr, label in training_metrics:
        orig_val = original[attr]
        impr_val = improved[attr]
        change = format_change(orig_val, impr_val) if isinstance(orig_val, (int, float)) else "N/A"
        print(f"{label:<20} {str(orig_val):<25} {str(impr_val):<25} {change:<20}")

    print("\n### PERFORMANCE ###")
    print(f"{'Metric':<20} {'Original':<25} {'Improved':<25} {'Change':<20}")
    print("-"*90)

    if original["test_acc"] is not None and improved["test_acc"] is not None:
        orig_test = original["test_acc"] * 100
        impr_test = improved["test_acc"] * 100
        test_change = f"+{impr_test - orig_test:.2f} pts"
        print(f"{'Test Accuracy':<20} {f'{orig_test:.2f}%':<25} {f'{impr_test:.2f}%':<25} {test_change:<20}")

    if original["test_top3"] is not None and improved["test_top3"] is not None:
        orig_top3 = original["test_top3"] * 100
        impr_top3 = improved["test_top3"] * 100
        top3_change = f"+{impr_top3 - orig_top3:.2f} pts"
        print(f"{'Test Top-3 Acc':<20} {f'{orig_top3:.2f}%':<25} {f'{impr_top3:.2f}%':<25} {top3_change:<20}")

    if original["val_acc"] is not None and improved["val_acc"] is not None:
        orig_val = original["val_acc"] * 100
        impr_val = improved["val_acc"] * 100
        val_change = f"+{impr_val - orig_val:.2f} pts"
        print(f"{'Val Accuracy':<20} {f'{orig_val:.2f}%':<25} {f'{impr_val:.2f}%':<25} {val_change:<20}")

    print("\n### COMPARISON TO BASELINES ###")
    print(f"{'Model':<30} {'Test Accuracy':<20} {'vs LSTM (65.7%)':<25}")
    print("-"*75)
    print(f"{'LSTM (baseline)':<30} {'65.7%':<20} {'-':<25}")

    if original["test_acc"] is not None:
        orig_acc = original["test_acc"] * 100
        orig_vs_lstm = f"+{orig_acc - 65.7:.2f} pts"
        print(f"{'Transformer (original)':<30} {f'{orig_acc:.2f}%':<20} {orig_vs_lstm:<25}")

    if improved["test_acc"] is not None:
        impr_acc = improved["test_acc"] * 100
        impr_vs_lstm = f"+{impr_acc - 65.7:.2f} pts"
        print(f"{'Transformer (improved)':<30} {f'{impr_acc:.2f}%':<20} {impr_vs_lstm:<25}")

    print("\n### SUMMARY ###")
    if original["test_acc"] is not None and improved["test_acc"] is not None:
        gain = (improved["test_acc"] - original["test_acc"]) * 100
        param_ratio = improved["n_params"] / original["n_params"]
        efficiency = gain / param_ratio if param_ratio > 1 else gain

        print(f"Accuracy gain: +{gain:.2f} percentage points")
        print(f"Parameter increase: {param_ratio:.1f}x")
        print(f"Efficiency (gain per param ratio): {efficiency:.2f} pts/x")

        if gain >= 5.0:
            print(f"\n✓ SUCCESS: Improved model beats original by {gain:.1f} points (target: 5+ points)")
        elif gain >= 3.0:
            print(f"\n⚠ PARTIAL: Improved model gains {gain:.1f} points (target: 5+ points)")
        else:
            print(f"\n✗ INSUFFICIENT: Improved model gains only {gain:.1f} points (target: 5+ points)")

    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare original and improved Transformers")
    parser.add_argument("--original", default="Transformer_baseline.pt",
                        help="path to original transformer checkpoint")
    parser.add_argument("--improved", default="Transformer_improved.pt",
                        help="path to improved transformer checkpoint")
    args = parser.parse_args()

    try:
        original = analyze_checkpoint(args.original)
    except FileNotFoundError:
        print(f"\nERROR: Original checkpoint not found: {args.original}")
        print("Run train_Transformer.py first to create the baseline.")
        return

    try:
        improved = analyze_checkpoint(args.improved)
    except FileNotFoundError:
        print(f"\nERROR: Improved checkpoint not found: {args.improved}")
        print("Run train_Transformer_improved.py first to create the improved model.")
        return

    print_comparison(original, improved)


if __name__ == "__main__":
    main()
