import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None


DEFAULT_INPUT = Path("dataset/pred_compare.csv")
DEFAULT_OUTPUT_DIR = Path("analyze/output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze prediction-vs-ground-truth results and optionally export plots."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to pred_compare.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only export summary CSV without generating plots.",
    )
    return parser.parse_args()


def infer_feature_names(df):
    return sorted(col[:-5] for col in df.columns if col.endswith("_pred") and col[:-5] in df.columns)


def summarize_feature(df, feature_name):
    actual = df[feature_name].to_numpy(dtype=float)
    predicted = df[f"{feature_name}_pred"].to_numpy(dtype=float)
    error = predicted - actual
    abs_error = np.abs(error)

    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
        corr = np.corrcoef(actual, predicted)[0, 1]
    else:
        corr = np.nan
    return {
        "feature": feature_name,
        "count": len(df),
        "mean_actual": float(np.mean(actual)),
        "mean_pred": float(np.mean(predicted)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "corr": float(corr),
    }


def plot_feature(df, feature_name, output_dir):
    if plt is None or sns is None:
        return None

    actual = df[feature_name]
    predicted = df[f"{feature_name}_pred"]
    error = predicted - actual

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=actual, y=predicted, alpha=0.45, ax=axes[0])
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    axes[0].set_title(f"{feature_name}: Actual vs Predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].grid(True, alpha=0.25)

    sns.histplot(error, kde=True, color="orange", ax=axes[1])
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{feature_name}: Prediction Error")
    axes[1].set_xlabel("Predicted - Actual")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    output_path = output_dir / f"{feature_name}_compare.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = infer_feature_names(df)
    if not feature_names:
        raise ValueError("No feature comparison columns found. Expected columns like tempo and tempo_pred.")

    summary_rows = []
    plot_paths = []
    for feature_name in feature_names:
        summary_rows.append(summarize_feature(df, feature_name))
        if not args.no_plots:
            plot_path = plot_feature(df, feature_name, args.output_dir)
            if plot_path is not None:
                plot_paths.append(plot_path)

    summary_df = pd.DataFrame(summary_rows).sort_values("mae")
    summary_path = args.output_dir / "pred_compare_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved metrics to {summary_path}")
    if args.no_plots:
        print("Skipped plot generation because --no-plots was used.")
    elif plt is None or sns is None:
        print("Metrics were generated, but plots were skipped because matplotlib/seaborn are not installed.")
    else:
        print(f"Saved {len(plot_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
