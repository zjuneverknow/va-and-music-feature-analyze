import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import INPUT_COLS, MODEL_PATH, load_model, sample_music_params


DATA_PATH = Path("dataset/audio_va.csv")
OUTPUT_PATH = Path("dataset/pred_compare.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ground-truth features with predicted features.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to audio_va.csv.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to saved model file.")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH, help="Path to output CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible predictions.")
    return parser.parse_args()


def build_compare_table(data_path, model_path, seed):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    model, scaler, feature_names = load_model(model_path)
    rng = np.random.default_rng(seed)

    output_cols = feature_names[len(INPUT_COLS):]
    rows = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        predicted = sample_music_params(
            valence=float(row_dict["valence_mean"]),
            valence_std=float(row_dict["valence_std"]),
            arousal=float(row_dict["arousal_mean"]),
            arousal_std=float(row_dict["arousal_std"]),
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            rng=rng,
        )

        compare_row = {
            "song_id": row_dict.get("song_id"),
            "valence_mean": row_dict["valence_mean"],
            "valence_std": row_dict["valence_std"],
            "arousal_mean": row_dict["arousal_mean"],
            "arousal_std": row_dict["arousal_std"],
        }

        for col in output_cols:
            compare_row[col] = row_dict[col]
            compare_row[f"{col}_pred"] = predicted[col]

        rows.append(compare_row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    compare_df = build_compare_table(args.data_path, args.model_path, args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(args.output_path, index=False)
    print(f"Saved {len(compare_df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
