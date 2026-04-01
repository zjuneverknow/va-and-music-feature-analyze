import csv
from pathlib import Path

from tools import analyze_mp3_features


DATASET_DIR = Path("dataset/MEMD_audio")
OUTPUT_CSV = Path("dataset/memd_audio_features.csv")


def main():
    audio_files = sorted(DATASET_DIR.glob("*.mp3"))
    if not audio_files:
        raise FileNotFoundError(f"No mp3 files found in {DATASET_DIR}")

    rows = []
    failed_files = []

    for index, audio_file in enumerate(audio_files, start=1):
        try:
            features = analyze_mp3_features(audio_file)
            rows.append(
                {
                    "file_name": audio_file.name,
                    "file_stem": audio_file.stem,
                    "file_path": str(audio_file),
                    **features,
                }
            )
            print(f"[{index}/{len(audio_files)}] analyzed {audio_file.name}")
        except Exception as exc:
            failed_files.append({"file_name": audio_file.name, "error": str(exc)})
            print(f"[{index}/{len(audio_files)}] failed {audio_file.name}: {exc}")

    if not rows:
        raise RuntimeError("No audio features were extracted.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

    if failed_files:
        failed_csv = OUTPUT_CSV.with_name("memd_audio_features_failed.csv")
        with failed_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["file_name", "error"])
            writer.writeheader()
            writer.writerows(failed_files)
        print(f"Saved {len(failed_files)} failures to {failed_csv}")


if __name__ == "__main__":
    main()
