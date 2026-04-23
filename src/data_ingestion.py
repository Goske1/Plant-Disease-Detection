"""
Plant Disease Detection — Data Ingestion
Crops: Tomato, Potato, Corn
Split: Train 60% / Validation 20% / Test 20%  (BBM406 l5-ml-methodology)
"""

import os
import shutil
import random
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
SUPPORTED_CROPS = ["Tomato", "Potato", "Corn"]
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20   # used ONLY once to report final performance

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _collect_images(source_dir: Path) -> list[Path]:
    return [
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _verify_split_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"


def split_dataset(
    raw_data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, int]]:
    """
    Reads images from raw_data_dir/<crop>/<disease_class>/ and copies them into:
        output_dir/train/<crop>/<disease_class>/
        output_dir/val/<crop>/<disease_class>/
        output_dir/test/<crop>/<disease_class>/

    Returns a summary dict: {split: {crop: count}}.
    """
    _verify_split_ratios()
    raw_data_dir = Path(raw_data_dir)
    output_dir   = Path(output_dir)

    random.seed(RANDOM_SEED)
    summary: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for crop in SUPPORTED_CROPS:
        crop_dir = raw_data_dir / crop
        if not crop_dir.exists():
            print(f"[WARN] Crop directory not found, skipping: {crop_dir}")
            continue

        for class_dir in sorted(crop_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            disease_class = class_dir.name
            images = _collect_images(class_dir)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * TRAIN_RATIO)
            n_val   = int(n_total * VAL_RATIO)

            splits = {
                "train": images[:n_train],
                "val":   images[n_train : n_train + n_val],
                "test":  images[n_train + n_val :],
            }

            for split_name, split_images in splits.items():
                dest = output_dir / split_name / crop / disease_class
                dest.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    shutil.copy2(img_path, dest / img_path.name)

                summary[split_name][f"{crop}/{disease_class}"] = len(split_images)

    _print_summary(summary, raw_data_dir)
    return summary


def _print_summary(summary: dict[str, dict[str, int]], source: Path) -> None:
    print(f"\n{'─'*55}")
    print(f"  Data Ingestion Summary  |  source: {source}")
    print(f"{'─'*55}")
    for split in ("train", "val", "test"):
        total = sum(summary[split].values())
        print(f"  {split:<6}: {total:>5} images")
    print(f"{'─'*55}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest and split plant disease images.")
    parser.add_argument("--raw",    required=True, help="Path to raw dataset root")
    parser.add_argument("--output", default="data/split", help="Destination for split data")
    args = parser.parse_args()

    split_dataset(args.raw, args.output)
