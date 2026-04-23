"""
Plant Disease Detection — Preprocessing Pipeline
Crops: Tomato, Potato, Corn
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import torchvision.transforms as T
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[WARN] PyTorch / torchvision not installed — preprocessing will be skipped.")


# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)   # input size for most CNN backbones
MEAN          = [0.485, 0.456, 0.406]   # ImageNet stats (good starting point)
STD           = [0.229, 0.224, 0.225]
MAX_PIXEL_VAL = 255.0


# ── Transform builders ─────────────────────────────────────────────────────────
def build_train_transform():
    """Augmentation for the training split only — never apply to val/test."""
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def build_eval_transform():
    """Deterministic transform for validation and test splits."""
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


# ── Dataset class ──────────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:
    from torch.utils.data import Dataset

    class PlantDiseaseDataset(Dataset):
        """
        Folder structure expected:
            root/<split>/<crop>/<disease_class>/<image>.jpg
        """

        def __init__(self, root: str | Path, split: str, transform=None):
            assert split in ("train", "val", "test"), f"Invalid split: {split}"
            self.root      = Path(root) / split
            self.transform = transform
            self.samples: list[tuple[Path, int]] = []
            self.classes:  list[str] = []
            self._load_samples()

        def _load_samples(self) -> None:
            label_map: dict[str, int] = {}
            for class_dir in sorted(self.root.rglob("*")):
                if not class_dir.is_dir():
                    continue
                label = f"{class_dir.parent.name}/{class_dir.name}"
                if label not in label_map:
                    label_map[label] = len(label_map)
                idx = label_map[label]
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        self.samples.append((img_path, idx))
            self.classes = list(label_map.keys())

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label


# ── Numpy fallback (no torch) ─────────────────────────────────────────────────
def load_image_numpy(path: str | Path) -> np.ndarray:
    """Return a (224, 224, 3) float32 array normalised to [0, 1]."""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.array(img, dtype=np.float32) / MAX_PIXEL_VAL


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Verify preprocessing pipeline.")
    parser.add_argument("--data", required=True, help="Path to split data root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    if _TORCH_AVAILABLE:
        transform = build_train_transform() if args.split == "train" else build_eval_transform()
        ds = PlantDiseaseDataset(args.data, args.split, transform)
        print(f"Split '{args.split}': {len(ds)} samples, {len(ds.classes)} classes")
        img, lbl = ds[0]
        print(f"  Sample shape: {img.shape}, label: {lbl}")
    else:
        print("Install PyTorch to use full preprocessing pipeline.")
