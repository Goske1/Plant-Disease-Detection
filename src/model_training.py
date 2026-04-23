"""
Plant Disease Detection — Model Training
Methodology: BBM406 l5-ml-methodology (Hacettepe University)
  - Use VALIDATION set for model selection & hyperparameter tuning
  - Use TEST set ONLY ONCE to report final performance
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.models as models
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed — cannot run training.")

from preprocessing import PlantDiseaseDataset, build_train_transform, build_eval_transform


# ── Hyperparameters (tune via validation set, never via test set) ──────────────
CONFIG = {
    "backbone":      "resnet18",    # candidates: resnet18, efficientnet_b0, mobilenet_v3_small
    "num_epochs":    30,
    "batch_size":    32,
    "learning_rate": 1e-3,
    "weight_decay":  1e-4,
    "lr_scheduler":  "cosine",      # step | cosine
    "early_stop_patience": 5,       # stop if val loss doesn't improve for N epochs
    "num_workers":   4,
    "device":        "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
}


# ── Model builder ──────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> "nn.Module":
    backbone = CONFIG["backbone"]
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ── Training loop ──────────────────────────────────────────────────────────────
def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def train(data_root: str | Path, output_dir: str | Path) -> dict:
    """
    Trains the model, selects the best checkpoint via validation loss,
    and evaluates ONCE on the test set.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training.")

    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = PlantDiseaseDataset(data_root, "train", build_train_transform())
    val_ds   = PlantDiseaseDataset(data_root, "val",   build_eval_transform())
    test_ds  = PlantDiseaseDataset(data_root, "test",  build_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"])

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG["learning_rate"],
                            weight_decay=CONFIG["weight_decay"])

    if CONFIG["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ── Training & validation ────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0
    history         = []
    best_ckpt_path  = output_dir / "best_model.pth"

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{CONFIG['num_epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        # Save best checkpoint (model selection via validation loss)
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= CONFIG["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} — val_loss hasn't improved for "
                      f"{CONFIG['early_stop_patience']} epochs.")
                break

    # ── Final test evaluation (done ONCE, after model selection) ─────────────
    print("\nLoading best checkpoint for final test evaluation …")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"\n{'─'*50}")
    print(f"  Final Test Loss : {test_loss:.4f}")
    print(f"  Final Test Acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"{'─'*50}\n")

    results = {
        "config":    CONFIG,
        "classes":   train_ds.classes,
        "history":   history,
        "test_loss": test_loss,
        "test_acc":  test_acc,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train plant disease detection model.")
    parser.add_argument("--data",   required=True,         help="Path to split data root")
    parser.add_argument("--output", default="models/run1", help="Output directory for checkpoints")
    args = parser.parse_args()

    train(args.data, args.output)
