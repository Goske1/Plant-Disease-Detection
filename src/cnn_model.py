"""
Plant Disease Detection CNN
Based on course concepts: metric-space representations (l2-knn), gradient-descent loss
minimization (l4-linear_regression), and train/val/test methodology (l5-ml-methodology).
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class ModelConfig:
    num_classes: int = 15
    dropout: float = 0.5
    learning_rate: float = 1e-3


class PlantDiseaseCNN(nn.Module):
    """CNN that learns convolutional filters as local pattern detectors
    (analogous to nearest-neighbor matching in a learned feature space)."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()

        # Block 1: 3 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 2: 32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 3: 64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling collapses spatial dims → fewer parameters than FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(config.dropout)

        # Single linear layer maps the 128-dim representation to class scores
        self.classifier = nn.Linear(128, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)   # (batch, 128, 1, 1) → (batch, 128)
        x = self.dropout(x)
        return self.classifier(x)


def build_criterion() -> nn.CrossEntropyLoss:
    # CrossEntropyLoss is the classification analogue of MSE: minimize prediction error
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, lr: float) -> optim.Adam:
    return optim.Adam(model.parameters(), lr=lr)


def model_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    config = ModelConfig()
    model = PlantDiseaseCNN(config)

    print("=== PlantDiseaseCNN Summary ===")
    model_summary(model)

    # Single forward pass to verify output shape — select model on val, report on test
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    print(f"\nInput  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expected: [1, {config.num_classes}])")
    assert out.shape == (1, config.num_classes), "Shape mismatch!"
    print("Forward pass OK.")
