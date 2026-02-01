"""
Custom CNN for pitch detection from CQT spectrograms.

ARCHITECTURE OVERVIEW:
    Input: CQT spectrogram (1, 84, T) — like a grayscale image
      |
      v
    [Conv2D -> BatchNorm -> ReLU -> MaxPool] x3   (learn frequency patterns)
      |
      v
    [Global Average Pooling]                       (collapse spatial dims)
      |
      v
    [Dense -> ReLU -> Dropout -> Dense(1)]         (predict frequency in Hz)

WHY THIS DESIGN:
- Conv2D layers learn to detect harmonic patterns in the spectrogram
  (e.g., "energy at bins 20, 32, 40" = a specific note's harmonics)
- BatchNorm stabilizes training (normalizes each layer's output)
- MaxPool reduces spatial size, making the model focus on "what" not "where"
- Global Average Pooling at the end makes the model accept any input length
- Dropout (30%) randomly zeros neurons during training to prevent overfitting
- Single output neuron predicts frequency in Hz (regression)

WHY REGRESSION vs CLASSIFICATION:
- CREPE uses classification (360 bins, each = 20 cents). This works great
  with lots of data because the model just picks the most likely bin.
- We use regression (predict Hz directly) because it's simpler to understand
  and works fine for a small model. The tradeoff: regression can produce
  predictions outside the valid range, and the loss landscape is harder.
"""

import torch.nn as nn


class PitchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: detect basic spectral shapes
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: combine basic shapes into harmonic patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: higher-level pitch features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # AdaptiveAvgPool collapses (H, W) to (1, 1) regardless of input size.
            # This lets us accept spectrograms of any time length.
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Single output: predicted frequency in Hz
        )

    def forward(self, x):
        # x shape: (batch, 1, n_bins, n_frames) — like a batch of grayscale images
        x = self.conv_blocks(x)  # -> (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten -> (batch, 128)
        x = self.head(x)  # -> (batch, 1)
        return x
