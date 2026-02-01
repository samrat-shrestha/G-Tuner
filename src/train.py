"""
Training script for the custom pitch detection CNN.

ML TRAINING LOOP — THE CORE IDEA:
  1. Forward pass:  feed data through model, get predictions
  2. Compute loss:  measure how wrong the predictions are (MSE loss)
  3. Backward pass:  compute gradients (how to nudge each weight to reduce loss)
  4. Update weights: optimizer adjusts weights using gradients
  5. Repeat for many epochs (full passes through the dataset)

WHAT IS AN EPOCH?
  One full pass through the entire training dataset. If you have 2400 training
  samples and batch_size=32, one epoch = 75 batches (2400/32).

WHAT IS MSE LOSS?
  Mean Squared Error = average of (predicted - actual)^2
  Penalizes large errors more than small ones. If we predict 115 Hz for a
  110 Hz note, loss = (115-110)^2 = 25. For 200 Hz, loss = 8100. This makes
  the model strongly avoid big mistakes.

WHAT IS CENT ERROR?
  Musicians measure pitch accuracy in "cents". 100 cents = 1 semitone.
  Cent error = 1200 * |log2(predicted/actual)|
  Under 10 cents is considered "in tune" for most practical purposes.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from src.dataset import generate_dataset
from src.features import extract_cqt
from src.model import PitchCNN


class PitchDataset(Dataset):
    """PyTorch Dataset wrapping our spectrograms and frequency labels."""

    def __init__(self, spectrograms, frequencies):
        self.spectrograms = spectrograms
        self.frequencies = frequencies

    def __len__(self):
        return len(self.frequencies)

    def __getitem__(self, idx):
        # unsqueeze(0) adds a channel dimension: (H, W) -> (1, H, W)
        # CNNs expect input shaped like images: (channels, height, width)
        spec = torch.FloatTensor(self.spectrograms[idx]).unsqueeze(0)
        freq = torch.FloatTensor([self.frequencies[idx]])
        return spec, freq


def hz_to_cents(predicted, target):
    """Convert Hz difference to cents (musical pitch unit)."""
    predicted = np.clip(predicted, 1e-7, None)
    target = np.clip(target, 1e-7, None)
    return 1200 * np.abs(np.log2(predicted / target))


def train():
    # --- Step 1: Generate synthetic training data ---
    print("Generating synthetic dataset...")
    signals, frequencies = generate_dataset(n_samples=3000)

    # --- Step 2: Extract features (audio -> spectrograms) ---
    print("Extracting CQT features...")
    spectrograms = [extract_cqt(s) for s in signals]

    # Pad all spectrograms to the same width (time dimension)
    # Different audio lengths -> different numbers of CQT frames.
    # Neural networks need fixed-size input within a batch.
    max_frames = max(s.shape[1] for s in spectrograms)
    spectrograms_padded = []
    for s in spectrograms:
        if s.shape[1] < max_frames:
            s = np.pad(s, ((0, 0), (0, max_frames - s.shape[1])))
        spectrograms_padded.append(s)
    spectrograms = np.array(spectrograms_padded)

    # --- Step 3: Train/validation split ---
    # 80% training, 20% validation. Validation data is NEVER trained on —
    # it tells us how the model performs on unseen data (generalization).
    split = int(0.8 * len(frequencies))
    train_ds = PitchDataset(spectrograms[:split], frequencies[:split])
    val_ds = PitchDataset(spectrograms[split:], frequencies[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- Step 4: Set up model, loss function, and optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchCNN().to(device)
    # MSE loss: penalizes squared difference between predicted and actual Hz
    criterion = nn.MSELoss()
    # Adam optimizer: adaptive learning rate per parameter (most popular choice)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {device} for {EPOCHS} epochs...")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print()

    # --- Step 5: Training loop ---
    for epoch in range(EPOCHS):
        # TRAINING PHASE
        model.train()  # Enable dropout, batchnorm in training mode
        train_loss = 0
        for specs, freqs in train_loader:
            specs, freqs = specs.to(device), freqs.to(device)

            optimizer.zero_grad()     # Reset gradients from previous batch
            preds = model(specs)      # Forward pass
            loss = criterion(preds, freqs)  # Compute loss
            loss.backward()           # Backward pass (compute gradients)
            optimizer.step()          # Update weights
            train_loss += loss.item()

        # VALIDATION PHASE
        model.eval()  # Disable dropout, batchnorm in eval mode
        val_cents_errors = []
        with torch.no_grad():  # No gradients needed for validation
            for specs, freqs in val_loader:
                specs, freqs = specs.to(device), freqs.to(device)
                preds = model(specs)
                cents = hz_to_cents(
                    preds.cpu().numpy(),
                    freqs.cpu().numpy(),
                )
                val_cents_errors.extend(cents.flatten())

        avg_loss = train_loss / len(train_loader)
        avg_cents = np.mean(val_cents_errors)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Cents Error: {avg_cents:.1f}")

    # --- Step 6: Save the trained model ---
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pitch_cnn.pth")
    print("\nModel saved to models/pitch_cnn.pth")

    return model


if __name__ == "__main__":
    train()
