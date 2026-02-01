"""
Synthetic dataset generator for training our pitch detection CNN.

WHY SYNTHETIC DATA?
Real guitar recordings need manual labeling (tedious). Instead, we generate
audio signals with known frequencies — giving us perfect ground-truth labels
for free. We simulate guitar-like sounds by combining:
  1. A fundamental frequency (the note's pitch)
  2. Harmonics (integer multiples of the fundamental — this is what gives
     instruments their unique timbre/tone color)
  3. Random noise (simulates real-world recording conditions)
"""

import numpy as np
from src.config import SAMPLE_RATE


def generate_harmonic_tone(f0, duration=1.0, sr=SAMPLE_RATE, n_harmonics=5, noise_level=0.01):
    """
    Generate an audio signal that mimics a plucked string.

    Args:
        f0: Fundamental frequency in Hz (the "pitch" we hear)
        duration: Length in seconds
        sr: Sample rate
        n_harmonics: Number of overtones. A pure sine wave has 0 harmonics.
                     Real guitar strings produce 5-15+ harmonics.
        noise_level: How much random noise to add (0 = clean, 1 = all noise)

    Returns:
        numpy array of audio samples (float32)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Build the signal by stacking harmonics
    # A guitar string vibrating at 110 Hz (A2) also produces energy at
    # 220 Hz, 330 Hz, 440 Hz, etc. Each harmonic is quieter than the last.
    for h in range(1, n_harmonics + 1):
        amplitude = 1.0 / h  # Each harmonic is softer: 1, 1/2, 1/3, ...
        signal += amplitude * np.sin(2 * np.pi * f0 * h * t)

    # Normalize to [-1, 1] range (standard for audio)
    signal = signal / np.max(np.abs(signal))

    # Add gaussian noise to simulate real recording conditions
    signal += noise_level * np.random.randn(len(signal))

    return signal.astype(np.float32)


def generate_dataset(n_samples=3000, freq_range=(70.0, 400.0), duration=1.0):
    """
    Generate a full dataset of synthetic audio samples with labels.

    The frequency range 70-400 Hz covers all standard guitar tuning notes
    (E2=82Hz to E4=330Hz) with some margin on both sides.

    Args:
        n_samples: How many training examples to generate
        freq_range: (min_hz, max_hz) range to sample from
        duration: Length of each audio clip in seconds

    Returns:
        signals: list of numpy arrays (audio samples)
        frequencies: numpy array of ground-truth f0 values
    """
    signals = []
    frequencies = []

    for _ in range(n_samples):
        # Randomly pick a fundamental frequency (uniform distribution)
        f0 = np.random.uniform(*freq_range)

        # Randomize the number of harmonics and noise to create variety
        # This helps the model generalize — it won't overfit to one "sound"
        n_harmonics = np.random.randint(3, 8)
        noise_level = np.random.uniform(0.005, 0.05)

        signal = generate_harmonic_tone(
            f0, duration=duration,
            n_harmonics=n_harmonics,
            noise_level=noise_level,
        )
        signals.append(signal)
        frequencies.append(f0)

    return signals, np.array(frequencies, dtype=np.float32)
