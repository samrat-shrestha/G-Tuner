"""
Feature extraction: converting raw audio into ML-friendly input.

WHY NOT FEED RAW AUDIO DIRECTLY?
Raw audio is a 1D signal (just amplitude over time). While models like CREPE
do work on raw audio, it's much easier for a small CNN to learn from a
spectrogram — a 2D image-like representation showing frequency content over time.

We use the Constant-Q Transform (CQT) because:
  - Its frequency bins are logarithmically spaced, matching musical note spacing
  - Each octave has the same number of bins (12 = one per semitone)
  - This means the pattern for "A2" and "A3" looks similar, just shifted up
  - Regular FFT/STFT uses linear spacing, which wastes resolution on high
    frequencies and lacks resolution at low frequencies (where guitar lives)
"""

import librosa
import numpy as np
from src.config import SAMPLE_RATE, N_BINS, BINS_PER_OCTAVE, FMIN, HOP_SIZE


def extract_cqt(audio, sr=SAMPLE_RATE):
    """
    Extract a CQT spectrogram from an audio signal.

    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate

    Returns:
        2D numpy array of shape (n_bins, n_frames) in decibels.
        Think of it like an image:
          - Y axis (rows) = frequency bins (low to high)
          - X axis (cols) = time frames
          - Pixel value = energy at that frequency and time (in dB)
    """
    cqt = librosa.cqt(
        audio,
        sr=sr,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN,
        hop_length=HOP_SIZE,
    )

    # cqt is complex-valued (has magnitude + phase).
    # We only care about magnitude (how loud each frequency is).
    # Convert to decibels for better dynamic range (like how humans hear).
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    return cqt_db
