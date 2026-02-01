"""
Pitch detection module — bridges ML models to the tuner app.

Two detectors:
  1. CREPEDetector: Uses the pre-trained CREPE model (accurate, production-ready)
  2. CustomDetector: Uses our hand-trained CNN (for learning, less accurate)

Both expose the same .detect(audio) interface so the app can swap between them.
"""

import numpy as np
import torch
import torchcrepe

from src.config import SAMPLE_RATE, STANDARD_TUNING
from src.features import extract_cqt
from src.model import PitchCNN


def freq_to_note_and_cents(freq, tuning=None):
    """
    Given a detected frequency, find the closest guitar string and how
    many cents sharp/flat it is.

    Args:
        freq: Detected frequency in Hz
        tuning: Dict of {note_label: target_hz}. Defaults to STANDARD_TUNING.

    Returns:
        (note_name, target_freq, cents_deviation)
        cents > 0 means sharp, cents < 0 means flat
    """
    if freq <= 0:
        return None, None, None

    if tuning is None:
        tuning = STANDARD_TUNING

    closest_note = None
    min_cents = float("inf")

    for note, target_freq in tuning.items():
        # cents = 1200 * log2(detected / target)
        # Positive = sharp (too high), negative = flat (too low)
        cents = 1200 * np.log2(freq / target_freq)
        if abs(cents) < abs(min_cents):
            min_cents = cents
            closest_note = note

    return closest_note, tuning[closest_note], min_cents


class CREPEDetector:
    """
    Pitch detection using CREPE (Convolutional Representation for Pitch Estimation).

    CREPE is a 6-layer CNN trained on millions of audio samples. It takes raw
    audio (not spectrograms) and outputs a probability distribution over 360
    pitch bins spanning C1 to B7. Each bin = 20 cents.

    We use the 'tiny' model variant for speed (fewer parameters).
    """

    def __init__(self, model_capacity="tiny"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_capacity = model_capacity

    def detect(self, audio, sr=SAMPLE_RATE):
        """
        Detect pitch from an audio signal.

        Returns:
            (frequency_hz, confidence)
        """
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        # torchcrepe returns pitch (Hz) and periodicity (confidence 0-1)
        frequency, confidence = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=512,
            fmin=50,
            fmax=550,
            model=self.model_capacity,
            batch_size=1,
            device=self.device,
            return_periodicity=True,
        )

        # Filter out low-confidence frames and take the median
        freq_np = frequency.squeeze().cpu().numpy()
        conf_np = confidence.squeeze().cpu().numpy()
        mask = conf_np > 0.5
        if mask.any():
            freq = float(np.median(freq_np[mask]))
            conf = float(np.mean(conf_np[mask]))
        else:
            freq = 0.0
            conf = 0.0

        return freq, conf


class CustomDetector:
    """
    Pitch detection using our custom-trained CNN.

    Unlike CREPE (which takes raw audio), our model takes CQT spectrograms
    as input and directly predicts frequency in Hz (regression).
    """

    def __init__(self, model_path="models/pitch_cnn.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PitchCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def detect(self, audio, sr=SAMPLE_RATE):
        """
        Detect pitch from an audio signal.

        Returns:
            (frequency_hz, confidence)
            Note: our simple model doesn't produce a confidence score,
            so we always return 1.0.
        """
        cqt = extract_cqt(audio, sr)
        # Shape: (n_bins, n_frames) -> (1, 1, n_bins, n_frames) for the CNN
        spec_tensor = (
            torch.FloatTensor(cqt).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            freq = self.model(spec_tensor).item()

        return max(0.0, freq), 1.0
