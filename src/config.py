# Tuning profiles: dict of { display_label: frequency_hz } per string (low to high)
TUNINGS = {
    "Standard": {
        "E2": 82.41,
        "A2": 110.00,
        "D3": 146.83,
        "G3": 196.00,
        "B3": 246.94,
        "E4": 329.63,
    },
    "Seasons (Chris Cornell)": {
        "F2(6)": 87.31,
        "F2(5)": 87.31,
        "C3(4)": 130.81,
        "C3(3)": 130.81,
        "C3(2)": 130.81,
        "F3(1)": 174.61,
    },
}

# Default tuning
STANDARD_TUNING = TUNINGS["Standard"]

# Audio settings
# CREPE expects 16kHz audio, so we standardize on that
SAMPLE_RATE = 16000

# Frame size for audio capture (~64ms window at 16kHz)
FRAME_SIZE = 1024

# Hop length: how many samples we slide between frames
# Smaller = more overlap = smoother but slower
HOP_SIZE = 512

# CQT (Constant-Q Transform) settings
# CQT is like an FFT but with logarithmic frequency spacing,
# which matches how musical notes are spaced (each octave doubles in frequency)
N_BINS = 48            # Total frequency bins (4 octaves * 12 bins per octave)
BINS_PER_OCTAVE = 12   # 12 = one bin per semitone (like piano keys)
FMIN = 65.41           # Lowest frequency: C2, just below the low E string

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
