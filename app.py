"""
Streamlit app for G-Tuner.

Run with:  streamlit run app.py

Real-time mode: uses st.rerun() to create a continuous listen-detect-display
loop. Each cycle records a short audio chunk, runs pitch detection, displays
the result, then reruns the script to capture the next chunk.
"""

import streamlit as st
import numpy as np
import sounddevice as sd

from src.config import SAMPLE_RATE, TUNINGS
from src.pitch import CREPEDetector, CustomDetector, freq_to_note_and_cents

st.set_page_config(page_title="G-Tuner", layout="centered")
st.title("G-Tuner")
st.caption("ML-powered guitar tuner")

# --- Session state init ---
if "listening" not in st.session_state:
    st.session_state.listening = False
if "last_note" not in st.session_state:
    st.session_state.last_note = None
    st.session_state.last_freq = 0.0
    st.session_state.last_cents = 0.0

# --- Tuning and model selection ---
col_tuning, col_model = st.columns(2)
with col_tuning:
    tuning_name = st.selectbox("Tuning", list(TUNINGS.keys()))
with col_model:
    model_choice = st.selectbox("Model", ["CREPE (Pre-trained)", "Custom CNN"])

active_tuning = TUNINGS[tuning_name]


@st.cache_resource
def load_crepe():
    return CREPEDetector()


@st.cache_resource
def load_custom():
    return CustomDetector()


# --- Show target frequencies ---
st.subheader(tuning_name)
cols = st.columns(6)
for i, (note, freq) in enumerate(active_tuning.items()):
    cols[i].metric(note, f"{freq:.0f} Hz")

st.divider()

# --- Start / Stop toggle ---
def toggle_listening():
    st.session_state.listening = not st.session_state.listening
    if not st.session_state.listening:
        st.session_state.last_note = None

if st.session_state.listening:
    st.button("Stop Listening", on_click=toggle_listening, type="primary")
else:
    st.button("Start Listening", on_click=toggle_listening, type="primary")

# --- Display results ---
result_container = st.empty()

if st.session_state.last_note is not None:
    note = st.session_state.last_note
    freq = st.session_state.last_freq
    cents = st.session_state.last_cents

    with result_container.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Note", note)
        col2.metric("Frequency", f"{freq:.1f} Hz")
        col3.metric("Cents Off", f"{cents:+.1f}")

        if abs(cents) < 5:
            st.success("In tune!")
        elif cents > 0:
            st.warning(f"Sharp by {cents:.1f} cents -- tune down")
        else:
            st.warning(f"Flat by {abs(cents):.1f} cents -- tune up")

# --- Continuous listening loop ---
if st.session_state.listening:
    if model_choice == "CREPE (Pre-trained)":
        detector = load_crepe()
    else:
        detector = load_custom()

    # Record a short chunk (0.5s is enough for pitch detection and feels responsive)
    audio = sd.rec(
        int(0.5 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    freq, confidence = detector.detect(audio)

    if freq > 0:
        note, target_freq, cents = freq_to_note_and_cents(freq, active_tuning)
        st.session_state.last_note = note
        st.session_state.last_freq = freq
        st.session_state.last_cents = cents
    else:
        st.session_state.last_note = None

    # Rerun to capture the next chunk — this creates the continuous loop
    st.rerun()
