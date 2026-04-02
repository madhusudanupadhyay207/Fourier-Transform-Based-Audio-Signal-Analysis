import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Audio DFT Analyzer", layout="wide")
st.title("🎤 Audio Analysis (Upload or Record) using Manual DFT")

# Sidebar
st.sidebar.header("⚙️ Settings")
sample_limit = st.sidebar.slider("Number of samples", 500, 3000, 1000)

# Input choice
mode = st.radio("Choose input method:", ["Upload WAV file", "Record from microphone"])

audio_bytes = None

# -------- OPTION 1: Upload --------
if mode == "Upload WAV file":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()

# -------- OPTION 2: Record --------
else:
    st.info("Click record, speak, then stop.")
    audio_file = st.audio_input("Record your voice")

    if audio_file is not None:
        audio_bytes = audio_file.read()

# -------- PROCESSING --------
if audio_bytes is not None:
    # Load audio from bytes
    audio, sr = librosa.load(io.BytesIO(audio_bytes))

    audio = audio[:sample_limit]

    st.markdown("### 📊 Audio Info")
    col1, col2 = st.columns(2)
    col1.metric("Sample Rate", sr)
    col2.metric("Samples Used", len(audio))

    # Waveform
    st.markdown("### 📈 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    st.pyplot(fig1)

    # Manual DFT
    st.markdown("### ⚙️ Manual DFT")
    with st.spinner("Computing DFT..."):
        N = len(audio)
        n = np.arange(N)
        k = n.reshape((N, 1))
        W = np.exp(-2j * np.pi * k * n / N)
        X = np.dot(W, audio)

    magnitude = np.abs(X)
    freq = np.arange(N) * sr / N

    # Spectrum
    st.markdown("### 📊 Frequency Spectrum")
    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:N//2], magnitude[:N//2])
    st.pyplot(fig2)

    st.success("Processing Complete")

else:
    st.warning("Upload or record audio to begin.")