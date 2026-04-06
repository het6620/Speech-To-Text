import streamlit as st
import os
import soundfile as sf

from preprocessing.audio_record import record_audio
from preprocessing.audio_cleaning import preprocess_audio
from features.extract_features import extract_mfcc
from model.asr_model import ASRModel
from decoder.decoder import decode_predictions

DATA_FOLDER = "data"
RECORDED_AUDIO = os.path.join(DATA_FOLDER, "recorded_audio.wav")
UPLOADED_AUDIO = os.path.join(DATA_FOLDER, "uploaded_audio.wav")

st.set_page_config(page_title="Speech to Text", layout="centered")

st.title("🎧 Speech to Text Conversion System")
st.write("Convert speech into text using Deep Learning")

# ------------------ CORE TRANSCRIPTION ------------------

def transcribe(audio_path):
    with st.spinner("Processing audio..."):
        audio, sr = preprocess_audio(audio_path)
        extract_mfcc(audio, sr)  # academic completeness

        asr = ASRModel()
        predicted_ids = asr.predict(audio, sr)
        text = decode_predictions(predicted_ids, asr.processor)

    return text

# ------------------ GUI OPTIONS ------------------

st.subheader("Choose Input Method")

tab1, tab2 = st.tabs(["🎙 Record Audio", "📂 Upload Audio"])

# -------- TAB 1: RECORD AUDIO --------
with tab1:
    st.write("Record audio using microphone")

    duration = st.slider(
        "Select recording duration (seconds)",
        min_value=1,
        max_value=30,
        value=5
    )

    if st.button("Start Recording"):
        record_audio(RECORDED_AUDIO, duration=duration)
        st.audio(RECORDED_AUDIO)

        result = transcribe(RECORDED_AUDIO)
        st.success("Transcription Completed")
        st.text_area("Recognized Text", result, height=150)

# -------- TAB 2: UPLOAD AUDIO --------
with tab2:
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV format recommended)",
        type=["wav"]
    )

    if uploaded_file is not None:
        # Save uploaded file
        with open(UPLOADED_AUDIO, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(UPLOADED_AUDIO)

        if st.button("Convert Uploaded Audio"):
            result = transcribe(UPLOADED_AUDIO)
            st.success("Transcription Completed")
            st.text_area("Recognized Text", result, height=150)

st.markdown("---")
st.caption("Speech Recognition using Deep Learning (Wav2Vec2)")
