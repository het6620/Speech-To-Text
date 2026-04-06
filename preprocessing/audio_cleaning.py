import librosa
import numpy as np

def preprocess_audio(audio_path, target_sr=16000):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    return audio, target_sr
