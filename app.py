from preprocessing.audio_cleaning import preprocess_audio
from preprocessing.audio_record import record_audio
from features.extract_features import extract_mfcc
from model.asr_model import ASRModel
from decoder.decoder import decode_predictions
import os

DATA_FOLDER = "data"
RECORDED_AUDIO = os.path.join(DATA_FOLDER, "recorded_audio.wav")
SAMPLE_AUDIO = os.path.join(DATA_FOLDER, "sample_audio.wav")

def transcribe(audio_path):
    print("\n🔊 Preprocessing audio...")
    audio, sr = preprocess_audio(audio_path)

    print("📊 Extracting features (MFCC)...")
    extract_mfcc(audio, sr)  # For academic completeness

    print("🧠 Loading ASR model...")
    asr = ASRModel()

    print("📝 Converting speech to text...")
    predicted_ids = asr.predict(audio, sr)
    text = decode_predictions(predicted_ids, asr.processor)

    print("\n✅ Transcription Result:")
    print(text)
    print("-" * 50)

def main():
    print("\n🎧 SPEECH TO TEXT CONVERSION SYSTEM 🎧\n")

    choice = input("Do you want to record a new audio message? (y/n): ").strip().lower()

    if choice == "y":
        record_audio(RECORDED_AUDIO, duration=5)
        transcribe(RECORDED_AUDIO)

    else:
        choice2 = input("Do you want to convert an existing audio file from data folder? (y/n): ").strip().lower()

        if choice2 == "y":
            if not os.path.exists(SAMPLE_AUDIO):
                print("❌ sample_audio.wav not found in data folder.")
                return
            transcribe(SAMPLE_AUDIO)

        else:
            print("\n👋 Goodbye! Thank you for using the Speech-to-Text system.")

if __name__ == "__main__":
    main()
