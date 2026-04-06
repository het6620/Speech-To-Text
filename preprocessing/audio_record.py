import sounddevice as sd
import soundfile as sf

def record_audio(output_path, duration, sample_rate=16000):
    print(f"🎙 Recording for {duration} seconds...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    sf.write(output_path, audio, sample_rate)
    print(f"✅ Audio saved to {output_path}")
