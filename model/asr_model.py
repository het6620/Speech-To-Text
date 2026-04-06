import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ASRModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def predict(self, audio, sr):
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return predicted_ids
