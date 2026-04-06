def decode_predictions(predicted_ids, processor):
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]
