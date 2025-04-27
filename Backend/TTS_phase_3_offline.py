import torch
from TTS.api import TTS

def main_TTS(podcast_text, final_speech_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "tts_models/en/ljspeech/fast_pitch"

    tts = TTS(model_name).to(device)
    tts.tts_to_file(text = podcast_text, file_path = final_speech_file)

#to check list of models, tts --list_models in cmd with conda environment diarisation activated