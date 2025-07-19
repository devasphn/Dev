import torch
import torchaudio
from transformers import CsmForConditionalGeneration, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CSM for TTS
csm_processor = AutoProcessor.from_pretrained("sesame/csm-1b")
csm_model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", device_map=device)

# Load Whisper for ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)

# Load Llama for conversation
llm = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", device=device)

def generate_speech(text, context=[]):
    inputs = csm_processor.apply_chat_template([{"role": "0", "content": [{"type": "text", "text": text}]}], tokenize=True, return_dict=True).to(device)
    audio = csm_model.generate(**inputs, output_audio=True)
    return audio

def transcribe_audio(audio_path):
    return asr(audio_path)["text"]

def generate_response(prompt):
    return llm(prompt, max_new_tokens=100)[0]["generated_text"]
