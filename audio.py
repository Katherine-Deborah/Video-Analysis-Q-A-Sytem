import librosa
import torch
import streamlit as st

def extract_audio(video_path):
    """Extract audio from video file"""
    try:
        # Use librosa to extract audio
        audio, sr = librosa.load(video_path, sr=16000)  # Whisper expects 16kHz
        return audio, sr
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None, None

def transcribe_audio(audio, sr, models):
    """Transcribe audio using Whisper"""
    try:
        inputs = models['whisper_processor'](audio, sampling_rate=sr, return_tensors="pt").input_features.to(models['device'])
        
        with torch.no_grad():
            pred_ids = models['whisper_model'].generate(inputs)
            transcription = models['whisper_processor'].batch_decode(pred_ids, skip_special_tokens=True)[0]
        
        return transcription
    except Exception as e:
        return f"Error transcribing audio: {e}"