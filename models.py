import sqlite3
import torch
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('video_analysis.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp REAL,
            caption TEXT,
            FOREIGN KEY (session_id) REFERENCES video_sessions (session_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            transcription TEXT,
            FOREIGN KEY (session_id) REFERENCES video_sessions (session_id)
        )
    ''')
    
    conn.commit()
    return conn

def load_models():
    """Load all AI models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load captioning model
        print("Loading captioning model...")
        caption_model_id = "quadranttechnologies/qhub-blip-image-captioning-finetuned"
        caption_processor = AutoProcessor.from_pretrained(caption_model_id)
        caption_model = AutoModelForVision2Seq.from_pretrained(caption_model_id).to(device)
        
        # Load transcription model
        print("Loading transcription model...")
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
        whisper_model.config.forced_decoder_ids = None
        
        # Load QA model
        print("Loading QA model...")
        qa_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        qa_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",torch_dtype="auto",device_map="auto")
        
        return {
            'caption_processor': caption_processor,
            'caption_model': caption_model,
            'whisper_processor': whisper_processor,
            'whisper_model': whisper_model,
            'qa_tokenizer': qa_tokenizer,
            'qa_model': qa_model,
            'device': device
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None