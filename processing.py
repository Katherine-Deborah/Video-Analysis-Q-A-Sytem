import streamlit as st
import time
from captions import extract_frames, generate_caption
from audio import extract_audio, transcribe_audio

def process_video(video_path, session_id, models, conn):
    """Process video: extract frames, generate captions, transcribe audio"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Extract frames
        status_text.text("Extracting frames...")
        frames, timestamps = extract_frames(video_path, interval=0.5)
        progress_bar.progress(0.2)  # 20%
        
        if not frames:
            st.error("No frames could be extracted from the video.")
            return
        
        # Generate captions
        status_text.text(f"Generating captions for {len(frames)} frames...")
        cursor = conn.cursor()
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            caption = generate_caption(frame, models)
            cursor.execute(
                "INSERT INTO captions (session_id, timestamp, caption) VALUES (?, ?, ?)",
                (session_id, timestamp, caption)
            )
            # Progress from 20% to 80% (60% range for captions)
            progress_value = 0.2 + (i / len(frames)) * 0.6
            progress_bar.progress(min(progress_value, 0.8))  # Ensure max 80%
            
            # Update status every 10 frames
            if i % 10 == 0:
                status_text.text(f"Generating captions... {i+1}/{len(frames)}")
        
        conn.commit()
        progress_bar.progress(0.8)  # 80%
        
        # Extract and transcribe audio
        status_text.text("Extracting and transcribing audio...")
        audio, sr = extract_audio(video_path)
        progress_bar.progress(0.9)  # 90%
        
        if audio is not None and len(audio) > 0:
            transcription = transcribe_audio(audio, sr, models)
            cursor.execute(
                "INSERT INTO transcriptions (session_id, transcription) VALUES (?, ?)",
                (session_id, transcription)
            )
            conn.commit()
        else:
            st.warning("No audio found in the video or audio extraction failed.")
        
        progress_bar.progress(1.0)  # 100%
        status_text.text("Processing complete!")
        time.sleep(1)
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()