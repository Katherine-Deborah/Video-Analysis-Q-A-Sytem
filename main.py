import streamlit as st
from models import init_database, load_models
from QA import get_context_for_qa, answer_question
from processing import process_video
import os
import tempfile
import time
import sqlite3
import cv2
import numpy as np
from datetime import datetime
import threading
import queue
import io
from PIL import Image

# Set page config
st.set_page_config(page_title="Video Analysis QA System", layout="wide")

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []
if 'recording' not in st.session_state:
    st.session_state.recording = False

def capture_webcam_frames(duration_seconds=10, fps=5):
    """Capture frames from webcam for specified duration"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return []
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frames = []
    start_time = time.time()
    frame_interval = 1.0 / fps
    last_capture_time = 0
    
    try:
        while (time.time() - start_time) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time() - start_time
            
            # Capture frame at specified intervals
            if current_time - last_capture_time >= frame_interval:
                # Convert BGR to RGB for PIL/Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'frame': frame_rgb,
                    'timestamp': current_time,
                    'frame_bgr': frame  # Keep BGR version for video writing
                })
                last_capture_time = current_time
    
    finally:
        cap.release()
    
    return frames

def save_frames_as_video(frames, output_path, fps=5):
    """Save captured frames as MP4 video"""
    if not frames:
        return False
    
    height, width, _ = frames[0]['frame_bgr'].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame_data in frames:
            out.write(frame_data['frame_bgr'])
        return True
    except Exception as e:
        st.error(f"Error saving video: {e}")
        return False
    finally:
        out.release()

def webcam_interface():
    """Enhanced webcam interface with actual capture functionality"""
    st.subheader("📸 Webcam Capture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera preview placeholder
        camera_placeholder = st.empty()
        
        # Check if webcam is available
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, caption="Webcam Preview", use_container_width=True)
                else:
                    camera_placeholder.error("Unable to read from webcam")
                cap.release()
            else:
                camera_placeholder.error("Webcam not accessible")
        except Exception as e:
            camera_placeholder.error(f"Webcam error: {e}")
    
    with col2:
        st.markdown("### Recording Settings")
        
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=3, max_value=30, value=10, step=1)
        
        fps = st.selectbox("Frames per second", [1, 2, 5, 10], index=2)
        
        st.info(f"Will capture ~{duration * fps} frames")
        
        # Recording controls
        if st.button("🔴 Start Recording", type="primary", disabled=st.session_state.recording):
            st.session_state.recording = True
            
            # Show recording progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            preview_placeholder = st.empty()
            
            with st.spinner("Recording from webcam..."):
                # Capture frames in real-time with progress updates
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam")
                    st.session_state.recording = False
                    return
                
                frames = []
                start_time = time.time()
                frame_interval = 1.0 / fps
                last_capture_time = 0
                
                try:
                    while (time.time() - start_time) < duration:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        current_time = time.time() - start_time
                        progress = min(current_time / duration, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Recording... {current_time:.1f}s / {duration}s")
                        
                        # Show live preview
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(frame_rgb, caption="Recording...", width=300)
                        
                        # Capture frame at specified intervals
                        if current_time - last_capture_time >= frame_interval:
                            frames.append({
                                'frame': frame_rgb,
                                'timestamp': current_time,
                                'frame_bgr': frame
                            })
                            last_capture_time = current_time
                
                finally:
                    cap.release()
                    progress_bar.empty()
                    status_text.empty()
                    preview_placeholder.empty()
                
                st.session_state.recorded_frames = frames
                st.session_state.recording = False
                
                if frames:
                    st.success(f"✅ Recorded {len(frames)} frames successfully!")
                    
                    # Show sample of recorded frames
                    st.markdown("### Recorded Frames Preview")
                    
                    # Display first few frames as preview
                    preview_cols = st.columns(min(4, len(frames)))
                    for i, frame_data in enumerate(frames[:4]):
                        with preview_cols[i]:
                            st.image(frame_data['frame'], 
                                   caption=f"Frame at {frame_data['timestamp']:.1f}s", 
                                   width=150)
                else:
                    st.error("No frames were captured. Please check your webcam.")
        
        # Process recorded video
        if st.session_state.recorded_frames and st.button("🚀 Process Recorded Video", type="secondary"):
            frames = st.session_state.recorded_frames
            
            # Create temporary video file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                video_path = tmp_file.name
            
            if save_frames_as_video(frames, video_path, fps):
                st.session_state.processing = True
                
                with st.spinner("Processing recorded video..."):
                    try:
                        session_id = st.session_state.current_session_id
                        process_video(video_path, session_id, st.session_state.models, st.session_state.conn)
                        st.success("✅ Video processed successfully!")
                        
                        # Clear recorded frames after processing
                        st.session_state.recorded_frames = []
                        
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    finally:
                        st.session_state.processing = False
                        # Clean up temp file
                        try:
                            os.unlink(video_path)
                        except:
                            pass
            else:
                st.error("Failed to save recorded frames as video")
        
        # Clear recording
        if st.session_state.recorded_frames and st.button("🗑️ Clear Recording"):
            st.session_state.recorded_frames = []
            st.success("Recording cleared")

def clear_database():
    """Clear all data from previous sessions"""
    try:
        conn = sqlite3.connect('video_analysis.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM captions")
        cursor.execute("DELETE FROM transcriptions") 
        cursor.execute("DELETE FROM video_sessions")
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error clearing database: {e}")

def main():
    st.title("🎥 Video Analysis QA System")
    st.markdown("Upload a video or use webcam to analyze content and ask questions!")
    
    if 'db_cleared' not in st.session_state:
        clear_database()
        st.session_state.db_cleared = True

    # Initialize database
    if not st.session_state.db_initialized:
        with st.spinner("Initializing database..."):
            st.session_state.conn = init_database()
            st.session_state.db_initialized = True
    
    session_id = "current_session"
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a few minutes."):
            models = load_models()
            if models is not None:
                st.session_state.models = models
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please check your internet connection and try again.")
                st.stop()
    
    # Sidebar for session management
    st.sidebar.header("Session Management")

    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = f"session_{int(time.time())}"
    
     # Show existing sessions in sidebar
    conn = sqlite3.connect('video_analysis.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, created_at FROM video_sessions ORDER BY created_at DESC LIMIT 10")
    existing_sessions = cursor.fetchall()
    conn.close()
    
    if existing_sessions:
        st.sidebar.subheader("Recent Sessions")
        for sess_id, created_at in existing_sessions:
            if st.sidebar.button(f"📁 {sess_id}", key=f"load_{sess_id}"):
                st.session_state.current_session_id = sess_id
                st.rerun()
    
    with st.sidebar.form("session_form"):
        manual_session = st.text_input("Edit Session ID", value=st.session_state.current_session_id)
        if st.form_submit_button("Update Session"):
            if manual_session != st.session_state.current_session_id:
                st.session_state.current_session_id = manual_session
                st.rerun()
    
    # Use the session_id from session state
    session_id = st.session_state.current_session_id
    
    # Create session in database
    conn = sqlite3.connect('video_analysis.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO video_sessions (session_id) VALUES (?)",
        (session_id,)
    )
    conn.commit()
    conn.close()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📹 Video Input", "🔍 Analysis Results", "❓ Ask Questions"])
    
    with tab1:
        st.header("Video Input")
        
        input_method = st.radio("Choose input method:", ["Upload Video", "Use Webcam"])
        
        if input_method == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                st.video(uploaded_file)
                
                if st.button("🚀 Process Video", type="primary"):
                    st.session_state.processing = True
                    process_video(video_path, session_id, st.session_state.models, st.session_state.conn)
                    st.session_state.processing = False
                    st.success("Video processed successfully!")
                    
                    # Clean up temp file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
        
        else:  # Webcam
            webcam_interface()
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if st.button("🔄 Refresh Results"):
            cursor = st.session_state.conn.cursor()
            
            # Display captions
            st.subheader("Frame Captions")
            cursor.execute(
                "SELECT timestamp, caption FROM captions WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            captions = cursor.fetchall()
            
            if captions:
                for timestamp, caption in captions:
                    st.write(f"**{timestamp:.1f}s:** {caption}")
            else:
                st.info("No captions found. Please process a video first.")
            
            # Display transcription
            st.subheader("Audio Transcription")
            cursor.execute(
                "SELECT transcription FROM transcriptions WHERE session_id = ?",
                (session_id,)
            )
            transcription_result = cursor.fetchone()
            
            if transcription_result:
                st.write(transcription_result[0])
            else:
                st.info("No transcription found. Please process a video with audio.")
    
    with tab3:
        st.header("❓ Ask Questions")
        
        question = st.text_input("Ask a question about the video:", placeholder="What was moving in the video?")
        
        if st.button("🤔 Get Answer", type="primary") and question:
            with st.spinner("Generating answer..."):
                context = get_context_for_qa(session_id, st.session_state.conn)
                
                if context.strip() == "CAPTIONS:":
                    st.warning("No video data found. Please process a video first.")
                else:
                    answer = answer_question(question, context, st.session_state.models)
                    st.write("**Answer:**")
                    st.write(answer)
        
        # Display recent questions (could be enhanced to store in DB)
        st.subheader("💡 Example Questions")
        example_questions = [
            "What objects were visible in the video?",
            "What was the person doing?",
            "What did someone say about [topic]?",
            "What was moving in the scene?",
            "Describe what happened at the beginning/middle/end"
        ]
        
        for eq in example_questions:
            if st.button(eq, key=f"example_{eq}"):
                st.session_state.example_question = eq
                st.rerun()
        
        if hasattr(st.session_state, 'example_question'):
            st.text_input("Ask a question about the video:", value=st.session_state.example_question, key="example_input")

if __name__ == "__main__":
    main()