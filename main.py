import gradio as gr
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

# Global variables
models = None
conn = None
current_session_id = "main_session"
current_fps_setting = 5

def clear_database_for_new_video(session_id, conn):
    """Clear database entries for a specific session (new video)"""
    try:
        cursor = conn.cursor()
        # Clear previous data for this session
        cursor.execute("DELETE FROM captions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM transcriptions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM video_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        print(f"Database cleared for session: {session_id}")
    except Exception as e:
        print(f"Error clearing database: {e}")

def process_video_with_fps(video_path, session_id, models, conn, fps):
    """Wrapper for process_video that handles FPS setting"""
    try:
        # Import your processing modules
        from captions import extract_frames_with_fps, generate_caption
        from audio import extract_audio, transcribe_audio
        
        # Extract frames with custom FPS
        print(f"Extracting frames at {fps} FPS...")
        interval = 1.0 / fps  # Convert FPS to interval
        frames, timestamps = extract_frames_with_fps(video_path, interval=interval)
        
        if not frames:
            print("No frames could be extracted from the video.")
            return
        
        # Generate captions
        print(f"Generating captions for {len(frames)} frames...")
        cursor = conn.cursor()
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            caption = generate_caption(frame, models)
            cursor.execute(
                "INSERT INTO captions (session_id, timestamp, caption) VALUES (?, ?, ?)",
                (session_id, timestamp, caption)
            )
            
            # Update status every 10 frames
            if i % 10 == 0:
                print(f"Generating captions... {i+1}/{len(frames)}")
        
        conn.commit()
        
        # Extract and transcribe audio
        print("Extracting and transcribing audio...")
        audio, sr = extract_audio(video_path)
        
        if audio is not None and len(audio) > 0:
            transcription = transcribe_audio(audio, sr, models)
            cursor.execute(
                "INSERT INTO transcriptions (session_id, transcription) VALUES (?, ?)",
                (session_id, transcription)
            )
            conn.commit()
        else:
            print("No audio found in the video or audio extraction failed.")
        
        print("Processing complete!")
        
    except ImportError:
        # Fallback to original process_video function if custom FPS functions don't exist
        print("Using original process_video function...")
        process_video(video_path, session_id, models, conn)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

def initialize_system():
    """Initialize database and load models - NO DATABASE CLEARING HERE"""
    global models, conn
    
    # Initialize database (but don't clear it here)
    conn = init_database()
    
    # Load models
    models = load_models()
    if models is None:
        raise Exception("Failed to load models. Please check your internet connection and try again.")
    
    return "✅ System initialized successfully!"

def process_uploaded_video(video_file, fps_setting, progress=gr.Progress()):
    """Process uploaded video file with FPS setting - CLEARS DB FIRST"""
    global models, conn, current_session_id
    
    if video_file is None:
        return "❌ Please upload a video file", "", ""
    
    if models is None or conn is None:
        return "❌ System not initialized. Please wait for initialization to complete.", "", ""
    
    progress(0.05, desc="Clearing previous data...")
    
    # CLEAR DATABASE FOR NEW VIDEO
    clear_database_for_new_video(current_session_id, conn)
    
    progress(0.1, desc="Processing video...")
    
    try:
        # Create fresh session in database
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO video_sessions (session_id) VALUES (?)",
            (current_session_id,)
        )
        conn.commit()
        
        progress(0.3, desc="Analyzing video content...")
        
        # Set global FPS setting for frame extraction
        global current_fps_setting
        current_fps_setting = fps_setting
        
        # Process the video (using global FPS setting)
        process_video_with_fps(video_file, current_session_id, models, conn, fps_setting)
        
        progress(0.8, desc="Retrieving results...")
        
        # Get results
        captions_text, transcription_text = get_analysis_results()
        
        progress(1.0, desc="Complete!")
        
        return "✅ Video processed successfully!", captions_text, transcription_text
    
    except Exception as e:
        return f"❌ Error processing video: {str(e)}", "", ""

def capture_webcam_video(duration, fps, progress=gr.Progress()):
    """Capture video from webcam"""
    global models, conn, current_session_id
    
    if models is None or conn is None:
        return "❌ System not initialized. Please wait for initialization to complete.", None, gr.Button(visible=False)
    
    progress(0.1, desc="Initializing webcam...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "❌ Could not open webcam. Please check your camera connection.", None, gr.Button(visible=False)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {actual_fps}, Requested: {fps}")
        
        # Create temporary video file with better naming
        timestamp = int(time.time())
        video_path = f"temp_webcam_{timestamp}.mp4"
        
        # Setup video writer with better codec settings
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
        
        if not out.isOpened():
            return "❌ Could not initialize video writer.", None, gr.Button(visible=False)
        
        start_time = time.time()
        frame_count = 0
        expected_frames = duration * fps
        
        progress(0.2, desc=f"Recording for {duration} seconds...")
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Resize frame to ensure consistent size
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
            frame_count += 1
            
            # Update progress
            elapsed = time.time() - start_time
            progress_val = 0.2 + (elapsed / duration) * 0.6
            progress(min(progress_val, 0.8), desc=f"Recording... {elapsed:.1f}s / {duration}s")
            
            # Control frame rate more precisely
            time.sleep(max(0, (1.0 / fps) - 0.01))
        
        cap.release()
        out.release()
        
        progress(0.9, desc="Finalizing video...")
        
        # Verify the video file was created and has content
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
            return "❌ Video file was not created properly.", None, gr.Button(visible=False)
        
        if frame_count == 0:
            try:
                os.unlink(video_path)
            except:
                pass
            return "❌ No frames were captured. Please check your webcam.", None, gr.Button(visible=False)
        
        progress(1.0, desc="Recording complete!")
        
        print(f"Video saved: {video_path}, Size: {os.path.getsize(video_path)} bytes, Frames: {frame_count}")
        
        return (
            f"✅ Webcam video recorded successfully! ({frame_count} frames, {frame_count/fps:.1f}s)", 
            video_path, 
            gr.Button("🚀 Process Recorded Video", visible=True, variant="secondary")
        )
    
    except Exception as e:
        print(f"Webcam capture error: {str(e)}")
        return f"❌ Error with webcam capture: {str(e)}", None, gr.Button(visible=False)

def process_webcam_video(video_path, fps_setting, progress=gr.Progress()):
    """Process the recorded webcam video - CLEARS DB FIRST"""
    global models, conn, current_session_id
    
    if not video_path:
        return "❌ No video to process", "", ""
    
    if models is None or conn is None:
        return "❌ System not initialized", "", ""
    
    progress(0.05, desc="Clearing previous data...")
    
    # CLEAR DATABASE FOR NEW VIDEO
    clear_database_for_new_video(current_session_id, conn)
    
    progress(0.1, desc="Processing recorded video...")
    
    try:
        # Create fresh session in database
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO video_sessions (session_id) VALUES (?)",
            (current_session_id,)
        )
        conn.commit()
        
        progress(0.3, desc="Analyzing video content...")
        
        # Set global FPS setting for frame extraction
        global current_fps_setting
        current_fps_setting = fps_setting
        
        # Process the recorded video with FPS setting
        process_video_with_fps(video_path, current_session_id, models, conn, fps_setting)
        
        progress(0.8, desc="Retrieving results...")
        
        # Get results
        captions_text, transcription_text = get_analysis_results()
        
        progress(1.0, desc="Complete!")
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass
        
        return "✅ Video processed successfully!", captions_text, transcription_text
    
    except Exception as e:
        return f"❌ Error processing video: {str(e)}", "", ""

def get_analysis_results():
    """Get analysis results for current session"""
    global conn, current_session_id
    
    if conn is None:
        return "System not initialized.", "System not initialized."
    
    cursor = conn.cursor()
    
    # Get captions
    cursor.execute(
        "SELECT timestamp, caption FROM captions WHERE session_id = ? ORDER BY timestamp",
        (current_session_id,)
    )
    captions = cursor.fetchall()
    
    if captions:
        captions_text = "\n".join([f"**{timestamp:.1f}s:** {caption}" for timestamp, caption in captions])
    else:
        captions_text = "No captions found. Please process a video first."
    
    # Get transcription
    cursor.execute(
        "SELECT transcription FROM transcriptions WHERE session_id = ?",
        (current_session_id,)
    )
    transcription_result = cursor.fetchone()
    
    if transcription_result:
        transcription_text = transcription_result[0]
    else:
        transcription_text = "No transcription found. Please process a video with audio."
    
    return captions_text, transcription_text

def refresh_results():
    """Refresh analysis results"""
    return get_analysis_results()

def answer_video_question(question):
    """Answer question about the video"""
    global models, conn, current_session_id
    
    if not question.strip():
        return "Please enter a question."
    
    if models is None or conn is None:
        return "System not initialized. Please wait for initialization to complete."
    
    try:
        context = get_context_for_qa(current_session_id, conn)
        
        if context.strip() == "CAPTIONS:":
            return "No video data found. Please process a video first."
        
        answer = answer_question(question, context, models)
        return f"**Answer:** {answer}"
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def set_example_question(question):
    """Set example question in the textbox"""
    return question

# Initialize system at startup (no database clearing here)
try:
    init_message = initialize_system()
    print(init_message)
except Exception as e:
    print(f"Initialization error: {e}")
    models = None
    conn = None

# Define example questions
example_questions = [
    "What objects were visible in the video?",
    "What was the person doing?", 
    "What did someone say about [topic]?",
    "What was moving in the scene?",
    "Describe what happened at the beginning/middle/end"
]

# Create Gradio interface
with gr.Blocks(title="Video Analysis QA System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Video Analysis QA System")
    gr.Markdown("Upload a video or use webcam to analyze content and ask questions!")
    
    # Store video path for webcam processing
    webcam_video_path = gr.State(value=None)
    
    # Main tabs
    with gr.Tabs():
        # Video Input Tab
        with gr.TabItem("📹 Video Input"):
            input_method = gr.Radio(
                choices=["Upload Video", "Use Webcam"],
                value="Upload Video",
                label="Choose input method"
            )
            
            # Upload Video Section
            with gr.Group(visible=True) as upload_section:
                gr.Markdown("### Upload Video")
                with gr.Row():
                    with gr.Column(scale=3):
                        video_upload = gr.File(
                            label="Choose a video file",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"]
                        )
                    with gr.Column(scale=1):
                        upload_fps = gr.Dropdown(
                            choices=[1, 2, 5, 10, 15, 30],
                            value=5,
                            label="Analysis FPS"
                        )
                
                video_preview = gr.Video(label="Video Preview")
                upload_btn = gr.Button("🚀 Process Video", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)
            
            # Webcam Section
            with gr.Group(visible=False) as webcam_section:
                gr.Markdown("### 📸 Webcam Capture")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        webcam_preview = gr.Image(
                            label="Webcam Preview",
                            sources=["webcam"],
                            streaming=True
                        )
                    
                    with gr.Column(scale=1):
                        duration_slider = gr.Slider(
                            minimum=3,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Recording Duration (seconds)"
                        )
                        
                        fps_dropdown = gr.Dropdown(
                            choices=[1, 2, 5, 10, 15],
                            value=5,
                            label="Recording FPS"
                        )
                        
                        webcam_analysis_fps = gr.Dropdown(
                            choices=[1, 2, 5, 10, 15, 30],
                            value=5,
                            label="Analysis FPS"
                        )
                        
                        webcam_info = gr.Markdown("Will capture approximately 50 frames")
                        webcam_btn = gr.Button("🔴 Start Recording", variant="primary")
                
                # Status and recorded video preview
                webcam_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        recorded_video_preview = gr.Video(label="Recorded Video", visible=True)
                    with gr.Column(scale=1):
                        process_webcam_btn = gr.Button("🚀 Process Recorded Video", visible=False, variant="secondary", size="lg")
        
        # Analysis Results Tab
        with gr.TabItem("🔍 Analysis Results"):
            refresh_btn = gr.Button("🔄 Refresh Results", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Frame Captions")
                    captions_output = gr.Textbox(
                        label="Captions",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### Audio Transcription")
                    transcription_output = gr.Textbox(
                        label="Transcription",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
        
        # Ask Questions Tab
        with gr.TabItem("❓ Ask Questions"):
            question_input = gr.Textbox(
                label="Ask a question about the video",
                placeholder="What was moving in the video?",
                lines=2
            )
            ask_btn = gr.Button("🤔 Get Answer", variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                lines=5,
                max_lines=10,
                interactive=False
            )
            
            gr.Markdown("### 💡 Example Questions")
            with gr.Row():
                for i, question in enumerate(example_questions):
                    example_btn = gr.Button(question, size="sm")
                    example_btn.click(
                        fn=set_example_question,
                        inputs=[gr.State(question)],
                        outputs=[question_input]
                    )
    
    # Event handlers
    def toggle_input_method(method):
        return (
            gr.Group(visible=(method == "Upload Video")),
            gr.Group(visible=(method == "Use Webcam"))
        )
    
    def update_webcam_info(duration, fps):
        estimated_frames = duration * fps
        return f"Will capture approximately {estimated_frames} frames"
    
    def preview_video(file):
        return file if file else None
    
    def handle_webcam_capture(duration, fps):
        """Handle webcam capture and return results"""
        status, video_path, _ = capture_webcam_video(duration, fps)
        
        if video_path:
            return (
                status,
                video_path,  # Store path in state
                video_path,  # Pass path directly to video component
                gr.Button("🚀 Process Recorded Video", visible=True, variant="secondary")
            )
        else:
            return (
                status,
                None,
                None,
                gr.Button("🚀 Process Recorded Video", visible=False, variant="secondary")
            )
    
    # Connect event handlers
    input_method.change(
        fn=toggle_input_method,
        inputs=[input_method],
        outputs=[upload_section, webcam_section]
    )
    
    duration_slider.change(
        fn=update_webcam_info,
        inputs=[duration_slider, fps_dropdown],
        outputs=[webcam_info]
    )
    
    fps_dropdown.change(
        fn=update_webcam_info,
        inputs=[duration_slider, fps_dropdown],
        outputs=[webcam_info]
    )
    
    video_upload.change(
        fn=preview_video,
        inputs=[video_upload],
        outputs=[video_preview]
    )
    
    upload_btn.click(
        fn=process_uploaded_video,
        inputs=[video_upload, upload_fps],
        outputs=[upload_status, captions_output, transcription_output]
    )
    
    webcam_btn.click(
        fn=handle_webcam_capture,
        inputs=[duration_slider, fps_dropdown],
        outputs=[webcam_status, webcam_video_path, recorded_video_preview, process_webcam_btn]
    )
    
    process_webcam_btn.click(
        fn=process_webcam_video,
        inputs=[webcam_video_path, webcam_analysis_fps],
        outputs=[webcam_status, captions_output, transcription_output]
    )
    
    refresh_btn.click(
        fn=refresh_results,
        outputs=[captions_output, transcription_output]
    )
    
    ask_btn.click(
        fn=answer_video_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Required for Hugging Face Spaces
        server_port=7860,       # Standard port for Hugging Face Spaces
        share=False,
        show_error=True
    )