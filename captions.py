import cv2
import streamlit as st
import torch
from PIL import Image

def extract_frames(video_path, interval=0.5):
    """Extract frames from video at specified intervals"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Could not open video file.")
            return [], []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            st.error("Could not determine video frame rate.")
            cap.release()
            return [], []
            
        frame_interval = max(1, int(fps * interval))  # Ensure at least 1 frame interval
        
        frames = []
        timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(frame_count / fps)
                
            frame_count += 1
        
        cap.release()
        return frames, timestamps
    except Exception as e:
        st.error(f"Error extracting frames: {e}")
        return [], []
    
def generate_caption(image, models):
    """Generate caption for a single image"""
    try:
        inputs = models['caption_processor'](images=image, return_tensors="pt").to(models['device'])
        
        with torch.no_grad():
            output_ids = models['caption_model'].generate(**inputs, max_new_tokens=50)
            caption = models['caption_processor'].batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return caption
    except Exception as e:
        return f"Error generating caption: {e}"