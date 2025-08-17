import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def extract_frames(video_path, interval=0.5):
    """Original function - extract frames at fixed interval"""
    return extract_frames_with_fps(video_path, interval=interval)

def extract_frames_with_fps(video_path, interval=0.5):
    """Extract frames from video at specified interval (supports FPS control)
    
    Args:
        video_path: Path to video file
        interval: Time interval between frames in seconds (1/fps)
    
    Returns:
        frames: List of PIL Images
        timestamps: List of timestamp values
    """
    frames = []
    timestamps = []
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames, timestamps
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video info: {fps:.2f} FPS, {duration:.2f}s duration, {total_frames} total frames")
        print(f"Extracting frames every {interval:.2f} seconds")
        
        frame_interval = int(fps * interval)  # Convert time interval to frame interval
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Calculate timestamp
                timestamp = frame_count / fps
                
                frames.append(pil_image)
                timestamps.append(timestamp)
                
                if len(frames) % 10 == 0:
                    print(f"Extracted {len(frames)} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Extraction complete: {len(frames)} frames extracted")
        
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
    
    return frames, timestamps

def generate_caption(image, models):
    """Generate caption for a single image using your custom model - FIXED VERSION"""
    try:
        # FIXED: Use the correct processor call with 'images=' parameter like your working original
        inputs = models['caption_processor'](images=image, return_tensors="pt").to(models['device'])
        
        with torch.no_grad():
            # FIXED: Use generate with max_new_tokens like your working original
            output_ids = models['caption_model'].generate(**inputs, max_new_tokens=50)
            caption = models['caption_processor'].batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return f"Error generating caption: {e}"

def batch_generate_captions(frames, models, batch_size=4):
    """Generate captions for multiple frames in batches (more efficient)"""
    captions = []
    
    try:
        processor = models['caption_processor']
        model = models['caption_model']
        device = models['device']
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # FIXED: Use the correct processor call with 'images=' parameter
            inputs = processor(images=batch_frames, return_tensors="pt").to(device)
            
            # Generate captions
            with torch.no_grad():
                # FIXED: Use max_new_tokens instead of max_length for your model
                outputs = model.generate(**inputs, max_new_tokens=50)
            
            # Decode captions - FIXED: Use batch_decode like your original
            batch_captions = processor.batch_decode(outputs, skip_special_tokens=True)
            
            captions.extend(batch_captions)
            print(f"Generated captions for batch {i//batch_size + 1}/{(len(frames)-1)//batch_size + 1}")
    
    except Exception as e:
        print(f"Error in batch caption generation: {str(e)}")
        # Fallback to individual processing using the working method
        for frame in frames:
            captions.append(generate_caption(frame, models))
    
    return captions