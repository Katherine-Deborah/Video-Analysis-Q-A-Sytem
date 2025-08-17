import time

def process_video(video_path, session_id, models, conn):
    """Original process_video function - maintains compatibility"""
    
    try:
        # Import your modules
        from captions import extract_frames, generate_caption
        from audio import extract_audio, transcribe_audio
        
        # Extract frames with default interval
        print("Extracting frames...")
        frames, timestamps = extract_frames(video_path, interval=0.5)
        
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
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

def process_video_with_fps(video_path, session_id, models, conn, fps=5):
    """Enhanced process_video function with FPS control"""
    
    try:
        # Import your modules
        from captions import extract_frames_with_fps, generate_caption, batch_generate_captions
        from audio import extract_audio, transcribe_audio
        
        # Calculate interval from FPS
        interval = 1.0 / fps
        
        # Extract frames with custom FPS
        print(f"Extracting frames at {fps} FPS (interval: {interval:.2f}s)...")
        frames, timestamps = extract_frames_with_fps(video_path, interval=interval)
        
        if not frames:
            print("No frames could be extracted from the video.")
            return
        
        # Generate captions (use batch processing for efficiency)
        print(f"Generating captions for {len(frames)} frames...")
        cursor = conn.cursor()
        
        # Option 1: Batch processing (more efficient)
        try:
            captions = batch_generate_captions(frames, models, batch_size=4)
            
            # Insert all captions
            for i, (timestamp, caption) in enumerate(zip(timestamps, captions)):
                cursor.execute(
                    "INSERT INTO captions (session_id, timestamp, caption) VALUES (?, ?, ?)",
                    (session_id, timestamp, caption)
                )
                
                if i % 10 == 0:
                    print(f"Inserting captions... {i+1}/{len(captions)}")
        
        except:
            # Option 2: Fallback to individual processing
            print("Batch processing failed, using individual processing...")
            for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                caption = generate_caption(frame, models)
                cursor.execute(
                    "INSERT INTO captions (session_id, timestamp, caption) VALUES (?, ?, ?)",
                    (session_id, timestamp, caption)
                )
                
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
        
    except Exception as e:
        print(f"Error processing video with FPS: {str(e)}")
        # Fallback to original function
        print("Falling back to original processing...")
        process_video(video_path, session_id, models, conn)