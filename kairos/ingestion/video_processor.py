"""
Video Processor
Handles video loading and frame extraction
"""
import os
import numpy as np
from typing import Tuple, Optional, List

from config import VIDEO_FPS, VIDEO_FRAME_SIZE


class VideoProcessor:
    """
    Video processing utilities for the Kairos system. 
    Handles loading and frame extraction from video files.
    """
    
    def __init__(self, target_fps: int = VIDEO_FPS, target_size:  Tuple[int, int] = VIDEO_FRAME_SIZE):
        self.target_fps = target_fps
        self.target_size = target_size
    
    def load_file(self, filepath: str) -> Tuple[np.ndarray, float]:
        """
        Load video file and extract frames.
        
        Args:
            filepath: Path to video file
            
        Returns:
            Tuple of (frame_array, fps)
        """
        print(f"  Loading video from: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"  ❌ Video file not found")
            return self._create_empty_frames(1), self.target_fps
        
        try:
            # Try moviepy first
            frames = self._load_with_moviepy(filepath)
            if frames is not None and len(frames) > 0:
                return frames, self.target_fps
        except Exception as e:
            print(f"  ⚠️ MoviePy failed: {e}")
        
        try:
            # Try OpenCV as fallback
            frames = self._load_with_opencv(filepath)
            if frames is not None and len(frames) > 0:
                return frames, self.target_fps
        except Exception as e:
            print(f"  ⚠️ OpenCV failed: {e}")
        
        print("  ❌ All video loading methods failed")
        return self._create_empty_frames(1), self.target_fps
    
    def _load_with_moviepy(self, filepath: str) -> Optional[np.ndarray]:
        """Load video using MoviePy."""
        from moviepy.editor import VideoFileClip
        import cv2
        
        video = VideoFileClip(filepath)
        
        duration = video.duration
        original_fps = video.fps
        
        print(f"  Video info: {duration:.2f}s, {original_fps}fps, size={video.size}")
        
        # Calculate number of frames at target FPS
        num_frames = max(1, int(duration * self.target_fps))
        
        # Sample frames evenly
        frame_times = np.linspace(0, max(0, duration - 0.1), num_frames)
        
        frames = []
        last_valid_frame = None
        
        for i, t in enumerate(frame_times):
            try:
                frame = video.get_frame(t)
                
                # Ensure correct format
                if frame is not None:
                    # Resize
                    if frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                        frame = cv2.resize(frame, self.target_size)
                    
                    # Ensure uint8
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    # Ensure 3 channels (RGB)
                    if len(frame.shape) == 2:
                        frame = np. stack([frame] * 3, axis=-1)
                    elif frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    
                    last_valid_frame = frame. copy()
                    frames.append(frame)
                else: 
                    if last_valid_frame is not None:
                        frames.append(last_valid_frame. copy())
                    else: 
                        frames.append(np. zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8))
                        
            except Exception as e: 
                print(f"  Warning: Frame {i} at t={t:.2f}s failed: {e}")
                if last_valid_frame is not None: 
                    frames.append(last_valid_frame.copy())
                else:
                    frames. append(np.zeros((self. target_size[1], self. target_size[0], 3), dtype=np.uint8))
        
        video.close()
        
        if frames: 
            return np.array(frames, dtype=np.uint8)
        return None
    
    def _load_with_opencv(self, filepath: str) -> Optional[np.ndarray]:
        """Load video using OpenCV."""
        import cv2
        
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            print("  ❌ OpenCV could not open video")
            return None
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  OpenCV:  {total_frames} frames at {original_fps}fps")
        
        # Calculate frame sampling
        if original_fps > 0:
            frame_interval = max(1, int(original_fps / self.target_fps))
        else:
            frame_interval = 1
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame = cv2.resize(frame, self.target_size)
                
                frames.append(frame. astype(np.uint8))
            
            frame_idx += 1
        
        cap.release()
        
        if frames: 
            return np.array(frames, dtype=np.uint8)
        return None
    
    def _create_empty_frames(self, num_frames: int) -> np.ndarray:
        """Create empty frame array."""
        return np.zeros(
            (num_frames, self.target_size[1], self.target_size[0], 3),
            dtype=np. uint8
        )