"""
File Loader
Handles loading and preprocessing of video, audio, and text inputs
"""
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple

from config import AUDIO_SAMPLE_RATE, VIDEO_FPS, VIDEO_FRAME_SIZE


class FileLoader:
    """Omni-Channel Ingestion (Static Batch Architecture)"""
    
    def __init__(self):
        self.audio_processor = None
        self.video_processor = None
        self._init_processors()
    
    def _init_processors(self):
        """Lazy initialize processors"""
        from . audio_processor import AudioProcessor
        from .video_processor import VideoProcessor
        self.audio_processor = AudioProcessor(target_sr=AUDIO_SAMPLE_RATE)
        self.video_processor = VideoProcessor(target_fps=VIDEO_FPS, target_size=VIDEO_FRAME_SIZE)
    
    def load(self, input_object: Dict[str, Any]) -> Dict[str, Any]:
        """Load input based on modality."""
        modality = input_object.get("modality", "text")
        filepath = input_object.get("filepath")
        text = input_object.get("text", "")
        
        result = {
            "audio_array": None,
            "video_array": None,
            "modality_type": modality,
            "video_missing": True,
            "audio_missing":  True,
            "text": text
        }
        
        if modality == "video":
            result = self._load_video(filepath, result)
        elif modality == "audio":
            result = self._load_audio(filepath, result)
        elif modality == "text":
            result = self._load_text(text, result)
        
        return result
    
    def _load_video(self, filepath: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Load video file, extract audio and frames."""
        print(f"  Loading video file: {filepath}")
        
        if not filepath or not os.path.exists(filepath):
            print(f"  Video file not found: {filepath}")
            return self._create_empty_arrays(result)
        
        try:
            # Extract audio from video
            print("  Extracting audio from video...")
            audio_array, audio_sr = self._extract_audio_from_video(filepath)
            
            if audio_array is not None and len(audio_array) > 0:
                if audio_sr != AUDIO_SAMPLE_RATE:
                    audio_array = self. audio_processor.resample(audio_array, audio_sr, AUDIO_SAMPLE_RATE)
                result["audio_array"] = audio_array. astype(np.float32)
                result["audio_missing"] = False
                print(f"  ✅ Audio extracted:  {len(audio_array)} samples")
            else:
                print("  ⚠️ No audio in video")
                result["audio_array"] = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
            
            # Extract video frames
            print("  Extracting video frames...")
            video_array = self._extract_video_frames(filepath)
            
            if video_array is not None and len(video_array) > 0:
                result["video_array"] = video_array
                result["video_missing"] = False
                print(f"  ✅ Video extracted: {len(video_array)} frames")
            else:
                print("  ⚠️ Could not extract video frames")
                result["video_array"] = self._create_empty_video_array(1)
            
            result["modality_type"] = "video"
            
        except Exception as e:
            print(f"  ❌ Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_arrays(result)
        
        return result
    
    def _extract_audio_from_video(self, filepath: str) -> Tuple[Optional[np.ndarray], int]: 
        """Extract audio track from video file."""
        # Try FFmpeg first (more reliable)
        try:
            audio, sr = self._extract_audio_ffmpeg(filepath)
            if audio is not None and len(audio) > 0:
                return audio, sr
        except Exception as e:
            print(f"  FFmpeg extraction failed: {e}")
        
        # Fallback to MoviePy
        try:
            from moviepy.editor import VideoFileClip
            
            video = VideoFileClip(filepath)
            
            if video.audio is None:
                print("  Video has no audio track")
                video.close()
                return None, AUDIO_SAMPLE_RATE
            
            audio_fps = int(video.audio.fps)
            duration = video.audio.duration
            
            # Use integer fps to avoid format specifier issues
            print(f"  Audio:  {duration:. 2f}s at {audio_fps}Hz")
            
            # Extract audio
            audio_array = video.audio.to_soundarray(fps=audio_fps)
            
            video.close()
            
            # Convert to mono
            if len(audio_array. shape) > 1:
                if audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                else:
                    audio_array = audio_array.flatten()
            
            audio_array = audio_array.astype(np.float32)
            
            # Normalize
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            return audio_array, audio_fps
            
        except Exception as e:
            print(f"  ❌ MoviePy audio extraction failed: {e}")
            return None, AUDIO_SAMPLE_RATE
    
    def _extract_audio_ffmpeg(self, filepath:  str) -> Tuple[Optional[np.ndarray], int]: 
        """Extract audio using ffmpeg directly."""
        import subprocess
        import tempfile
        import librosa
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cmd = [
                'ffmpeg', '-y', '-i', filepath,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(AUDIO_SAMPLE_RATE),
                '-ac', '1',
                '-loglevel', 'error',
                tmp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result. returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                audio, sr = librosa.load(tmp_path, sr=AUDIO_SAMPLE_RATE, mono=True)
                print(f"  ✅ FFmpeg extracted audio: {len(audio)} samples")
                return audio. astype(np.float32), sr
            else:
                if result.stderr:
                    print(f"  FFmpeg stderr: {result.stderr[: 200]}")
                return None, AUDIO_SAMPLE_RATE
                
        except subprocess.TimeoutExpired:
            print("  FFmpeg timed out")
            return None, AUDIO_SAMPLE_RATE
        except Exception as e: 
            print(f"  FFmpeg error: {e}")
            return None, AUDIO_SAMPLE_RATE
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    
    def _extract_video_frames(self, filepath: str) -> Optional[np.ndarray]:
        """Extract video frames at target FPS."""
        try:
            from moviepy.editor import VideoFileClip
            import cv2
            
            video = VideoFileClip(filepath)
            
            duration = video.duration
            original_fps = video.fps
            
            # Use simple formatting without potential issues
            dur_str = f"{duration:.2f}"
            fps_str = f"{original_fps:.1f}"
            print(f"  Video: {dur_str}s at {fps_str}fps")
            
            num_frames = int(duration * VIDEO_FPS)
            if num_frames < 1:
                num_frames = 1
            
            frame_times = np.linspace(0, max(0, duration - 0.1), num_frames)
            
            frames = []
            last_valid_frame = None
            
            for t in frame_times:
                try:
                    frame = video. get_frame(t)
                    
                    if frame is not None: 
                        if frame.shape[: 2] != (VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0]):
                            frame = cv2.resize(frame, VIDEO_FRAME_SIZE)
                        
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = np.clip(frame, 0, 255).astype(np.uint8)
                        
                        if len(frame.shape) == 2:
                            frame = np. stack([frame] * 3, axis=-1)
                        elif frame.shape[2] == 4:
                            frame = frame[:, :, :3]
                        
                        last_valid_frame = frame. copy()
                        frames.append(frame)
                    elif last_valid_frame is not None: 
                        frames.append(last_valid_frame. copy())
                    else:
                        frames.append(np.zeros((VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8))
                        
                except Exception as e:
                    if last_valid_frame is not None: 
                        frames.append(last_valid_frame.copy())
                    else:
                        frames.append(np.zeros((VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8))
            
            video.close()
            
            if frames:
                return np.array(frames, dtype=np.uint8)
            return None
            
        except Exception as e:
            print(f"  ❌ Video frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_audio(self, filepath: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Load audio file."""
        print(f"  Loading audio file...")
        
        if not filepath or not os.path.exists(filepath):
            print(f"  Audio file not found: {filepath}")
            return self._create_empty_arrays(result)
        
        try:
            audio_array, sr = self. audio_processor.load_file(filepath)
            
            if audio_array is not None and len(audio_array) > 0:
                if sr != AUDIO_SAMPLE_RATE: 
                    audio_array = self. audio_processor.resample(audio_array, sr, AUDIO_SAMPLE_RATE)
                
                result["audio_array"] = audio_array.astype(np.float32)
                result["audio_missing"] = False
                
                dur = len(audio_array) / AUDIO_SAMPLE_RATE
                print(f"  ✅ Audio loaded: {len(audio_array)} samples ({dur:.2f}s)")
            else:
                print("  ⚠️ Audio file is empty")
                result["audio_array"] = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
            
            num_frames = max(1, int(len(result["audio_array"]) / AUDIO_SAMPLE_RATE * VIDEO_FPS))
            result["video_array"] = self._create_empty_video_array(num_frames)
            result["video_missing"] = True
            result["modality_type"] = "audio"
            
        except Exception as e:
            print(f"  ❌ Error loading audio: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_arrays(result)
        
        return result
    
    def _load_text(self, text: str, result:  Dict[str, Any]) -> Dict[str, Any]:
        """Handle text-only input."""
        print(f"  Processing text input...")
        
        result["audio_array"] = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        result["video_array"] = self._create_empty_video_array(1)
        result["audio_missing"] = True
        result["video_missing"] = True
        result["modality_type"] = "text"
        result["text"] = text
        
        return result
    
    def _create_empty_arrays(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty arrays when loading fails."""
        result["audio_array"] = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        result["video_array"] = self._create_empty_video_array(1)
        result["audio_missing"] = True
        result["video_missing"] = True
        return result
    
    def _create_empty_video_array(self, num_frames: int) -> np.ndarray:
        """Create zero-filled video array."""
        return np.zeros(
            (num_frames, VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3),
            dtype=np.uint8
        )