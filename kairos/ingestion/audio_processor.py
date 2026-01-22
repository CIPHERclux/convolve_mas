"""
Audio Processor
Handles audio loading, extraction from video, and preprocessing
"""
import os
import numpy as np
from typing import Tuple, Optional

import librosa


class AudioProcessor:
    """
    Audio processing utilities for the Kairos system. 
    Handles loading, extraction, and preprocessing of audio data.
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_file(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa with fallback methods.
        
        Args:
            filepath: Path to audio file
            
        Returns: 
            Tuple of (audio_array as float32, sample_rate)
        """
        print(f"  Attempting to load audio from: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}")
            # Try to find similar files
            directory = os.path.dirname(filepath) or "/content"
            filename = os.path.basename(filepath)
            print(f"  Looking for similar files in {directory}...")
            try:
                files = os.listdir(directory)
                similar = [f for f in files if filename. split('. ')[0].replace('_', ' ').lower() in f.lower() 
                          or f.lower() in filename.replace('_', ' ').lower()]
                if similar:
                    print(f"  Found similar files: {similar}")
                    # Try the first similar file
                    filepath = os.path.join(directory, similar[0])
                    print(f"  Trying:  {filepath}")
            except Exception as e:
                print(f"  Could not search directory: {e}")
            
            if not os.path.exists(filepath):
                return np.zeros(self.target_sr, dtype=np.float32), self.target_sr
        
        print(f"  📊 File size: {os.path.getsize(filepath)} bytes")
        
        # Try multiple loading methods
        audio = None
        sr = None
        
        # Method 1: Direct librosa load
        try:
            print("  Trying librosa. load()...")
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            print(f"  ✅ Loaded with librosa: {len(audio)} samples at {sr}Hz")
        except Exception as e:
            print(f"  ⚠️ librosa. load() failed: {e}")
        
        # Method 2: Use pydub for m4a/mp3 files
        if audio is None:
            try: 
                print("  Trying pydub...")
                from pydub import AudioSegment
                
                # Determine format from extension
                ext = os.path.splitext(filepath)[1].lower().replace('.', '')
                if ext == 'm4a':
                    ext = 'mp4'  # pydub uses mp4 for m4a
                
                audio_segment = AudioSegment.from_file(filepath, format=ext)
                
                # Convert to mono
                audio_segment = audio_segment.set_channels(1)
                
                # Get sample rate
                sr = audio_segment.frame_rate
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Normalize to float32 [-1, 1]
                if audio_segment.sample_width == 2:  # 16-bit
                    audio = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    audio = samples. astype(np.float32) / 2147483648.0
                else:
                    audio = samples.astype(np. float32) / np.max(np.abs(samples))
                
                print(f"  ✅ Loaded with pydub: {len(audio)} samples at {sr}Hz")
                
            except Exception as e:
                print(f"  ⚠️ pydub failed:  {e}")
        
        # Method 3: Use moviepy for any audio file
        if audio is None: 
            try:
                print("  Trying moviepy...")
                from moviepy.editor import AudioFileClip
                
                clip = AudioFileClip(filepath)
                sr = clip.fps
                audio = clip.to_soundarray()
                
                # Convert to mono if stereo
                if len(audio. shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                elif len(audio.shape) > 1:
                    audio = audio.flatten()
                
                audio = audio.astype(np. float32)
                clip.close()
                
                print(f"  ✅ Loaded with moviepy: {len(audio)} samples at {sr}Hz")
                
            except Exception as e:
                print(f"  ⚠️ moviepy failed: {e}")
        
        # Method 4: Use soundfile
        if audio is None:
            try:
                print("  Trying soundfile...")
                import soundfile as sf
                audio, sr = sf.read(filepath)
                
                if len(audio.shape) > 1: 
                    audio = np.mean(audio, axis=1)
                
                audio = audio.astype(np.float32)
                print(f"  ✅ Loaded with soundfile: {len(audio)} samples at {sr}Hz")
                
            except Exception as e:
                print(f"  ⚠️ soundfile failed: {e}")
        
        # If all methods failed
        if audio is None:
            print("  ❌ All loading methods failed.  Returning silence.")
            return np.zeros(self.target_sr, dtype=np.float32), self.target_sr
        
        # Ensure float32 dtype
        audio = audio.astype(np. float32)
        
        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        print(f"  ✅ Audio loaded successfully: {len(audio)} samples, {len(audio)/sr:.2f} seconds")
        
        return audio, int(sr)
    
    def extract_from_video(self, video_filepath: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio track from video file using moviepy.
        
        Args:
            video_filepath: Path to video file
            
        Returns:
            Tuple of (audio_array as float32, sample_rate)
        """
        print(f"  Extracting audio from video: {video_filepath}")
        
        if not os.path.exists(video_filepath):
            print(f"  ❌ Video file not found: {video_filepath}")
            return np.zeros(self.target_sr, dtype=np.float32), self.target_sr
        
        try: 
            from moviepy.editor import VideoFileClip
            
            video = VideoFileClip(video_filepath)
            
            if video.audio is None:
                print("  ⚠️ Video has no audio track")
                video.close()
                return np. zeros(self.target_sr, dtype=np.float32), self.target_sr
            
            audio_fps = video.audio.fps
            audio_array = video.audio.to_soundarray()
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            elif len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            audio_array = audio_array.astype(np.float32)
            
            video.close()
            
            print(f"  ✅ Extracted {len(audio_array)} samples at {audio_fps}Hz")
            
            return audio_array, int(audio_fps)
            
        except Exception as e: 
            print(f"  ❌ Could not extract audio from video: {e}")
            return np.zeros(self.target_sr, dtype=np.float32), self.target_sr
    
    def resample(self, audio:  np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio array
            orig_sr:  Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array as float32
        """
        if orig_sr == target_sr: 
            return audio
        
        try:
            print(f"  Resampling from {orig_sr}Hz to {target_sr}Hz...")
            resampled = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
            print(f"  ✅ Resampled:  {len(audio)} → {len(resampled)} samples")
            return resampled. astype(np.float32)
        except Exception as e:
            print(f"  ⚠️ Resampling failed: {e}")
            return audio
    
    def compute_snr(self, audio: np. ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.
        
        Args:
            audio: Audio array
            
        Returns: 
            SNR in decibels
        """
        if len(audio) == 0:
            return 0.0
        
        signal_power = np.mean(audio ** 2)
        
        frame_size = 1024
        hop_size = 512
        
        if len(audio) < frame_size:
            return 10.0
        
        num_frames = (len(audio) - frame_size) // hop_size + 1
        frame_energies = []
        
        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            frame_energies.append(np.mean(frame ** 2))
        
        if not frame_energies:
            return 10.0
        
        sorted_energies = sorted(frame_energies)
        noise_frames = sorted_energies[: max(1, len(sorted_energies) // 10)]
        noise_power = np.mean(noise_frames)
        
        if noise_power <= 0:
            return 40.0
        
        snr = 10 * np. log10(signal_power / noise_power)
        return float(snr)