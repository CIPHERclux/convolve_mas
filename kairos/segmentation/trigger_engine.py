"""
Trigger Engine (Step 2)
Semantic segmentation using faster_whisper transcription
"""
import numpy as np
from typing import List, Dict, Any, Optional


class TriggerEngine:
    """
    Determines where the "events" are in the file.
    Uses faster_whisper for transcription and segmentation.
    """
    
    def __init__(self):
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the Whisper model"""
        if self._model_loaded:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            print("  Loading Whisper model (this may take a moment)...")
            
            # Use small model for balance of speed and accuracy
            # In Colab with T4, we can use float16
            self.model = WhisperModel(
                "small",
                device="cuda",
                compute_type="float16"
            )
            self._model_loaded = True
            print("  Whisper model loaded successfully")
            
        except Exception as e: 
            print(f"  Warning:  Could not load CUDA model, falling back to CPU: {e}")
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8"
                )
                self._model_loaded = True
                print("  Whisper model loaded on CPU")
            except Exception as e2:
                print(f"  Error loading Whisper model: {e2}")
                self. model = None
    
    def segment(self, input_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segment input into semantic events.
        
        If Modality == "text": Skip AI processing.  Create a single "Pseudo-Segment".
        Else (Video/Audio): Run faster_whisper on the entire audio_array at once.
        
        Args:
            input_object: Dict containing audio_array, text, and modality_type
            
        Returns:
            List of segment dicts:  [{start, end, text}, ...]
        """
        modality = input_object.get("modality_type", "text")
        text = input_object.get("text")
        audio_array = input_object.get("audio_array")
        
        if modality == "text": 
            # Skip AI processing for text-only input
            return [{
                "start": 0.0,
                "end": 0.0,
                "text": text,
                "confidence": 1.0
            }]
        
        # For video/audio, transcribe
        return self._transcribe_audio(audio_array, input_object. get("duration", 0.0))
    
    def _transcribe_audio(self, audio_array: np.ndarray, duration: float) -> List[Dict[str, Any]]:
        """
        Transcribe audio using faster_whisper.
        
        Args:
            audio_array: Float32 numpy array of audio at 16kHz
            duration:  Audio duration in seconds
            
        Returns:
            List of segment dicts with timestamps and text
        """
        # Load model if needed
        self._load_model()
        
        if self.model is None:
            return [{
                "start": 0.0,
                "end": duration,
                "text": "[Transcription unavailable]",
                "confidence":  0.0
            }]
        
        try:
            print("  Transcribing audio...")
            
            # Ensure audio is in correct format
            if audio_array. dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val
            
            # Transcribe entire audio at once
            segments_gen, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language="en",  # Can be set to None for auto-detection
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500
                }
            )
            
            # Collect all segments
            segments = []
            full_text_parts = []
            
            for segment in segments_gen:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text. strip(),
                    "confidence":  segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.9
                })
                full_text_parts.append(segment.text.strip())
            
            print(f"  Transcribed {len(segments)} segments")
            
            # If no segments detected, create placeholder
            if not segments:
                return [{
                    "start": 0.0,
                    "end": duration,
                    "text": "[No speech detected]",
                    "confidence": 0.0
                }]
            
            return segments
            
        except Exception as e:
            print(f"  Warning: Transcription failed: {e}")
            return [{
                "start": 0.0,
                "end": duration,
                "text": "[Transcription error]",
                "confidence": 0.0
            }]
    
    def get_full_text(self, segments: List[Dict[str, Any]]) -> str:
        """
        Combine all segment texts into full transcript.
        
        Args:
            segments: List of segment dicts
            
        Returns:
            Combined text string
        """
        return " ".join(seg. get("text", "") for seg in segments if seg.get("text"))