"""
Reliability Gate
Computes dynamic quality scores and reliability masks
"""
import numpy as np
from typing import Dict, Any, List, Optional

from config import STATIC_WEIGHTS, MIN_SNR_THRESHOLD, MIN_FACE_CONFIDENCE, BIOMARKER_DIM


class ReliabilityGate:
    """
    Implements Phase 2:  The Reliability Gating Logic
    Establishes trust before feature extraction. 
    """
    
    def __init__(self):
        self.static_weights = STATIC_WEIGHTS
    
    def compute_dynamic_quality(
        self,
        audio_array: Optional[np.ndarray] = None,
        video_array: Optional[np.ndarray] = None,
        modality: str = "text",
        video_missing: bool = True,
        audio_missing: bool = True
    ) -> Dict[str, float]:
        """
        Compute dynamic quality scores for each modality.
        
        Args:
            audio_array: Audio data array
            video_array: Video frame array
            modality: Input modality type
            video_missing: Whether video is unavailable
            audio_missing: Whether audio is unavailable
            
        Returns:
            Dict with quality scores for audio, video, linguistic
        """
        quality = {
            "audio": 0.0,
            "video": 0.0,
            "linguistic": 1.0,  # Always available if we have text
            "latency": 1.0
        }
        
        # Audio quality assessment
        if not audio_missing and audio_array is not None and len(audio_array) > 0:
            snr = self._estimate_snr(audio_array)
            if snr >= MIN_SNR_THRESHOLD: 
                quality["audio"] = min(1.0, snr / 30.0)  # Normalize SNR
            else:
                quality["audio"] = 0.0
        
        # Video quality assessment
        if not video_missing and video_array is not None and len(video_array) > 0:
            # Basic check:  non-zero frames
            if np.any(video_array > 0):
                quality["video"] = 0.8  # Assume decent quality if frames exist
            else:
                quality["video"] = 0.0
        
        # Modality-based gates
        if modality == "text":
            quality["audio"] = 0.0
            quality["video"] = 0.0
            quality["latency"] = 0.0  # Can't measure latency for text-only
        elif modality == "audio":
            quality["video"] = 0.0
        
        return quality
    
    def compute_mask(
        self,
        quality_scores: Dict[str, float],
        modality: str = "text"
    ) -> np.ndarray:
        """
        Compute the 32-dimensional reliability mask.
        
        Combines static weights with dynamic quality scores.
        
        Args:
            quality_scores: Dynamic quality scores from compute_dynamic_quality
            modality: Input modality type
            
        Returns:
            32-dim reliability mask (1. 0 = trusted, 0.0 = missing)
        """
        mask = np.zeros(BIOMARKER_DIM, dtype=np.float32)
        
        # Acoustic features (0-7)
        acoustic_weight = self.static_weights["acoustic"] * quality_scores. get("audio", 0.0)
        mask[0:8] = acoustic_weight
        
        # Visual features (8-15)
        visual_weight = self.static_weights["visual"] * quality_scores.get("video", 0.0)
        mask[8:16] = visual_weight
        
        # Linguistic features (16-23)
        linguistic_weight = self.static_weights["linguistic"] * quality_scores.get("linguistic", 1.0)
        mask[16:24] = linguistic_weight
        
        # Special signals (24-31)
        # Laughter detection depends on acoustic
        mask[24] = acoustic_weight
        # Reserved features
        mask[25:32] = 0.0
        
        return mask
    
    def _estimate_snr(self, audio:  np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio. 
        
        Args:
            audio: Audio array
            
        Returns:
            SNR in dB
        """
        if len(audio) == 0:
            return 0.0
        
        # Simple SNR estimation
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise from quietest portions
        frame_size = min(1024, len(audio) // 10)
        if frame_size < 100:
            return 10.0  # Default for very short audio
        
        hop = frame_size // 2
        frame_powers = []
        
        for i in range(0, len(audio) - frame_size, hop):
            frame = audio[i:i + frame_size]
            frame_powers.append(np.mean(frame ** 2))
        
        if not frame_powers:
            return 10.0
        
        # Noise estimate from bottom 10%
        sorted_powers = sorted(frame_powers)
        noise_frames = sorted_powers[: max(1, len(sorted_powers) // 10)]
        noise_power = np.mean(noise_frames)
        
        if noise_power <= 1e-10:
            return 40.0  # Very clean signal
        
        snr = 10 * np. log10(signal_power / noise_power)
        return float(max(0, snr))