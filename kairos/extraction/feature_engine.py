"""
Feature Engine
Main coordinator for all feature extraction modules
Produces the 32-dimensional phenotype vector
"""
import numpy as np
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer

from . acoustic_engine import AcousticEngine
from .visual_engine import VisualEngine
from .linguistic_engine import LinguisticEngine
from .special_signals import SpecialSignalsEngine
from kairos.utils.reliability_gate import ReliabilityGate
from kairos.memory.baseline_manager import BaselineManager
from config import (
    SEMANTIC_DIM, BIOMARKER_DIM, RELIABILITY_DIM,
    STATIC_WEIGHTS, MIN_SNR_THRESHOLD, MIN_FACE_CONFIDENCE
)


class FeatureEngine: 
    """
    Main feature extraction coordinator. 
    Converts raw multimodal buffers into clinically valid 32-dimensional phenotype vector.
    Includes personalized baseline calibration. 
    """
    
    def __init__(self, baseline_manager: BaselineManager):
        self.baseline_manager = baseline_manager
        
        # Initialize sub-engines
        self.acoustic_engine = AcousticEngine()
        self.visual_engine = VisualEngine()
        self.linguistic_engine = LinguisticEngine()
        self.special_signals = SpecialSignalsEngine()
        self.reliability_gate = ReliabilityGate()
        
        # Initialize semantic encoder
        print("  Loading sentence transformer model...")
        self.semantic_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("  Sentence transformer loaded")
    
    def extract(self, input_object: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from input. 
        
        Args:
            input_object: Dict containing audio_array, video_array, text, segments, etc.
            
        Returns:
            Dict with vectors (semantic, biomarker) and payload (reliability_mask, raw_metrics)
        """
        print("  Extracting features...")
        
        # Get inputs
        audio_array = input_object.get("audio_array")
        video_array = input_object.get("video_array")
        text = input_object. get("text", "")
        segments = input_object.get("segments", [])
        modality = input_object.get("modality_type", "text")
        video_missing = input_object.get("video_missing", True)
        audio_missing = input_object.get("audio_missing", True)
        
        # Debug: Print what we received
        print(f"     [FeatureEngine] Modality: {modality}")
        print(f"     [FeatureEngine] Audio missing: {audio_missing}")
        print(f"     [FeatureEngine] Video missing: {video_missing}")
        
        if audio_array is not None:
            print(f"     [FeatureEngine] Audio array:  shape={audio_array.shape}, dtype={audio_array.dtype}")
            print(f"     [FeatureEngine] Audio range: [{audio_array.min():.4f}, {audio_array.max():.4f}]")
            print(f"     [FeatureEngine] Audio non-zero: {np.count_nonzero(audio_array)} / {len(audio_array)}")
        else:
            print(f"     [FeatureEngine] Audio array:  None")
        
        # Combine segment texts if needed
        if not text and segments:
            text = " ".join(seg. get("text", "") for seg in segments)
        
        # Phase 2: Dynamic Quality Check
        quality_scores = self.reliability_gate.compute_dynamic_quality(
            audio_array=audio_array,
            video_array=video_array,
            modality=modality,
            video_missing=video_missing,
            audio_missing=audio_missing
        )
        
        print(f"     [FeatureEngine] Quality scores: {quality_scores}")
        
        # Phase 3: Extract raw features from each module
        
        # Module A: Acoustic features (indices 0-7)
        if not audio_missing and audio_array is not None and len(audio_array) > 0:
            print(f"     [FeatureEngine] Extracting acoustic features...")
            acoustic_features = self.acoustic_engine.extract(audio_array)
        else:
            print(f"     [FeatureEngine] Skipping acoustic features (audio missing or empty)")
            acoustic_features = np.zeros(8, dtype=np.float32)
        
        print(f"     [FeatureEngine] Acoustic features:  {acoustic_features}")
        
        # Module B:  Visual features (indices 8-15)
        if not video_missing and video_array is not None and len(video_array) > 0 and np.any(video_array > 0):
            print(f"     [FeatureEngine] Extracting visual features...")
            visual_features = self.visual_engine.extract(video_array)
        else:
            print(f"     [FeatureEngine] Skipping visual features (video missing or empty)")
            visual_features = np.zeros(8, dtype=np.float32)
        
        # Module C: Linguistic features (indices 16-23)
        print(f"     [FeatureEngine] Extracting linguistic features...")
        linguistic_features = self.linguistic_engine.extract(
            text,
            input_object.get("last_system_end_time")
        )
        print(f"     [FeatureEngine] Linguistic features: {linguistic_features}")
        
        # Module D: Special signals (indices 24-31)
        special_features = self.special_signals.extract(
            acoustic_features=acoustic_features,
            audio_array=audio_array
        )
        
        # Combine into raw biomarker vector
        raw_biomarker = np.concatenate([
            acoustic_features,
            visual_features,
            linguistic_features,
            special_features
        ]).astype(np.float32)
        
        print(f"     [FeatureEngine] Raw biomarker (first 8): {raw_biomarker[:8]}")
        
        # Phase 4: Apply reliability weighting and baseline correction
        
        # Step A: Compute reliability mask
        reliability_mask = self.reliability_gate.compute_mask(
            quality_scores=quality_scores,
            modality=modality
        )
        
        # Step B: Apply baseline correction
        corrected_biomarker, raw_metrics = self._apply_baseline_correction(
            raw_biomarker,
            reliability_mask
        )
        
        # Generate semantic embedding
        semantic_vector = self._encode_semantic(text)
        
        # Phase 5: Prepare final output
        return {
            "vectors": {
                "semantic": semantic_vector. tolist(),
                "biomarker": corrected_biomarker.tolist(),
                "reliability":  reliability_mask.tolist()
            },
            "payload": {
                "reliability_mask": reliability_mask.tolist(),
                "raw_metrics": raw_metrics,
                "modality_type": modality,
                "quality_scores": quality_scores
            },
            "text": text
        }
    
    def _apply_baseline_correction(
        self,
        raw_biomarker:  np.ndarray,
        reliability_mask: np.ndarray
    ) -> tuple:
        """
        Apply personalized baseline correction.
        
        Formula: Feature_final = Feature_raw - μ_user_baseline
        """
        # Get user baseline
        baseline = self.baseline_manager.get_baseline()
        
        # Calculate delta from baseline
        if baseline is not None:
            corrected_biomarker = raw_biomarker - baseline
        else:
            # First session: use raw values (no baseline yet)
            corrected_biomarker = raw_biomarker. copy()
        
        # DON'T zero out - we want to see the actual values
        # The reliability mask is stored separately for the LLM to know what to trust
        
        # Update baseline with trusted features
        self.baseline_manager. update_baseline(raw_biomarker, reliability_mask)
        
        # Prepare raw metrics dict for storage
        raw_metrics = {
            "jitter_raw": float(raw_biomarker[0]),
            "shimmer_raw": float(raw_biomarker[1]),
            "f0_var_raw": float(raw_biomarker[2]),
            "loudness_raw": float(raw_biomarker[3]),
            "speech_rate_raw": float(raw_biomarker[6]),
            "sentiment_raw": float(raw_biomarker[22]) if len(raw_biomarker) > 22 else 0.0,
            "laughter_raw": float(raw_biomarker[24]) if len(raw_biomarker) > 24 else 0.0
        }
        
        return corrected_biomarker. astype(np.float32), raw_metrics
    
    def _encode_semantic(self, text:  str) -> np.ndarray:
        """Generate semantic embedding from text."""
        if not text or not text.strip():
            return np. zeros(SEMANTIC_DIM, dtype=np.float32)
        
        try:
            embedding = self.semantic_encoder.encode(text, convert_to_numpy=True)
            return embedding. astype(np.float32)
        except Exception as e: 
            print(f"  Warning:  Semantic encoding failed: {e}")
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)