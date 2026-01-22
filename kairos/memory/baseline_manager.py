"""
Baseline Manager
Handles user-specific baseline calibration using Welford's algorithm
"""
import os
import json
import numpy as np
from typing import Optional, Dict, Any

from config import USER_BASELINES_DIR, BIOMARKER_DIM


class BaselineManager:
    """
    Manages user-specific baseline statistics. 
    Implements personalized baseline calibration to prevent
    natural traits from being misclassified as pathology.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Ensure directory exists
        os.makedirs(USER_BASELINES_DIR, exist_ok=True)
        
        self.stats_file = os.path.join(USER_BASELINES_DIR, f"{user_id}_stats.json")
        
        # Welford's algorithm state
        self.count = 0
        self.mean = None
        self.m2 = None  # Sum of squared differences
        
        # Load existing stats if available
        self._load_stats()
    
    def _load_stats(self):
        """Load user statistics from file."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    data = json. load(f)
                
                self.count = data.get("count", 0)
                self.mean = np.array(data.get("mean", []), dtype=np.float32) if data.get("mean") else None
                self.m2 = np.array(data.get("m2", []), dtype=np.float32) if data.get("m2") else None
                
                print(f"  Loaded baseline for user {self.user_id} (n={self.count})")
                
            except Exception as e:
                print(f"  Warning: Could not load baseline stats:  {e}")
                self._initialize_stats()
        else:
            self._initialize_stats()
    
    def _initialize_stats(self):
        """Initialize empty statistics."""
        self.count = 0
        self.mean = None
        self.m2 = None
    
    def _save_stats(self):
        """Save statistics to file."""
        data = {
            "user_id": self.user_id,
            "count": self. count,
            "mean": self.mean. tolist() if self.mean is not None else None,
            "m2": self.m2.tolist() if self.m2 is not None else None
        }
        
        try:
            with open(self. stats_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Warning: Could not save baseline stats: {e}")
    
    def get_baseline(self) -> Optional[np.ndarray]:
        """
        Get current baseline (mean) for the user.
        
        Returns:
            32-dim baseline vector or None if no baseline exists
        """
        return self.mean
    
    def get_std(self) -> Optional[np.ndarray]:
        """
        Get current standard deviation for the user. 
        
        Returns:
            32-dim std vector or None if insufficient data
        """
        if self.count < 2 or self.m2 is None:
            return None
        
        variance = self.m2 / (self.count - 1)
        return np.sqrt(variance)
    
    def update_baseline(
        self,
        raw_biomarker: np.ndarray,
        reliability_mask: np.ndarray
    ):
        """
        Update baseline using Welford's online algorithm.
        Only updates features with trust_mask > 0.
        
        Args:
            raw_biomarker: New 32-dim raw feature vector
            reliability_mask: 32-dim trust mask (1. 0 = trusted, 0.0 = missing)
        """
        # Initialize if first update
        if self.mean is None:
            self.mean = np.zeros(BIOMARKER_DIM, dtype=np.float32)
            self.m2 = np.zeros(BIOMARKER_DIM, dtype=np.float32)
        
        # Create mask for trusted features
        trusted_mask = reliability_mask > 0.5
        
        # Skip update if no trusted features
        if not np.any(trusted_mask):
            return
        
        self.count += 1
        
        # Welford's algorithm update (only for trusted features)
        delta = raw_biomarker - self.mean
        self.mean = np.where(trusted_mask, self.mean + delta / self.count, self.mean)
        
        delta2 = raw_biomarker - self.mean
        self.m2 = np.where(trusted_mask, self.m2 + delta * delta2, self.m2)
        
        # Save periodically (every 10 updates)
        if self.count % 10 == 0:
            self._save_stats()
    
    def force_save(self):
        """Force save current statistics."""
        self._save_stats()
    
    def get_z_score(self, raw_biomarker: np.ndarray) -> np.ndarray:
        """
        Calculate Z-scores relative to baseline. 
        
        Args:
            raw_biomarker: 32-dim raw feature vector
            
        Returns: 
            32-dim Z-score vector (0 if no baseline)
        """
        if self.mean is None or self.count < 2:
            return np.zeros_like(raw_biomarker)
        
        std = self.get_std()
        if std is None:
            return raw_biomarker - self.mean
        
        # Avoid division by zero
        std = np.where(std < 0.001, 0.001, std)
        
        z_scores = (raw_biomarker - self.mean) / std
        
        return z_scores
    
    def is_bootstrap(self) -> bool:
        """Check if this is the first session (bootstrap)."""
        return self. count == 0