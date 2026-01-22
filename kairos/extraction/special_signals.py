"""
Special Signals Engine - FIXED
FIX #2: Much more conservative detection, fewer false positives
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa


class SpecialSignalsEngine: 
    """
    FIX #2: Much more conservative thresholds. 
    Default to LOW probability, require STRONG evidence to increase.
    """
    
    def __init__(self):
        self.sample_rate = 16000
    
    def extract(
        self,
        acoustic_features: np.ndarray = None,
        audio_array: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract special signal features. 
        
        FIX #2: Much more conservative defaults
        """
        features = np.zeros(8, dtype=np.float32)
        
        # VERY conservative defaults - assume nothing detected
        features[0] = 0.05  # Laughter - very low default
        features[1] = 0.05  # Crying - very low default
        features[2] = 0.05  # Sigh
        features[3] = 0.05  # Strain
        
        if audio_array is None or len(audio_array) < self.sample_rate:
            return features
        
        try:
            # Only increase from baseline if we have STRONG evidence
            
            # Laughter - need multiple strong indicators
            laughter_score = self._detect_laughter_conservative(audio_array, acoustic_features)
            features[0] = laughter_score
            
            # Crying - need voice breaks + distress in voice
            crying_score = self._detect_crying_conservative(audio_array, acoustic_features)
            features[1] = crying_score
            
            # Sigh
            features[2] = self._detect_sigh_conservative(audio_array)
            
            # Strain
            features[3] = self._detect_strain_conservative(acoustic_features)
            
            print(f"     [SpecialSignals] Laughter: {features[0]:.3f}, "
                  f"Crying:  {features[1]:.3f}, Sigh: {features[2]:.3f}, "
                  f"Strain: {features[3]:.3f}")
            
        except Exception as e:
            print(f"     [SpecialSignals] Error: {e}")
        
        return features
    
    def _detect_laughter_conservative(self, audio:  np.ndarray, 
                                      acoustic_features: np.ndarray = None) -> float:
        """
        FIX #2: Very conservative laughter detection. 
        Only report high probability if MULTIPLE strong indicators present.
        """
        try:
            duration = len(audio) / self.sample_rate
            if duration < 1.0:
                return 0.05
            
            evidence_score = 0.0
            indicators_found = 0
            
            # Indicator 1: Rhythmic energy bursts (characteristic of laughter)
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                                      hop_length=hop_length)[0]
            
            if len(rms) > 20:
                # Look for regular bursts
                rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
                
                # Find peaks
                peaks = []
                for i in range(1, len(rms_norm) - 1):
                    if (rms_norm[i] > rms_norm[i-1] and 
                        rms_norm[i] > rms_norm[i+1] and 
                        rms_norm[i] > 0.5):  # Higher threshold
                        peaks.append(i)
                
                # Check for rhythmic pattern (regular spacing)
                if len(peaks) >= 5:  # Need more peaks
                    intervals = np.diff(peaks)
                    if len(intervals) > 0:
                        interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                        if interval_cv < 0.3:  # Very regular = laughter
                            evidence_score += 0.25
                            indicators_found += 1
            
            # Indicator 2: High F0 variance (pitch jumps during laughter)
            if acoustic_features is not None and len(acoustic_features) > 2:
                f0_var = acoustic_features[2]
                if f0_var > 0.5:  # HIGH pitch variation
                    evidence_score += 0.2
                    indicators_found += 1
            
            # Indicator 3: Specific spectral pattern
            # Laughter has characteristic high-frequency bursts
            spec = np.abs(librosa.stft(audio))
            n_bins = spec.shape[0]
            
            # Check for intermittent high-frequency energy
            high_freq = spec[n_bins*2//3:, :]
            hf_variance = np.var(np.mean(high_freq, axis=0))
            
            if hf_variance > 0.01:  # Variable high-frequency = possible laughter
                evidence_score += 0.15
                indicators_found += 1
            
            # REQUIRE multiple indicators for confident detection
            if indicators_found >= 2:
                final_score = min(0.7, 0.1 + evidence_score)
            elif indicators_found == 1:
                final_score = min(0.3, 0.1 + evidence_score * 0.5)
            else:
                final_score = 0.05
            
            return final_score
            
        except: 
            return 0.05
    
    def _detect_crying_conservative(self, audio: np.ndarray,
                                    acoustic_features: np.ndarray = None) -> float:
        """
        FIX #2: Conservative crying detection.
        Need voice breaks AND high vocal distress indicators.
        """
        try:
            duration = len(audio) / self.sample_rate
            if duration < 1.5:
                return 0.05
            
            evidence_score = 0.0
            indicators_found = 0
            
            # Indicator 1: Voice breaks (sudden drops in energy)
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                                      hop_length=hop_length)[0]
            
            if len(rms) > 10:
                rms_diff = np.diff(rms)
                threshold = np.std(rms_diff) * 2.5  # Higher threshold
                sudden_drops = np.sum(rms_diff < -threshold)
                drops_per_sec = sudden_drops / duration
                
                if drops_per_sec > 2.0:  # Multiple voice breaks per second
                    evidence_score += 0.3
                    indicators_found += 1
                elif drops_per_sec > 1.0:
                    evidence_score += 0.15
                    indicators_found += 1
            
            # Indicator 2: High jitter + shimmer (vocal instability)
            if acoustic_features is not None and len(acoustic_features) >= 2:
                jitter = acoustic_features[0]
                shimmer = acoustic_features[1]
                
                # Need BOTH to be elevated
                if jitter > 0.4 and shimmer > 0.4:
                    evidence_score += 0.25
                    indicators_found += 1
                elif jitter > 0.3 and shimmer > 0.3:
                    evidence_score += 0.1
            
            # Indicator 3: Irregular rhythm (crying is NOT rhythmic like laughter)
            if len(rms) > 30:
                # Check periodicity - crying should be aperiodic
                autocorr = np.correlate(rms - np.mean(rms), 
                                       rms - np.mean(rms), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if len(autocorr) > 20:
                    periodicity = np.max(autocorr[10:40]) / (autocorr[0] + 1e-8)
                    if periodicity < 0.2:  # Very aperiodic
                        evidence_score += 0.15
                        indicators_found += 1
            
            # REQUIRE multiple indicators
            if indicators_found >= 2:
                final_score = min(0.8, 0.1 + evidence_score)
            elif indicators_found == 1:
                final_score = min(0.35, 0.1 + evidence_score * 0.5)
            else:
                final_score = 0.05
            
            return final_score
            
        except:
            return 0.05
    
    def _detect_sigh_conservative(self, audio: np.ndarray) -> float:
        """Conservative sigh detection."""
        try:
            duration = len(audio) / self.sample_rate
            if duration < 0.5:
                return 0.05
            
            frame_length = int(0.05 * self.sample_rate)
            hop_length = int(0.025 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                                      hop_length=hop_length)[0]
            
            # Look for sustained declining energy (sigh pattern)
            sigh_score = 0.05
            for i in range(len(rms) - 8):
                segment = rms[i:i+8]
                # Check for smooth decline
                if all(segment[j] >= segment[j+1] * 0.9 for j in range(7)):
                    if np.mean(segment) > 0.15 * np.max(rms):
                        sigh_score = min(0.5, sigh_score + 0.15)
            
            return sigh_score
            
        except: 
            return 0.05
    
    def _detect_strain_conservative(self, acoustic_features: np.ndarray = None) -> float:
        """Conservative strain detection based on acoustic features."""
        if acoustic_features is None or len(acoustic_features) < 6:
            return 0.05
        
        try:
            strain_score = 0.05
            
            jitter = acoustic_features[0]
            shimmer = acoustic_features[1]
            hnr = acoustic_features[5]
            
            # Strain = moderate jitter/shimmer + low HNR
            if jitter > 0.3: 
                strain_score += 0.15
            if shimmer > 0.3:
                strain_score += 0.15
            if hnr < -0.3:
                strain_score += 0.2
            
            return min(0.7, strain_score)
            
        except: 
            return 0.05