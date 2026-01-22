"""
Acoustic Engine (Module A)
FIX #5: Better calibration and more accurate feature extraction
"""
import numpy as np
from typing import Optional, Tuple, List

import librosa
from scipy import signal as scipy_signal

from config import AUDIO_SAMPLE_RATE


class AcousticEngine: 
    """
    Module A:  Acoustic Feature Extraction (8 features)
    
    FIX #5: Improved calibration
    - Features normalized to -1 to 1 range
    - 0 = typical baseline
    - Values are damped to avoid inflation
    - Clear interpretations for each feature
    """
    
    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        
        # Calibration constants based on research on speech patterns
        # These represent typical values for neutral speech
        self.baselines = {
            'jitter': {'mean': 0.015, 'std': 0.01, 'max_raw': 0.1},
            'shimmer': {'mean': 0.04, 'std': 0.025, 'max_raw': 0.2},
            'f0_cv': {'mean': 0.2, 'std': 0.1, 'max_raw':  0.6},
            'loudness_range': {'mean': 20, 'std': 8, 'max_raw': 50},  # dB
            'teo': {'mean': 1.0, 'std': 0.3, 'max_raw': 3.0},
            'hnr': {'mean': 12, 'std': 5, 'max_raw': 30},  # dB
            'speech_rate': {'mean': 4.5, 'std': 1.2, 'max_raw': 10},  # syllables/sec
            'pause_rate': {'mean': 2.5, 'std': 1.5, 'max_raw': 10}  # pauses/min
        }
    
    def extract(self, audio:  np.ndarray) -> np.ndarray:
        """
        Extract all 8 acoustic features. 
        
        Returns values normalized to -1 to 1 range where: 
        - 0 = typical/baseline
        - Positive = elevated
        - Negative = reduced
        """
        features = np.zeros(8, dtype=np.float32)
        
        if audio is None:
            print("     [AcousticEngine] Audio is None")
            return features
        
        print(f"     [AcousticEngine] Received audio:  shape={audio.shape}")
        
        if len(audio) < self.sample_rate // 2:  # Less than 0.5 seconds
            print(f"     [AcousticEngine] Audio too short: {len(audio)} samples")
            return features
        
        duration = len(audio) / self.sample_rate
        print(f"     [AcousticEngine] Processing {len(audio)} samples ({duration:.2f}s)")
        
        # Normalize audio
        audio = audio. astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        else:
            print("     [AcousticEngine] Audio is silent")
            return features
        
        # Check if audio has actual content
        rms = np.sqrt(np. mean(audio**2))
        if rms < 0.01: 
            print(f"     [AcousticEngine] Audio appears nearly silent (RMS={rms:.4f})")
            return features
        
        print(f"     [AcousticEngine] Audio RMS: {rms:.4f}")
        
        try:
            # Extract F0 (pitch)
            print("     [AcousticEngine] Extracting F0...")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=60,
                fmax=400,  # More conservative range
                sr=self.sample_rate,
                frame_length=2048,
                hop_length=512
            )
            
            f0_valid = f0[~np.isnan(f0)]
            print(f"     [AcousticEngine] F0: {len(f0_valid)} valid frames out of {len(f0)}")
            
            if len(f0_valid) < 5:
                print("     [AcousticEngine] Insufficient voiced frames, using fallback")
                f0_valid = self._fallback_f0(audio)
            
            # Extract each feature with proper calibration
            features[0] = self._extract_jitter(f0_valid)
            features[1] = self._extract_shimmer(audio)
            features[2] = self._extract_f0_variance(f0_valid)
            features[3] = self._extract_loudness_dynamics(audio)
            features[4] = self._extract_teo(audio)
            features[5] = self._extract_hnr(audio, f0_valid)
            features[6] = self._extract_speech_rate(audio, voiced_probs)
            features[7] = self._extract_pause_frequency(audio, duration)
            
            # Log results with interpretation
            names = ['Jitter', 'Shimmer', 'F0_Var', 'Loudness', 'TEO', 'HNR', 'SpeechRate', 'PauseFreq']
            for name, val in zip(names, features):
                interp = self._interpret_value(val)
                print(f"     [AcousticEngine] {name}: {val:.4f} ({interp})")
            
        except Exception as e:
            print(f"     [AcousticEngine] ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        return features
    
    def _interpret_value(self, val: float) -> str:
        """Interpret a normalized value."""
        if val > 0.7:
            return "VERY HIGH"
        elif val > 0.4:
            return "HIGH"
        elif val > 0.2:
            return "elevated"
        elif val < -0.7:
            return "VERY LOW"
        elif val < -0.4:
            return "LOW"
        elif val < -0.2:
            return "reduced"
        else:
            return "normal"
    
    def _normalize_feature(self, raw_value: float, feature_name: str) -> float:
        """Normalize a raw feature value to -1 to 1 range."""
        baseline = self.baselines. get(feature_name, {'mean': 0, 'std': 1})
        
        # Z-score normalization
        z_score = (raw_value - baseline['mean']) / (baseline['std'] + 1e-8)
        
        # Soft clip to -1 to 1 using tanh
        # This prevents extreme values while keeping relative ordering
        normalized = np.tanh(z_score * 0.5)  # 0.5 dampens the response
        
        return float(normalized)
    
    def _fallback_f0(self, audio: np.ndarray) -> np.ndarray:
        """Fallback pitch detection using piptrack."""
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                fmin=60,
                fmax=400
            )
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if 60 < pitch < 400:
                    pitch_values.append(pitch)
            
            if len(pitch_values) >= 3:
                return np.array(pitch_values)
        except:
            pass
        
        # Return default if all else fails
        return np.array([150.0, 150.0, 150.0])
    
    def _extract_jitter(self, f0: np.ndarray) -> float:
        """
        Jitter:  Cycle-to-cycle pitch variation.
        HIGH = anxiety, nervousness, emotional distress
        """
        if len(f0) < 3:
            return 0.0
        
        try:
            # Convert to periods
            f0_safe = np.maximum(f0, 50)
            periods = 1.0 / f0_safe
            
            # Compute relative average perturbation
            diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            
            if mean_period <= 0:
                return 0.0
            
            jitter_raw = np.mean(diffs) / mean_period
            
            # Normalize
            return self._normalize_feature(jitter_raw, 'jitter')
            
        except Exception as e:
            return 0.0
    
    def _extract_shimmer(self, audio: np.ndarray) -> float:
        """
        Shimmer: Cycle-to-cycle amplitude variation.
        HIGH = emotional turbulence, voice instability
        """
        try:
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) < 3:
                return 0.0
            
            # Compute amplitude perturbation
            diffs = np.abs(np. diff(rms))
            mean_rms = np.mean(rms)
            
            if mean_rms <= 0:
                return 0.0
            
            shimmer_raw = np. mean(diffs) / mean_rms
            
            return self._normalize_feature(shimmer_raw, 'shimmer')
            
        except: 
            return 0.0
    
    def _extract_f0_variance(self, f0: np.ndarray) -> float:
        """
        F0 Variance: Pitch variation (coefficient of variation).
        LOW = monotone, flat affect (depression indicator)
        HIGH = animated, expressive
        """
        if len(f0) < 3:
            return 0.0
        
        try:
            mean_f0 = np.mean(f0)
            if mean_f0 < 50:
                return 0.0
            
            # Coefficient of variation
            cv = np. std(f0) / mean_f0
            
            return self._normalize_feature(cv, 'f0_cv')
            
        except: 
            return 0.0
    
    def _extract_loudness_dynamics(self, audio: np.ndarray) -> float:
        """
        Loudness Dynamics: Range of intensity. 
        LOW = subdued, withdrawn, low energy
        HIGH = assertive, expressive, possibly agitated
        """
        try:
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) < 3:
                return 0.0
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)
            
            # Dynamic range (95th - 5th percentile)
            p95 = np.percentile(rms_db, 95)
            p5 = np. percentile(rms_db, 5)
            dynamic_range = p95 - p5
            
            return self._normalize_feature(dynamic_range, 'loudness_range')
            
        except: 
            return 0.0
    
    def _extract_teo(self, audio: np.ndarray) -> float:
        """
        TEO (Teager Energy Operator): Vocal effort/stress.
        HIGH = emotional intensity, stress
        """
        if len(audio) < 3:
            return 0.0
        
        try:
            # TEO:  x[n]^2 - x[n-1]*x[n+1]
            teo = audio[1:-1]**2 - audio[:-2] * audio[2:]
            
            # Average TEO normalized by signal energy
            signal_energy = np.mean(audio**2)
            if signal_energy < 1e-8:
                return 0.0
            
            teo_normalized = np.mean(np.abs(teo)) / signal_energy
            
            return self._normalize_feature(teo_normalized, 'teo')
            
        except:
            return 0.0
    
    def _extract_hnr(self, audio: np.ndarray, f0: np.ndarray) -> float:
        """
        HNR (Harmonics-to-Noise Ratio): Voice clarity.
        LOW = breathy, strained, hoarse
        HIGH = clear voice
        """
        if len(f0) < 1 or len(audio) < 2048:
            return 0.0
        
        try:
            avg_f0 = np.mean(f0)
            if avg_f0 < 60:
                return 0.0
            
            period_samples = int(self.sample_rate / avg_f0)
            
            # Autocorrelation-based HNR estimation
            frame_size = min(4096, len(audio))
            frame = audio[:frame_size]
            
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) <= period_samples:
                return 0.0
            
            # Find peak at pitch period
            search_start = max(1, period_samples - 10)
            search_end = min(len(autocorr) - 1, period_samples + 10)
            
            if search_end <= search_start:
                return 0.0
            
            r_max = np.max(autocorr[search_start:search_end])
            r_0 = autocorr[0]
            
            if r_0 <= 0:
                return 0.0
            
            r_ratio = r_max / r_0
            r_ratio = np.clip(r_ratio, 0.01, 0.99)
            
            # Convert to dB
            hnr_db = 10 * np.log10(r_ratio / (1 - r_ratio + 1e-8))
            
            return self._normalize_feature(hnr_db, 'hnr')
            
        except:
            return 0.0
    
    def _extract_speech_rate(self, audio: np.ndarray, voiced_probs: np.ndarray = None) -> float:
        """
        Speech Rate: Syllables per second.
        HIGH = anxious, rushing, manic
        LOW = depressed, tired, cognitive load
        """
        try:
            duration = len(audio) / self.sample_rate
            if duration < 0.5:
                return 0.0
            
            # Use onset detection as syllable proxy
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                backtrack=False
            )
            
            syllable_rate = len(onsets) / duration
            
            return self._normalize_feature(syllable_rate, 'speech_rate')
            
        except: 
            return 0.0
    
    def _extract_pause_frequency(self, audio: np. ndarray, duration: float) -> float:
        """
        Pause Frequency: Number of significant pauses.
        HIGH = hesitation, cognitive load, difficulty expressing
        """
        try:
            if duration < 2: 
                return 0.0
            
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) == 0:
                return 0.0
            
            # Dynamic threshold
            threshold = np.percentile(rms, 25)
            
            # Minimum pause duration:  250ms
            min_pause_frames = int(0.25 * self.sample_rate / hop_length)
            
            # Count pauses
            is_pause = rms < threshold
            pause_count = 0
            current_pause = 0
            
            for is_p in is_pause:
                if is_p:
                    current_pause += 1
                else:
                    if current_pause >= min_pause_frames: 
                        pause_count += 1
                    current_pause = 0
            
            if current_pause >= min_pause_frames:
                pause_count += 1
            
            # Pauses per minute
            pauses_per_min = pause_count / (duration / 60)
            
            return self._normalize_feature(pauses_per_min, 'pause_rate')
            
        except: 
            return 0.0