"""
Biomarker Tracker
Tracks biomarker signals across turns for trend analysis and LLM context.
"""
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from datetime import datetime

from kairos.memory.trajectory_matcher import TrajectoryMatcher


class BiomarkerTracker: 
    """
    Tracks biomarker signals across conversation turns. 
    
    Features:
    - Rolling history of biomarker vectors
    - Delta analysis (change from previous turn)
    - Trend detection (escalation, de-escalation)
    - Summary generation for LLM context
    - Trajectory pattern matching
    """
    
    def __init__(self, history_size: int = 15, biomarker_dim: int = 32):
        self.history_size = history_size
        self.biomarker_dim = biomarker_dim
        
        # Rolling history
        self.biomarker_history:  deque = deque(maxlen=history_size)
        self.turn_numbers: deque = deque(maxlen=history_size)
        self.modalities: deque = deque(maxlen=history_size)
        self.timestamps: deque = deque(maxlen=history_size)
        
        # Trajectory matcher for pattern detection
        self.trajectory_matcher = TrajectoryMatcher(
            biomarker_dim=biomarker_dim,
            trajectory_window=5
        )
        
        # Feature names mapping - Updated to match FeatureEngine output
        self.feature_names = {
            # === MODULE A: ACOUSTIC (0-7) ===
            0: 'jitter',           # Voice tremor
            1: 'shimmer',          # Voice instability
            2: 'f0_variance',      # Monotone vs Animated
            3: 'loudness_range',   # Dynamic range
            4: 'teo',              # Vocal effort/stress
            5: 'hnr',              # Voice quality/clarity
            6: 'speech_rate',      # Speed
            7: 'pause_rate',       # Hesitation

            # === MODULE B: VISUAL (8-15) ===
            8: 'masking_score',    # Fake/held expression
            9: 'brow_tension',     # Stress/concentration
            10: 'gaze_aversion',   # Looking away/avoidance
            11: 'facial_dynamism', # Animation level
            12: 'stare_duration',  # Fixed gaze
            13: 'blink_rate',      # Anxiety indicator
            14: 'head_nodding',    # Agreement/submission
            15: 'head_tilt',       # Engagement

            # === MODULE C: LINGUISTIC (16-23) ===
            16: 'absolutist_index', # "always", "never"
            17: 'i_ratio',          # Self-focus
            18: 'response_latency', # Delay
            19: 'lexical_density',  # Complexity
            20: 'past_tense_ratio', # Stuck in past
            21: 'filler_density',   # Cognitive load
            22: 'sentiment',        # Semantic sentiment
            23: 'rumination',       # Repetitive negative thought

            # === MODULE D: SPECIAL (24-31) ===
            24: 'laughter',
            25: 'crying',
            26: 'sigh',
            27: 'strain'
        }
        
        # Interpretation thresholds
        self.thresholds = {
            # Acoustic
            'jitter_high': 0.3,
            'shimmer_high': 0.3,
            'f0_variance_low': -0.3,
            'f0_variance_high': 0.3,
            'speech_rate_fast': 0.3,
            'speech_rate_slow': -0.3,
            'pause_rate_high': 0.3,
            
            # Visual (0.0 to 1.0 range from VisualEngine)
            'brow_tension_high': 0.6,
            'gaze_aversion_high': 0.5,
            'stare_high': 0.7,
            'masking_high': 0.6,
            'dynamism_low': 0.2,
            'blink_high': 0.7,
            
            # Linguistic
            'sentiment_negative': -0.3,
            'sentiment_positive': 0.3,
            'rumination_high': 0.5,
            'absolutist_high': 0.5,
            
            # Special
            'crying_detected': 0.3,
            'laughter_detected': 0.3,
            'arousal_high': 0.6,
            'arousal_low': -0.3,
        }
        
        # Significant change threshold
        self.significant_change_threshold = 0.15
    
    def add_turn(
        self,
        biomarker:  List[float],
        modality:  str = "text",
        turn_number: int = 0
    ):
        """
        Add a new turn's biomarker to history.
        
        Args:
            biomarker:  Biomarker vector
            modality: Input modality
            turn_number: Turn number
        """
        # Ensure proper format
        if isinstance(biomarker, np.ndarray):
            biomarker = biomarker.tolist()
        
        # Pad if needed
        while len(biomarker) < self.biomarker_dim:
            biomarker.append(0.0)
        biomarker = biomarker[:self.biomarker_dim]
        
        # Store
        self.biomarker_history.append(biomarker)
        self.turn_numbers.append(turn_number)
        self.modalities.append(modality)
        self.timestamps.append(datetime.now())
        
        # Update trajectory matcher
        self.trajectory_matcher.add_biomarker(
            biomarker=biomarker,
            turn_number=turn_number
        )
    
    def get_delta(self) -> Dict[str, Any]:  
        """
        Get change from previous turn.
        
        Returns:
            Dict with delta analysis
        """
        if len(self.biomarker_history) < 2:
            return {
                'has_delta': False,
                'message': 'Not enough history for delta'
            }
        
        current = np.array(self.biomarker_history[-1])
        previous = np.array(self.biomarker_history[-2])
        delta = current - previous
        
        # Find significant changes
        significant_changes = []
        for i, change in enumerate(delta):
            if abs(change) >= self.significant_change_threshold:
                feature_name = self.feature_names. get(i, f'feature_{i}')
                interpretation = self._interpret_change(i, change, current[i])
                direction = "increased" if change > 0 else "decreased"
                significant_changes.append({
                    'index': i,
                    'feature':  feature_name,
                    'change': float(change),
                    'direction':  direction,
                    'current_value': float(current[i]),
                    'interpretation': interpretation
                })
        
        # Sort by absolute change
        significant_changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return {
            'has_delta':  True,
            'significant_changes': significant_changes[: 5],  # Top 5
            'total_changes': len(significant_changes),
            'delta_magnitude': float(np.linalg.norm(delta)),
            'previous_turn': self.turn_numbers[-2] if len(self.turn_numbers) >= 2 else None,
            'current_turn': self.turn_numbers[-1] if self.turn_numbers else None
        }
    
    def _interpret_change(self, index:  int, change: float, current_value: float) -> str:
        """Interpret a biomarker change."""
        feature = self.feature_names.get(index, f'feature_{index}')
        direction = "increased" if change > 0 else "decreased"
        
        interpretations = {
            0: f"Voice tremor {direction}" + (" (anxiety indicator)" if change > 0 else " (calming)"),
            1: f"Voice instability {direction}" + (" (emotional distress)" if change > 0 else " (stabilizing)"),
            2: f"Pitch variation {direction}" + (" (more animated)" if change > 0 else " (flatter/depressed)"),
            3: f"Volume {direction}",
            6: f"Speech rate {direction}" + (" (anxious/excited)" if change > 0 else " (slowing down)"),
            7: f"Pauses {direction}" + (" (hesitation/processing)" if change > 0 else " (more fluent)"),
            # Visual changes
            9: f"Brow tension {direction}" + (" (more stress)" if change > 0 else " (relaxing)"),
            10: f"Gaze aversion {direction}",
            11: f"Facial expression {direction}" + (" (more dynamic)" if change > 0 else " (becoming flatter)"),
            # Linguistic changes
            22: f"Sentiment {direction}" + (" (more positive)" if change > 0 else " (more negative)"),
            23: f"Rumination {direction}" + (" (more repetitive)" if change > 0 else " (less repetitive)"),
            24: f"Laughter {direction}",
            25: f"Crying indicators {direction}",
        }
        
        return interpretations.get(index, f"{feature} {direction}")
    
    def get_trajectory_patterns(self) -> Dict[str, Any]:
        """Get trajectory patterns from the trajectory matcher."""
        return self.trajectory_matcher.get_trajectory_pattern()
    
    def get_trajectory_alert(self, turn_number: int = 0) -> Optional[Dict[str, Any]]: 
        """Get trajectory alert if warranted."""
        return self.trajectory_matcher.get_alert(turn_number)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current biomarker state with interpretations.
        """
        if not self.biomarker_history:
            return {
                'has_data': False,
                'message': 'No biomarker data yet'
            }
        
        current = self.biomarker_history[-1]
        modality = self.modalities[-1] if self.modalities else 'unknown'
        
        if modality == 'text':
            return {
                'has_data': True,
                'modality': 'text',
                'message':  'Text-only input, no voice/visual signals'
            }
        
        # Interpret current state
        interpretations = []
        
        # --- ACOUSTIC (0-7) ---
        # Jitter (anxiety)
        if len(current) > 0 and current[0] > self.thresholds['jitter_high']:
            interpretations.append({
                'signal': 'jitter',
                'value': current[0],
                'interpretation': 'Voice tremor detected (anxiety/stress indicator)'
            })
        
        # Shimmer (instability)
        if len(current) > 1 and current[1] > self.thresholds['shimmer_high']:
            interpretations.append({
                'signal': 'shimmer',
                'value': current[1],
                'interpretation': 'Voice instability (emotional distress)'
            })
        
        # F0 variance (flat affect)
        if len(current) > 2 and current[2] < self.thresholds['f0_variance_low']:
            interpretations.append({
                'signal': 'f0_variance',
                'value': current[2],
                'interpretation': 'Flat pitch (possible depression/numbness)'
            })
        
        # Speech rate
        if len(current) > 6: 
            if current[6] > self.thresholds['speech_rate_fast']:
                interpretations.append({
                    'signal': 'speech_rate',
                    'value': current[6],
                    'interpretation': 'Fast speech (anxiety/excitement)'
                })
            elif current[6] < self. thresholds['speech_rate_slow']:
                interpretations.append({
                    'signal': 'speech_rate',
                    'value': current[6],
                    'interpretation': 'Slow speech (sadness/fatigue)'
                })
        
        # Pause rate
        if len(current) > 7 and current[7] > self. thresholds['pause_rate_high']:
            interpretations.append({
                'signal': 'pause_rate',
                'value': current[7],
                'interpretation': 'Frequent pauses (processing/hesitation)'
            })
        
        # --- VISUAL (8-15) ---
        if modality in ['video', 'face']:
            # Brow Tension (Index 9)
            if len(current) > 9 and current[9] > self.thresholds['brow_tension_high']:
                interpretations.append({
                    'signal': 'brow_tension',
                    'value': current[9],
                    'interpretation': 'High brow tension (stress/worry)'
                })
            
            # Gaze Aversion (Index 10)
            if len(current) > 10 and current[10] > self.thresholds['gaze_aversion_high']:
                interpretations.append({
                    'signal': 'gaze_aversion',
                    'value': current[10],
                    'interpretation': 'Avoiding eye contact (shame/discomfort)'
                })
            
            # Facial Dynamism (Index 11) - Flat Affect
            if len(current) > 11 and current[11] < self.thresholds['dynamism_low']:
                interpretations.append({
                    'signal': 'flat_affect',
                    'value': current[11],
                    'interpretation': 'Reduced facial expression (flat affect)'
                })
            
            # Masking (Index 8)
            if len(current) > 8 and current[8] > self.thresholds['masking_high']:
                interpretations.append({
                    'signal': 'masking',
                    'value': current[8],
                    'interpretation': 'Potential emotional masking detected'
                })
        
        # --- LINGUISTIC (16-23) ---
        # Sentiment
        if len(current) > 22:
            if current[22] < self.thresholds['sentiment_negative']:
                interpretations.append({
                    'signal': 'sentiment',
                    'value': current[22],
                    'interpretation': 'Negative vocal/semantic sentiment'
                })
            elif current[22] > self.thresholds['sentiment_positive']:
                interpretations.append({
                    'signal': 'sentiment',
                    'value': current[22],
                    'interpretation': 'Positive vocal/semantic sentiment'
                })
        
        # Rumination
        if len(current) > 23 and current[23] > self.thresholds['rumination_high']:
             interpretations.append({
                'signal': 'rumination',
                'value': current[23],
                'interpretation': 'High rumination (repetitive negative thoughts)'
            })

        # --- SPECIAL (24-31) ---
        # Crying
        if len(current) > 25 and current[25] > self. thresholds['crying_detected']: 
            interpretations.append({
                'signal': 'crying',
                'value': current[25],
                'interpretation': 'Crying detected in voice'
            })
        
        # Laughter
        if len(current) > 24 and current[24] > self. thresholds['laughter_detected']:
            interpretations.append({
                'signal': 'laughter',
                'value': current[24],
                'interpretation':  'Laughter detected'
            })
        
        return {
            'has_data':  True,
            'modality':  modality,
            'interpretations': interpretations,
            'turn_number': self.turn_numbers[-1] if self.turn_numbers else 0,
            'raw_vector': current[: 10]  # First 10 for debug
        }
    
    def get_trend(self, n_turns: int = 5) -> Dict[str, Any]:
        """
        Analyze trend over recent turns.
        
        Args:
            n_turns: Number of turns to analyze
        """
        if len(self.biomarker_history) < 2:
            return {
                'has_trend':  False,
                'message': 'Not enough history for trend'
            }
        
        # Get recent history
        recent = list(self.biomarker_history)[-n_turns:]
        
        if len(recent) < 2:
            return {
                'has_trend': False,
                'message': 'Not enough history for trend'
            }
        
        # Calculate trends for key features
        trends = {}
        # Added indices for visual (9=brow) and linguistic (23=rumination)
        feature_indices = [0, 1, 2, 6, 7, 9, 22, 23, 25]  
        
        for idx in feature_indices:
            if idx < len(recent[0]):
                values = [b[idx] for b in recent if len(b) > idx]
                if len(values) >= 2:
                    # Linear trend
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    feature = self.feature_names.get(idx, f'feature_{idx}')
                    
                    if abs(slope) > 0.05: 
                        trends[feature] = {
                            'slope': float(slope),
                            'direction': 'increasing' if slope > 0 else 'decreasing',
                            'values': [float(v) for v in values]
                        }
        
        # Overall distress trend
        distress_scores = []
        for b in recent:
            distress = 0.0
            if len(b) > 0:
                distress += max(0, b[0]) * 0.2  # jitter
            if len(b) > 1:
                distress += max(0, b[1]) * 0.15  # shimmer
            if len(b) > 9:
                 distress += max(0, b[9]) * 0.15 # brow tension
            if len(b) > 22:
                distress += max(0, -b[22]) * 0.2  # negative sentiment
            if len(b) > 23:
                distress += max(0, b[23]) * 0.1  # rumination
            if len(b) > 25:
                distress += max(0, b[25]) * 0.2  # crying
            distress_scores.append(min(1.0, distress))
        
        distress_trend = 'stable'
        if len(distress_scores) >= 2:
            change = distress_scores[-1] - distress_scores[0]
            if change > 0.1:
                distress_trend = 'increasing'
            elif change < -0.1:
                distress_trend = 'decreasing'
        
        return {
            'has_trend':  True,
            'turns_analyzed': len(recent),
            'feature_trends': trends,
            'distress_scores': distress_scores,
            'distress_trend': distress_trend,
            'overall_direction': distress_trend
        }
    
    def get_session_trend(self, n_turns: int = None) -> Dict[str, Any]:
        """
        Get session-wide trend analysis.
        Alias for get_trend() with session-specific defaults.
        
        Args:
            n_turns: Number of turns to analyze (default: all session turns)
        
        Returns:
            Dict with trend analysis
        """
        if n_turns is None:
            n_turns = len(self. biomarker_history)
        
        if n_turns < 1:
            n_turns = 5  # Default fallback
        
        return self. get_trend(n_turns)
    
    def get_biomarker_summary_for_llm(self) -> str:
        """
        Get formatted biomarker summary for LLM context.
        
        Returns:
            Formatted string for LLM
        """
        if not self.biomarker_history:
            return ""
        
        current_state = self.get_current_state()
        
        if not current_state. get('has_data'):
            return ""
        
        if current_state.get('modality') == 'text':
            return ""  # No voice/visual data
        
        lines = [f"=== BIOMARKER ANALYSIS ({current_state. get('modality', 'unknown').upper()}) ==="]
        
        # Current interpretations
        interpretations = current_state.get('interpretations', [])
        if interpretations:
            lines.append("Current signals:")
            for interp in interpretations[: 5]: 
                lines.append(f"  • {interp['interpretation']} ({interp['value']:.2f})")
        else:
            lines.append("No notable signals in current input")
        
        # Delta from previous
        delta = self.get_delta()
        if delta.get('has_delta') and delta.get('significant_changes'):
            lines.append("\nChanges from last turn:")
            for change in delta['significant_changes'][:3]: 
                direction = "↑" if change['change'] > 0 else "↓"
                lines.append(f"  • {direction} {change['interpretation']}")
        
        # Trend
        trend = self.get_trend()
        if trend.get('has_trend'):
            lines.append(f"\nDistress trend: {trend['distress_trend']. upper()}")
            if trend. get('feature_trends'):
                for feature, data in list(trend['feature_trends'].items())[:2]:
                    lines.append(f"  • {feature}: {data['direction']}")
        
        # Trajectory patterns
        trajectory_patterns = self.get_trajectory_patterns()
        if trajectory_patterns.get('pattern') != 'INSUFFICIENT_DATA':
            pattern = trajectory_patterns.get('pattern', 'UNKNOWN')
            confidence = trajectory_patterns.get('confidence', 0)
            if confidence > 0.5:
                lines.append(f"\nTrajectory pattern: {pattern} (confidence: {confidence:.2f})")
                if trajectory_patterns.get('recommendation'):
                    lines.append(f"  Recommendation: {trajectory_patterns['recommendation']}")
        
        return "\n".join(lines)
    
    def detect_masking(self) -> Dict[str, Any]: 
        """
        Detect if there's a mismatch suggesting masking.
        
        Returns:
            Dict with masking analysis
        """
        if not self.biomarker_history:
            return {'detected': False, 'reason': 'No data'}
        
        current = self.biomarker_history[-1]
        
        # Check for contradictory signals
        contradictions = []
        
        # High arousal (proxied by loudness/jitter) but positive sentiment (nervous positivity)
        # Using Loudness (3) as proxy for arousal since index 23 is now Rumination
        if len(current) > 22 and len(current) > 3:
            if current[3] > 0.5 and current[22] > 0.3:
                contradictions.append({
                    'type': 'nervous_positivity',
                    'description': 'High loudness/arousal with positive words (possibly masking anxiety)'
                })
        
        # High jitter but neutral/positive sentiment (hidden anxiety)
        if len(current) > 0 and len(current) > 22:
            if current[0] > 0.4 and current[22] > 0:
                contradictions.append({
                    'type': 'hidden_anxiety',
                    'description': 'Voice tremor despite positive words (possible hidden anxiety)'
                })
        
        # Flat affect but high arousal indicators (suppressed emotions)
        if len(current) > 2 and len(current) > 3:
            if current[2] < -0.3 and current[3] > 0.4:
                contradictions.append({
                    'type': 'suppressed_emotions',
                    'description': 'Flat voice pitch despite high volume/arousal (possibly suppressing emotions)'
                })
        
        # Visual Masking: High masking score (8) + high brow tension (9)
        if len(current) > 9:
             if current[8] > 0.5 and current[9] > 0.5:
                contradictions.append({
                    'type': 'visual_masking',
                    'description': 'High facial masking score with high brow tension'
                })

        # Crying + laughter (mixed signals)
        if len(current) > 24 and len(current) > 25:
            if current[24] > 0.3 and current[25] > 0.3:
                contradictions.append({
                    'type': 'mixed_signals',
                    'description': 'Both laughter and crying detected (complex emotional state)'
                })
        
        return {
            'detected': len(contradictions) > 0,
            'contradictions': contradictions,
            'masking_likely': len(contradictions) >= 2
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'history_length': len(self.biomarker_history),
            'max_history':  self.history_size,
            'modalities_seen': list(set(self.modalities)),
            'turns_tracked': list(self.turn_numbers),
            'trajectory_stats': self.trajectory_matcher.get_stats()
        }
    
    def reset(self):
        """Reset tracker."""
        self.biomarker_history. clear()
        self.turn_numbers.clear()
        self.modalities.clear()
        self.timestamps.clear()
        self.trajectory_matcher.reset()