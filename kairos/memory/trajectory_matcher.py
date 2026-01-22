"""
Trajectory Matcher
Matches biomarker trajectories over time to detect patterns. 

This enables recognition of:
- Escalation patterns (increasing distress over time)
- De-escalation (user calming down)
- Cyclical patterns (mood swings)
- Flat affect (depression indicator)
- Crisis approach patterns (can predict incoming crisis)

Uses Dynamic Time Warping (DTW) concepts for pattern matching.
"""
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import math


class TrajectoryMatcher:
    """
    Tracks biomarker trajectories and detects patterns.
    
    Features:
    - Rolling window of biomarker history
    - Pattern detection (escalation, de-escalation, volatility)
    - Trajectory similarity matching
    - Predictive alerts
    - Episode vector generation for similarity search
    """
    
    def __init__(self, biomarker_dim: int = 32, trajectory_window: int = 5):
        """
        Initialize trajectory matcher. 
        
        Args:
            biomarker_dim:  Dimension of biomarker vectors
            trajectory_window: Number of turns to track
        """
        self.biomarker_dim = biomarker_dim
        self.trajectory_window = trajectory_window
        
        # Rolling history
        self.biomarker_history:  deque = deque(maxlen=trajectory_window)
        self.turn_numbers: deque = deque(maxlen=trajectory_window)
        self.timestamps: deque = deque(maxlen=trajectory_window)
        self.emotions: deque = deque(maxlen=trajectory_window)
        self.distress_history: deque = deque(maxlen=trajectory_window * 2)  # Longer for trend
        
        # Pattern detection thresholds
        self.thresholds = {
            'escalation': 0.1,  # Trend slope threshold
            'de_escalation':  -0.1,
            'volatility': 0.2,  # Standard deviation threshold
            'flat_affect': 0.05,  # Low variance threshold
            'high_distress': 0.6,
            'critical_distress': 0.8,
        }
        
        # Distress weights (same as intervention tracker for consistency)
        self.distress_weights = {
            'jitter': 0.20,
            'shimmer': 0.15,
            'f0_variance': 0.10,  # Low F0 variance = flat affect
            'negative_sentiment': 0.30,
            'crying': 0.25,
        }
        
        # Alert history
        self.alerts: List[Dict[str, Any]] = []
        self.last_alert_turn:  int = -999
    
    def add_biomarker(
        self,
        biomarker: List[float],
        turn_number: int,
        emotion: str = None
    ):
        """
        Add a biomarker to the trajectory. 
        
        Args:
            biomarker: 32-dim biomarker vector
            turn_number: Current turn number
            emotion:  Detected emotion (optional)
        """
        # Convert and pad if needed
        if isinstance(biomarker, np.ndarray):
            biomarker = biomarker.tolist()
        
        while len(biomarker) < self.biomarker_dim:
            biomarker.append(0.0)
        biomarker = biomarker[:self.biomarker_dim]
        
        # Store
        self.biomarker_history.append(biomarker)
        self.turn_numbers.append(turn_number)
        self.timestamps.append(datetime.now())
        self.emotions.append(emotion)
        
        # Calculate and store distress
        distress = self._calculate_distress(biomarker)
        self.distress_history.append(distress)
    
    def get_trajectory_vector(self) -> List[float]:
        """
        Get flattened trajectory vector for similarity matching.
        
        Returns:
            160-dim vector (32 dims × 5 turns)
        """
        trajectory_dim = self.biomarker_dim * self.trajectory_window
        
        if len(self.biomarker_history) == 0:
            return [0.0] * trajectory_dim
        
        # Flatten available history (most recent last)
        flattened = []
        for biomarker in self.biomarker_history:
            flattened.extend(biomarker)
        
        # Pad with zeros if not enough history
        while len(flattened) < trajectory_dim:
            # Pad at the beginning (older turns)
            flattened = [0.0] * self.biomarker_dim + flattened
        
        return flattened[: trajectory_dim]
    
    def get_trajectory_pattern(self) -> Dict[str, Any]:
        """
        Analyze the trajectory for patterns.
        
        Returns:
            Dict with pattern type, confidence, and recommendations
        """
        if len(self.distress_history) < 2:
            return {
                "pattern": "INSUFFICIENT_DATA",
                "confidence":  0.0,
                "description": "Not enough data points for pattern detection",
                "recommendation": None,
                "distress_scores": list(self.distress_history)
            }
        
        distress_scores = list(self.distress_history)
        n = len(distress_scores)
        
        # Calculate statistics
        mean_distress = np.mean(distress_scores)
        std_distress = np.std(distress_scores)
        
        # Calculate trend using linear regression
        x = np.arange(n)
        if n >= 3:
            # Linear fit:  y = mx + b
            slope = np. polyfit(x, distress_scores, 1)[0]
        else:
            slope = distress_scores[-1] - distress_scores[0]
        
        # Recent trend (last 3 points)
        if n >= 3:
            recent_slope = np.polyfit(np.arange(3), distress_scores[-3:], 1)[0]
        else:
            recent_slope = slope
        
        # Detect patterns
        pattern = "STABLE"
        confidence = 0.5
        description = "No clear pattern detected"
        recommendation = "Continue attentive support"
        alert_level = "NONE"
        
        # Check for escalation
        if slope > self.thresholds['escalation']: 
            if recent_slope > slope * 1.5:
                pattern = "RAPID_ESCALATION"
                confidence = min(1.0, abs(recent_slope) * 3)
                description = "Distress is escalating rapidly"
                recommendation = "Consider de-escalation, grounding, check crisis indicators"
                alert_level = "HIGH"
            else:
                pattern = "ESCALATING"
                confidence = min(1.0, abs(slope) * 2)
                description = "User's distress is increasing over time"
                recommendation = "Monitor closely, consider de-escalation techniques"
                alert_level = "MODERATE"
        
        # Check for de-escalation
        elif slope < self.thresholds['de_escalation']:
            pattern = "DE_ESCALATING"
            confidence = min(1.0, abs(slope) * 2)
            description = "User is calming down"
            recommendation = "Continue current approach, reinforce progress"
            alert_level = "NONE"
        
        # Check for volatility
        elif std_distress > self.thresholds['volatility']:
            pattern = "VOLATILE"
            confidence = min(1.0, std_distress * 2)
            description = "User's emotional state is fluctuating significantly"
            recommendation = "Focus on stability, consistent support, grounding"
            alert_level = "MODERATE"
        
        # Check for flat affect (low variance + moderate distress)
        elif std_distress < self.thresholds['flat_affect'] and mean_distress > 0.3:
            pattern = "FLAT_AFFECT"
            confidence = 0.7
            description = "Consistently muted emotional expression (possible depression)"
            recommendation = "Gently explore, watch for depression indicators"
            alert_level = "LOW"
        
        # Check for stable states
        elif mean_distress < 0.3: 
            pattern = "STABLE_LOW"
            confidence = 0.7
            description = "User appears stable with low distress"
            recommendation = "Maintain connection, explore deeper if appropriate"
            alert_level = "NONE"
        
        elif mean_distress > self.thresholds['high_distress']:
            pattern = "STABLE_HIGH"
            confidence = 0.8
            description = "User has persistent high distress"
            recommendation = "Consider crisis resources, direct support, professional referral"
            alert_level = "HIGH"
        
        # Check for critical distress
        if distress_scores[-1] > self.thresholds['critical_distress']:
            pattern = "CRITICAL_DISTRESS"
            confidence = 0.9
            description = "Current distress level is critical"
            recommendation = "Immediate crisis support, safety check, resources"
            alert_level = "CRITICAL"
        
        return {
            "pattern": pattern,
            "confidence": round(confidence, 3),
            "description": description,
            "recommendation": recommendation,
            "alert_level": alert_level,
            "trend_slope": round(slope, 4),
            "recent_slope": round(recent_slope, 4),
            "mean_distress": round(mean_distress, 3),
            "std_distress": round(std_distress, 3),
            "current_distress": round(distress_scores[-1], 3),
            "distress_scores": [round(d, 3) for d in distress_scores],
            "turns_analyzed": n
        }
    
    def _calculate_distress(self, biomarker: List[float]) -> float:
        """Calculate distress score from biomarker."""
        if not biomarker or len(biomarker) < 8:
            return 0.5
        
        distress = 0.0
        
        # Jitter (anxiety)
        if biomarker[0] > 0:
            distress += biomarker[0] * self.distress_weights['jitter']
        
        # Shimmer (instability)
        if len(biomarker) > 1 and biomarker[1] > 0:
            distress += biomarker[1] * self.distress_weights['shimmer']
        
        # F0 variance - low variance indicates flat affect
        if len(biomarker) > 2:
            # If F0 variance is very negative (flat), add to distress
            if biomarker[2] < -0.3:
                distress += abs(biomarker[2]) * self.distress_weights['f0_variance']
        
        # Negative sentiment
        if len(biomarker) > 22 and biomarker[22] < 0:
            distress += abs(biomarker[22]) * self.distress_weights['negative_sentiment']
        
        # Crying
        if len(biomarker) > 25 and biomarker[25] > 0:
            distress += biomarker[25] * self.distress_weights['crying']
        
        return min(1.0, max(0.0, distress))
    
    def get_alert(self, turn_number: int, cooldown_turns: int = 3) -> Optional[Dict[str, Any]]: 
        """
        Check if current trajectory warrants an alert.
        
        Args:
            turn_number: Current turn number
            cooldown_turns:  Minimum turns between alerts
            
        Returns:
            Alert dict if warranted, None otherwise
        """
        # Check cooldown
        if turn_number - self.last_alert_turn < cooldown_turns:
            return None
        
        pattern_data = self.get_trajectory_pattern()
        
        if pattern_data['pattern'] == "INSUFFICIENT_DATA":
            return None
        
        # Alert on concerning patterns
        alert_patterns = {
            "CRITICAL_DISTRESS": "CRITICAL",
            "RAPID_ESCALATION": "HIGH",
            "STABLE_HIGH": "HIGH",
            "ESCALATING": "MODERATE",
            "VOLATILE": "MODERATE",
        }
        
        if pattern_data['pattern'] in alert_patterns:
            alert = {
                "turn_number":  turn_number,
                "pattern": pattern_data['pattern'],
                "level": alert_patterns[pattern_data['pattern']],
                "description": pattern_data['description'],
                "recommendation": pattern_data['recommendation'],
                "confidence": pattern_data['confidence'],
                "current_distress": pattern_data['current_distress'],
                "timestamp": datetime.now().isoformat()
            }
            
            self.alerts.append(alert)
            self.last_alert_turn = turn_number
            
            return alert
        
        return None
    
    def get_emotion_trajectory(self) -> List[Tuple[int, str]]:
        """Get trajectory of emotions."""
        return list(zip(self.turn_numbers, self.emotions))
    
    def get_distress_trend(self, n_points: int = 5) -> Dict[str, Any]:
        """Get recent distress trend."""
        scores = list(self.distress_history)[-n_points:]
        
        if len(scores) < 2:
            return {"trend": "UNKNOWN", "scores": scores}
        
        # Calculate change
        change = scores[-1] - scores[0]
        
        if change > 0.2:
            trend = "INCREASING"
        elif change < -0.2:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        
        return {
            "trend": trend,
            "change": round(change, 3),
            "scores": [round(s, 3) for s in scores],
            "current":  round(scores[-1], 3),
            "average": round(np.mean(scores), 3)
        }
    
    def calculate_similarity(self, other_trajectory: List[float]) -> float:
        """
        Calculate similarity between current trajectory and another.
        
        Uses cosine similarity on flattened trajectory vectors.
        """
        current = np.array(self.get_trajectory_vector())
        other = np.array(other_trajectory)
        
        if len(other) != len(current):
            return 0.0
        
        # Cosine similarity
        norm_current = np.linalg.norm(current)
        norm_other = np.linalg.norm(other)
        
        if norm_current == 0 or norm_other == 0:
            return 0.0
        
        return float(np.dot(current, other) / (norm_current * norm_other))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trajectory statistics."""
        pattern_data = self.get_trajectory_pattern()
        
        return {
            "history_length": len(self.biomarker_history),
            "trajectory_window": self.trajectory_window,
            "current_pattern": pattern_data['pattern'],
            "pattern_confidence": pattern_data['confidence'],
            "current_distress": pattern_data. get('current_distress', 0),
            "mean_distress": pattern_data. get('mean_distress', 0),
            "alerts_generated": len(self.alerts),
            "last_alert_turn":  self.last_alert_turn if self.last_alert_turn >= 0 else None,
            "emotions": list(self.emotions),
            "distress_trend": self.get_distress_trend()
        }
    
    def reset(self):
        """Reset trajectory history."""
        self.biomarker_history.clear()
        self.turn_numbers.clear()
        self.timestamps.clear()
        self.emotions.clear()
        self.distress_history. clear()
        self.alerts = []
        self.last_alert_turn = -999