"""
Intervention Tracker
Tracks which therapeutic interventions worked and which didn't.

This enables LEARNING from past effectiveness - if VALIDATION worked
well for this user when they were anxious, recommend it again. 

Stores: 
- Pre-intervention state (biomarkers)
- Intervention type used
- Post-intervention state (biomarkers from next turn)
- Success score (did distress decrease?)
"""
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math

import numpy as np


class InterventionTracker: 
    """
    Tracks intervention outcomes to learn what therapeutic approaches work.
    
    Intervention Types:
    - VALIDATION: Acknowledging and validating feelings
    - SUPPORT: Providing emotional support
    - GENTLE_EXPLORATION: Carefully exploring deeper
    - COGNITIVE_REFRAME: Offering different perspectives
    - GROUNDING: Grounding exercises for anxiety/dissociation
    - CRISIS_SUPPORT: Crisis intervention
    - DIRECT_ANSWER:  Answering factual questions
    
    Success is measured by:
    - Distress reduction (primary)
    - Emotional intensity change
    - User engagement (response length, depth)
    - Explicit feedback if given
    """
    
    def __init__(self, user_id: str, storage_path: str = "./interventions"):
        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.tracker_file = self.storage_path / f"{user_id}_interventions.json"
        
        # Intervention records
        self.interventions: List[Dict[str, Any]] = []
        
        # Aggregated outcomes by type
        self.intervention_outcomes: Dict[str, List[float]] = defaultdict(list)
        
        # Outcomes by emotion-intervention pair
        self.emotion_intervention_outcomes: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Statistics
        self.total_interventions = 0
        self.effective_count = 0
        self.ineffective_count = 0
        
        # Weights for distress calculation
        self.distress_weights = {
            'jitter': 0.20,
            'shimmer': 0.15,
            'pauses': 0.10,
            'negative_sentiment': 0.30,
            'crying': 0.25,
        }
        
        # Crisis thresholds
        self.crisis_thresholds = {
            'distress_score': 0.75,
            'keyword_weight': 0.85,
            'trajectory_escalation': 0.80,
            'trigger_entity_boost': 0.15,
        }
        
        # Require MULTIPLE signals for crisis
        self.crisis_confirmation_required = 2
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load intervention history from disk."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                
                self.interventions = data.get('interventions', [])
                
                # Reconstruct aggregates
                for intervention in self.interventions:
                    int_type = intervention.get('intervention_type', 'UNKNOWN')
                    success = intervention.get('success_score', 0.5)
                    emotion = intervention.get('user_emotion', 'unknown')
                    
                    self.intervention_outcomes[int_type]. append(success)
                    self.emotion_intervention_outcomes[emotion][int_type].append(success)
                    
                    if success > 0.5:
                        self.effective_count += 1
                    else:
                        self.ineffective_count += 1
                
                self.total_interventions = len(self.interventions)
                
                print(f"  [InterventionTracker] Loaded {self.total_interventions} intervention records")
                
            except Exception as e:
                print(f"  [InterventionTracker] Warning: Could not load data: {e}")
    
    def _save_data(self):
        """Save intervention history to disk."""
        try:
            data = {
                'user_id': self.user_id,
                'interventions': self.interventions[-500: ],  # Keep last 500
                'total_interventions': self.total_interventions,
                'effective_count': self.effective_count,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.tracker_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [InterventionTracker] Warning: Could not save data: {e}")
    
    def record_intervention(
        self,
        pre_state_vector: List[float],
        post_state_vector: List[float],
        intervention_type: str,
        user_emotion: str,
        session_id: str,
        turn_number: int = 0,
        user_text_length: int = 0,
        response_text_length: int = 0,
        was_crisis: bool = False,
        explicit_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record an intervention and calculate its outcome.
        
        Args:
            pre_state_vector:  Biomarker vector before response
            post_state_vector:  Biomarker vector after user's next message
            intervention_type: Type of intervention used
            user_emotion:  Detected user emotion
            session_id: Current session ID
            turn_number:  Turn number
            user_text_length: Length of user's next response
            response_text_length:  Length of system response
            was_crisis: Whether this was a crisis situation
            explicit_feedback: Any explicit user feedback
            
        Returns: 
            Dict with intervention outcome data
        """
        intervention_id = str(uuid.uuid4())[:8]
        
        # Calculate distress scores
        pre_distress = self._calculate_distress(pre_state_vector)
        post_distress = self._calculate_distress(post_state_vector)
        
        # Calculate success score
        # Positive = distress decreased
        distress_delta = pre_distress - post_distress
        
        # Base success from distress change
        # Map [-1, 1] distress delta to [0, 1] success
        base_success = 0.5 + (distress_delta * 0.5)
        
        # Adjust for engagement (longer responses suggest engagement)
        engagement_bonus = 0.0
        if user_text_length > 50:
            engagement_bonus = 0.05
        elif user_text_length > 100:
            engagement_bonus = 0.10
        
        # Adjust for explicit feedback
        feedback_adjustment = 0.0
        if explicit_feedback:
            feedback_lower = explicit_feedback.lower()
            if any(word in feedback_lower for word in ['thank', 'help', 'better', 'good', 'yes']):
                feedback_adjustment = 0.15
            elif any(word in feedback_lower for word in ['no', 'not', 'wrong', 'bad', 'stop']):
                feedback_adjustment = -0.15
        
        # Final success score
        success_score = max(0.0, min(1.0, base_success + engagement_bonus + feedback_adjustment))
        
        # Determine if effective
        is_effective = success_score > 0.5
        
        # Create intervention record
        intervention_data = {
            "id": intervention_id,
            "intervention_type": intervention_type,
            "user_emotion": user_emotion,
            "pre_distress": round(pre_distress, 4),
            "post_distress": round(post_distress, 4),
            "distress_delta": round(distress_delta, 4),
            "success_score": round(success_score, 4),
            "is_effective": is_effective,
            "was_crisis": was_crisis,
            "engagement_bonus": engagement_bonus,
            "feedback_adjustment": feedback_adjustment,
            "session_id": session_id,
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store
        self.interventions.append(intervention_data)
        self.intervention_outcomes[intervention_type].append(success_score)
        self.emotion_intervention_outcomes[user_emotion][intervention_type].append(success_score)
        
        self. total_interventions += 1
        if is_effective:
            self.effective_count += 1
        else:
            self.ineffective_count += 1
        
        # Periodic save
        if self.total_interventions % 5 == 0:
            self._save_data()
        
        return intervention_data
    
    def _calculate_distress(self, biomarker:  List[float]) -> float:
        """
        Calculate distress score from biomarker vector.
        
        Returns value 0-1 where higher = more distress. 
        """
        if not biomarker or len(biomarker) < 8:
            return 0.5  # Neutral if no data
        
        distress = 0.0
        
        # Jitter (index 0) - anxiety indicator
        if len(biomarker) > 0: 
            jitter = max(0, biomarker[0])
            distress += jitter * self.distress_weights['jitter']
        
        # Shimmer (index 1) - voice instability
        if len(biomarker) > 1:
            shimmer = max(0, biomarker[1])
            distress += shimmer * self.distress_weights['shimmer']
        
        # Pauses (index 7) - cognitive load
        if len(biomarker) > 7:
            pauses = max(0, biomarker[7])
            distress += pauses * self.distress_weights['pauses']
        
        # Negative sentiment (index 22)
        if len(biomarker) > 22:
            # Sentiment is -1 to 1, negative = distress
            neg_sentiment = max(0, -biomarker[22])
            distress += neg_sentiment * self. distress_weights['negative_sentiment']
        
        # Crying (index 25)
        if len(biomarker) > 25:
            crying = max(0, biomarker[25])
            distress += crying * self.distress_weights['crying']
        
        return min(1.0, distress)
    
    def get_recommended_intervention(
        self,
        current_emotion: str,
        current_biomarker: List[float] = None,
        exclude_types: List[str] = None,
        min_samples: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Get recommended intervention based on past effectiveness.
        
        Args:
            current_emotion: User's current emotion
            current_biomarker: Current biomarker vector
            exclude_types: Intervention types to exclude
            min_samples: Minimum samples needed for recommendation
            
        Returns:
            Dict with recommended intervention and confidence, or None
        """
        exclude_types = exclude_types or []
        
        # First try emotion-specific recommendation
        emotion_outcomes = self.emotion_intervention_outcomes.get(current_emotion, {})
        
        best_intervention = None
        best_score = 0.0
        best_confidence = 0.0
        
        for int_type, scores in emotion_outcomes.items():
            if int_type in exclude_types:
                continue
            if len(scores) < min_samples:
                continue
            
            avg_score = sum(scores) / len(scores)
            # Confidence based on sample size
            confidence = min(1.0, len(scores) / 10)
            
            if avg_score > best_score:
                best_score = avg_score
                best_intervention = int_type
                best_confidence = confidence
        
        # If no emotion-specific, fall back to general
        if not best_intervention or best_score < 0.5:
            for int_type, scores in self.intervention_outcomes.items():
                if int_type in exclude_types:
                    continue
                if len(scores) < min_samples:
                    continue
                
                avg_score = sum(scores) / len(scores)
                confidence = min(1.0, len(scores) / 20)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_intervention = int_type
                    best_confidence = confidence * 0.8  # Lower confidence for non-specific
        
        if best_intervention and best_score > 0.5:
            return {
                "intervention_type": best_intervention,
                "avg_success_score": round(best_score, 3),
                "confidence": round(best_confidence, 3),
                "sample_count": len(self.intervention_outcomes. get(best_intervention, [])),
                "emotion_specific": current_emotion in emotion_outcomes
            }
        
        return None
    
    def get_intervention_success_rate(self, intervention_type: str, emotion: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed success rate for a specific intervention type."""
        if emotion: 
            scores = self.emotion_intervention_outcomes.get(emotion, {}).get(intervention_type, [])
            context = f"{emotion}_{intervention_type}"
        else:
            scores = self.intervention_outcomes.get(intervention_type, [])
            context = intervention_type
        
        if not scores:
            return None
        
        return {
            "intervention_type":  intervention_type,
            "emotion": emotion,
            "avg_success":  round(sum(scores) / len(scores), 3),
            "sample_count": len(scores),
            "effective_rate": round(sum(1 for s in scores if s > 0.5) / len(scores), 3),
            "min_success": round(min(scores), 3),
            "max_success": round(max(scores), 3),
            "recent_trend": round(sum(scores[-5:]) / len(scores[-5:]), 3) if len(scores) >= 5 else None
        }
    
    def get_all_success_rates(self) -> Dict[str, Dict[str, Any]]:
        """Get success rates for all intervention types."""
        rates = {}
        for int_type in self.intervention_outcomes:
            rate_data = self.get_intervention_success_rate(int_type)
            if rate_data:
                rates[int_type] = rate_data
        return rates
    
    def get_ineffective_interventions(self, emotion: str = None) -> List[str]:
        """Get list of intervention types that haven't worked well."""
        ineffective = []
        
        source = self.emotion_intervention_outcomes.get(emotion, {}) if emotion else self.intervention_outcomes
        
        for int_type, scores in source.items():
            if len(scores) >= 3:  # Need minimum samples
                avg_score = sum(scores) / len(scores)
                if avg_score < 0.4: 
                    ineffective.append(int_type)
        
        return ineffective
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive intervention tracking statistics."""
        success_rates = {}
        for int_type, scores in self.intervention_outcomes. items():
            if scores:
                success_rates[int_type] = round(sum(scores) / len(scores), 3)
        
        most_effective = None
        least_effective = None
        if success_rates:
            sorted_rates = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            most_effective = sorted_rates[0][0] if sorted_rates[0][1] > 0.5 else None
            least_effective = sorted_rates[-1][0] if sorted_rates[-1][1] < 0.5 else None
        
        return {
            "total_interventions": self.total_interventions,
            "effective_count": self.effective_count,
            "ineffective_count": self.ineffective_count,
            "effectiveness_rate": round(self.effective_count / max(1, self.total_interventions), 3),
            "intervention_types_used": list(self.intervention_outcomes.keys()),
            "intervention_success_rates": success_rates,
            "most_effective_intervention": most_effective,
            "least_effective_intervention": least_effective,
            "emotions_tracked": list(self.emotion_intervention_outcomes.keys()),
            "recent_interventions": [
                {
                    "type": i['intervention_type'],
                    "emotion": i['user_emotion'],
                    "success":  i['success_score'],
                    "effective":  i['is_effective']
                }
                for i in self.interventions[-5:]
            ]
        }
    
    def force_save(self):
        """Force save to disk."""
        self._save_data()