"""
Salience Tracker
Calculates salience scores for memories to determine which should always be retrieved. 

High salience = core memories (crisis, breakthroughs, important revelations)
Low salience = can fade (small talk, greetings)

This implements memory prioritization - not all memories are equal.
Critical moments should ALWAYS be retrievable regardless of semantic similarity.
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import math


class SalienceTracker: 
    """
    Tracks memory salience for prioritized retrieval.
    
    Salience factors:
    - Emotional intensity (high emotion = important)
    - Crisis/trauma content (always important)
    - Novel information (first mentions of facts)
    - User-indicated importance ("this is important", "I need to tell you")
    - Therapeutic breakthroughs (insights, realizations)
    - Session anchors (first turns establish context)
    - Entity significance (mentions of trigger entities)
    
    Salience Score Ranges:
    - 0.85-1.0: CORE_MEMORY - Always retrieve
    - 0.70-0.84: IMPORTANT - High priority retrieval
    - 0.40-0.69: MODERATE - Normal retrieval
    - 0.00-0.39: ROUTINE - May fade over time
    """
    
    def __init__(self):
        # Salience thresholds
        self.thresholds = {
            'always_retrieve': 0.85,
            'high_salience': 0.70,
            'moderate_salience': 0.40,
        }
        
        # Weights for different factors
        self.weights = {
            'crisis':  0.40,
            'emotional_intensity': 0.25,
            'novel_facts': 0.20,
            'user_importance': 0.15,
            'breakthrough': 0.20,
            'session_anchor':  0.30,
            'trigger_entity': 0.15,
            'text_depth': 0.10,
        }
        
        # Keywords indicating user-marked importance
        self.importance_markers = [
            'this is important',
            'i need to tell you',
            'i have to say',
            'listen',
            'please remember',
            "don't forget",
            'this matters',
            'big deal',
            'huge',
            'never told anyone',
            'first time saying',
            'confession',
            'secret',
        ]
        
        # Keywords indicating breakthroughs
        self.breakthrough_markers = [
            'i realized',
            'i understand now',
            'it hit me',
            'i finally see',
            'makes sense now',
            'i get it',
            'eureka',
            'aha',
            'lightbulb',
            'clicked',
            'figured out',
            'breakthrough',
            'epiphany',
        ]
        
        # Session tracking
        self.session_memories:  List[Dict[str, Any]] = []
        self.session_stats = {
            'total_count': 0,
            'core_memory_count': 0,
            'important_count': 0,
            'moderate_count': 0,
            'routine_count': 0,
            'total_salience': 0.0,
        }
        
        # Historical tracking for decay calculations
        self.salience_history: Dict[str, List[float]] = defaultdict(list)
    
    def calculate_salience(
        self,
        text: str,
        emotional_intensity: float,
        crisis_score: float,
        is_crisis: bool,
        detected_emotion: str,
        turn_number: int = 0,
        is_first_turn: bool = False,
        contains_new_facts: bool = False,
        new_facts_count: int = 0,
        trigger_entities: List[str] = None,
        modality: str = "text",
        biomarker_notable: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive salience score for a memory. 
        
        Args:
            text: The user's message
            emotional_intensity:  0-1 intensity score
            crisis_score: 0-1 crisis score
            is_crisis: Boolean crisis flag
            detected_emotion: Primary emotion detected
            turn_number:  Turn number in session
            is_first_turn: Whether this is turn 0
            contains_new_facts:  Whether new facts were extracted
            new_facts_count:  Number of new facts extracted
            trigger_entities: List of trigger entities mentioned
            modality: Input modality (text/audio/video)
            biomarker_notable: Whether biomarkers showed notable signals
            
        Returns:
            Dict with salience_score, category, factors, and metadata
        """
        salience = 0.0
        factors = []
        factor_contributions = {}
        
        text_lower = text.lower() if text else ""
        
        # =====================================================================
        # FACTOR 1: Crisis Content (highest weight)
        # =====================================================================
        if is_crisis:
            contribution = self. weights['crisis']
            salience += contribution
            factors.append("crisis_content")
            factor_contributions['crisis'] = contribution
        elif crisis_score > 0.5:
            contribution = self.weights['crisis'] * 0.6
            salience += contribution
            factors.append("elevated_risk")
            factor_contributions['elevated_risk'] = contribution
        elif crisis_score > 0.3:
            contribution = self.weights['crisis'] * 0.3
            salience += contribution
            factors.append("mild_risk_indicators")
            factor_contributions['mild_risk'] = contribution
        
        
        if emotional_intensity > 0.6:  # Was 0.7
            contribution = self.weights['emotional_intensity']
            salience += contribution
            factors.append("very_high_intensity")
            factor_contributions['intensity'] = contribution
        elif emotional_intensity > 0.3:  # Was 0.4
            contribution = self.weights['emotional_intensity'] * 0.95 
            salience += contribution
            factors.append("high_intensity")
            factor_contributions['intensity'] = contribution
        elif emotional_intensity > 0.25:  # NEW tier for moderate emotions
            contribution = self.weights['emotional_intensity'] * 0.6
            salience += contribution
            factors.append("moderate_intensity")
            factor_contributions['intensity'] = contribution
        
        
        # FACTOR 3: Session Anchor (first turns)
        
        if is_first_turn or turn_number == 0:
            contribution = self.weights['session_anchor']
            salience += contribution
            factors.append("session_start")
            factor_contributions['session_anchor'] = contribution
        elif turn_number <= 2:
            contribution = self.weights['session_anchor'] * 0.5
            salience += contribution
            factors.append("early_session")
            factor_contributions['early_session'] = contribution
        
        # =====================================================================
        # FACTOR 4: Novel Facts
        # =====================================================================
        if contains_new_facts:
            base_contribution = self.weights['novel_facts']
            # Bonus for multiple facts
            fact_multiplier = min(1.5, 1.0 + (new_facts_count - 1) * 0.2)
            contribution = base_contribution * fact_multiplier
            salience += contribution
            factors.append(f"new_facts_{new_facts_count}")
            factor_contributions['novel_facts'] = contribution
        
        # =====================================================================
        # FACTOR 5: User-Indicated Importance
        # =====================================================================
        importance_found = any(marker in text_lower for marker in self.importance_markers)
        if importance_found:
            contribution = self.weights['user_importance']
            salience += contribution
            factors.append("user_marked_important")
            factor_contributions['user_importance'] = contribution
        
        # =====================================================================
        # FACTOR 6: Breakthrough Indicators
        # =====================================================================
        breakthrough_found = any(marker in text_lower for marker in self.breakthrough_markers)
        if breakthrough_found:
            contribution = self. weights['breakthrough']
            salience += contribution
            factors.append("therapeutic_breakthrough")
            factor_contributions['breakthrough'] = contribution
        
        # =====================================================================
        # FACTOR 7: Trigger Entities
        # =====================================================================
        if trigger_entities and len(trigger_entities) > 0:
            contribution = self.weights['trigger_entity'] * min(1.0, len(trigger_entities) * 0.5)
            salience += contribution
            factors.append(f"trigger_entities_{len(trigger_entities)}")
            factor_contributions['triggers'] = contribution
        
        # =====================================================================
        # FACTOR 8: Text Depth (proxy for meaningful content)
        # =====================================================================
        word_count = len(text.split()) if text else 0
        if word_count > 75:
            contribution = self.weights['text_depth']
            salience += contribution
            factors.append("detailed_sharing")
            factor_contributions['depth'] = contribution
        elif word_count > 40:
            contribution = self.weights['text_depth'] * 0.5
            salience += contribution
            factors.append("moderate_detail")
            factor_contributions['depth'] = contribution
        
        # =====================================================================
        # FACTOR 9: Specific Emotions (trauma, grief, etc.) - ENHANCED
        # =====================================================================
        high_salience_emotions = [
            'grief', 'trauma', 'betrayal', 'devastation', 'despair',
            'breakthrough', 'revelation', 'epiphany', 'joy', 'love',
            'overwhelmed', 'hopeless', 'suicidal', 'depressed'  # ADDED
        ]

        # Also check for distress emotion + high intensity combo
        distress_emotions = ['sad', 'anxious', 'afraid', 'scared', 'lonely', 'overwhelmed']

        if detected_emotion and detected_emotion.lower() in high_salience_emotions:
            contribution = 0.15
            salience += contribution
            factors.append(f"significant_emotion_{detected_emotion}")
            factor_contributions['emotion_type'] = contribution
        elif detected_emotion and detected_emotion.lower() in distress_emotions:
            # Distress emotion + high intensity = core memory
            if emotional_intensity > 0.6:
                contribution = 0.20
                salience += contribution
                factors.append(f"high_distress_{detected_emotion}")
                factor_contributions['distress_combo'] = contribution
        # =====================================================================
        # FACTOR 11: Emotional Content Keywords (NEW - for "everyone hates me" etc.)
        # =====================================================================
        distress_keywords = {
            'hate', 'hates', 'hated', 'everyone hates', 'nobody likes',
            'always', 'never', 'everyone', 'nobody', 'alone', 'reject', 'rejected',
            'bully', 'bullied', 'called me', 'they call me'
        }

        text_lower_words = set(text.lower().split())
        keyword_matches = text_lower_words & distress_keywords

        if keyword_matches:
            contribution = 0.20 * min(1.0, len(keyword_matches) * 0.5)
            salience += contribution
            factors.append(f"distress_keywords_{len(keyword_matches)}")
            factor_contributions['distress_keywords'] = contribution
            
        # =====================================================================
        # FACTOR 10: Multimodal Bonus
        # =====================================================================
        if modality in ['audio', 'video'] and biomarker_notable: 
            contribution = 0.10
            salience += contribution
            factors.append(f"notable_biomarkers_{modality}")
            factor_contributions['biomarkers'] = contribution
        
        # =====================================================================
        # Cap salience at 1.0
        # =====================================================================
        salience = min(1.0, salience)
        
        # =====================================================================
        # Determine category
        # =====================================================================
        if salience >= self.thresholds['always_retrieve']:
            category = "CORE_MEMORY"
            should_always_retrieve = True
            self.session_stats['core_memory_count'] += 1
        elif salience >= self.thresholds['high_salience']: 
            category = "IMPORTANT"
            should_always_retrieve = False
            self.session_stats['important_count'] += 1
        elif salience >= self.thresholds['moderate_salience']: 
            category = "MODERATE"
            should_always_retrieve = False
            self.session_stats['moderate_count'] += 1
        else:
            category = "ROUTINE"
            should_always_retrieve = False
            self.session_stats['routine_count'] += 1
        
        # Update stats
        self.session_stats['total_count'] += 1
        self.session_stats['total_salience'] += salience
        
        # Build result
        result = {
            "salience_score": round(salience, 4),
            "category": category,
            "should_always_retrieve": should_always_retrieve,
            "factors":  factors,
            "factor_contributions": factor_contributions,
            "turn_number": turn_number,
            "modality": modality,
            "word_count": word_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in session memories
        self.session_memories. append(result)
        
        return result
    
    def calculate_decay(
        self,
        original_salience: float,
        days_since_creation: int,
        access_count: int = 0,
        category: str = "MODERATE"
    ) -> float:
        """
        Calculate decayed salience based on time and access patterns.
        
        Core memories don't decay. 
        Important memories decay slowly.
        Routine memories decay faster.
        """
        # Core memories never decay
        if category == "CORE_MEMORY" or original_salience >= self.thresholds['always_retrieve']:
            return original_salience
        
        # Decay rates by category (per day)
        decay_rates = {
            "IMPORTANT": 0.005,
            "MODERATE": 0.015,
            "ROUTINE": 0.03,
        }
        
        decay_rate = decay_rates.get(category, 0.02)
        
        # Access bonus (frequent access slows decay)
        access_bonus = min(0.5, access_count * 0.05)
        effective_decay_rate = max(0, decay_rate - access_bonus)
        
        # Calculate decay
        decay_factor = math.exp(-effective_decay_rate * days_since_creation)
        decayed_salience = original_salience * decay_factor
        
        # Minimum floor
        min_salience = 0.1 if category == "IMPORTANT" else 0.05
        
        return max(min_salience, decayed_salience)
    
    def get_retrieval_boost(self, salience_score: float, category: str) -> float:
        """
        Get retrieval score boost based on salience. 
        
        Used to boost vector similarity scores for high-salience memories.
        """
        if category == "CORE_MEMORY":
            return 0.3  # Significant boost
        elif category == "IMPORTANT": 
            return 0.15
        elif category == "MODERATE": 
            return 0.05
        else:
            return 0.0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive salience statistics for the session."""
        if self.session_stats['total_count'] == 0:
            return {
                "count": 0,
                "avg_salience": 0.0,
                "high_salience_count": 0,
                "always_retrieve_count": 0,
                "categories": {}
            }
        
        scores = [m['salience_score'] for m in self.session_memories]
        
        return {
            "count": self.session_stats['total_count'],
            "avg_salience": round(self.session_stats['total_salience'] / self.session_stats['total_count'], 3),
            "max_salience": max(scores) if scores else 0.0,
            "min_salience":  min(scores) if scores else 0.0,
            "high_salience_count": self.session_stats['important_count'] + self.session_stats['core_memory_count'],
            "always_retrieve_count": self. session_stats['core_memory_count'],
            "categories": {
                "CORE_MEMORY": self.session_stats['core_memory_count'],
                "IMPORTANT": self. session_stats['important_count'],
                "MODERATE": self. session_stats['moderate_count'],
                "ROUTINE": self. session_stats['routine_count'],
            },
            "recent_memories": [
                {
                    "turn":  m['turn_number'],
                    "salience": m['salience_score'],
                    "category": m['category'],
                    "factors": m['factors'][: 3]
                }
                for m in self.session_memories[-5:]
            ]
        }
    
    def get_high_salience_turns(self) -> List[int]:
        """Get turn numbers of high-salience memories."""
        return [
            m['turn_number']
            for m in self.session_memories
            if m['category'] in ['CORE_MEMORY', 'IMPORTANT']
        ]
    
    def reset_session(self):
        """Reset session tracking for new session."""
        self.session_memories = []
        self.session_stats = {
            'total_count': 0,
            'core_memory_count': 0,
            'important_count': 0,
            'moderate_count': 0,
            'routine_count': 0,
            'total_salience':  0.0,
        }