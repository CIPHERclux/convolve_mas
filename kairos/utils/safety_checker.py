"""
Safety Checker - CRITICAL SAFETY MODULE
Evaluates text for crisis indicators and safety concerns
Integrates with sparse encoder and entity graph for comprehensive detection
"""
import re
from typing import List, Tuple, Dict, Optional, Any


class SafetyChecker: 
    """
    Evaluates user input for crisis indicators. 
    
    CRITICAL:  This is a safety-critical module. 
    False negatives (missing a crisis) are MUCH worse than false positives.
    
    Integrates with: 
    - Sparse encoder for exact keyword matching
    - Entity graph for trigger detection
    """
    
    def __init__(self, sparse_encoder=None, entity_graph=None):
        self.sparse_encoder = sparse_encoder
        self.entity_graph = entity_graph
        
        # Base threshold (can be lowered if triggers detected)
        self.base_threshold = 0.7
        self.current_threshold = self.base_threshold
        
        # Crisis patterns with severity weights
        self.crisis_patterns = [
            # CRITICAL - Direct suicide/self-harm (weight 0.9-1.0)
            (r'\b(kill myself|kill me|end my life|take my life)\b', 1.0),
            (r'\b(want to die|wanna die|ready to die)\b', 1.0),
            (r'\bsuicid(e|al|ing)\b', 1.0),
            (r'\b(should ? n[o\']? t be alive)\b', 1.0),
            (r'\b(don[\'t]?  ?want to live)\b', 1.0),
            (r'\b(better off dead)\b', 1.0),
            (r'\b(end it all|ending it all)\b', 0.95),
            (r'\b(no point in living|no reason to live)\b', 0.95),
            (r'\b(off myself)\b', 0.95),
            
            # EUPHEMISMS
            (r'\b(unalive|un-alive)\s*(myself|me)?\b', 1.0),
            (r'\b(kms|kys)\b', 1.0),
            (r'\b(ctb|catch the bus)\b', 0.95),
            (r'\b(not wake up|never wake up|sleep forever)\b', 0.9),
            (r'\b(disappear forever|cease to exist)\b', 0.85),
            
            # SELF-HARM
            (r'\b(cut myself|cutting myself|hurt myself)\b', 0.9),
            (r'\b(burn myself|starve myself)\b', 0.9),
            
            # METHODS (high severity)
            (r'\b(jump off|hang myself|shoot myself|overdose)\b', 1.0),
            (r'\b(slit|slitting)\s*(my\s*)?(wrist|throat)\b', 1.0),
            
            # HOPELESSNESS + INTENT
            (r'\b(nothing to live for|no future)\b', 0.9),
            (r'\b(burden to everyone)\b', 0.9),
            (r'\b(goodbye|final goodbye).*(forever|world)\b', 0.9),
            (r'\b(world|everyone).*(better|fine).*(without me)\b', 0.9),
            
            # HIGH - Severe distress (weight 0.7-0.85)
            (r'\b(hopeless|no hope|lost all hope)\b', 0.8),
            (r'\b(can[\'t]?  go on|can[\'t]? do this anymore)\b', 0.75),
            (r'\b(can[\'t]? take it anymore)\b', 0.75),
            (r'\b(give up|giving up) on (life|everything|myself)\b', 0.8),
            (r'\b(worthless|completely worthless)\b', 0.7),
            (r'\b(everyone hates me|nobody cares)\b', 0.7),
            (r'\b(no one would miss)\b', 0.8),
            (r'\b(tired of living|tired of life)\b', 0.75),
            
            # MODERATE - Notable distress (weight 0.5-0.65)
            (r'\b(don[\'t]? want to be here)\b', 0.65),
            (r'\b(wish i was dead|wish i were dead)\b', 0.85),
            (r'\b(hate my life|hate myself)\b', 0.6),
        ]
        
        # Compile patterns
        self.compiled_patterns = []
        for pattern, weight in self.crisis_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.compiled_patterns.append((compiled, pattern, weight))
            except re.error as e:
                print(f"  [SafetyChecker] Warning: Invalid pattern '{pattern}': {e}")
        
        # Protective factors (slight reduction)
        self.protective_patterns = [
            r'\b(but i won[\'t]? |but i wouldn[\'t]?)\b',
            r'\b(joking|just kidding|jk)\b',
            r'\b(in the past|used to|years ago)\b',
            r'\b(hypothetically|hypothetical)\b',
            r'\b(movie|book|song|character|story|game)\b',
        ]
        
        self.compiled_protective = []
        for pattern in self.protective_patterns:
            try:
                self.compiled_protective.append(re.compile(pattern, re. IGNORECASE))
            except re.error:
                pass
        
        # Statistics tracking
        self.total_evaluations = 0
        self. crisis_count = 0
        self.recent_scores = []
    
    def evaluate(self, text: str, conversation_context: List[str] = None) -> float:
        """
        Evaluate text for crisis indicators.
        
        Returns:
            Crisis score (0.0 to 1.0)
        """
        result = self.evaluate_with_details(text, conversation_context)
        return result['score']
    
    def evaluate_with_details(
        self,
        text: str,
        conversation_context: List[str] = None,
        entities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate text with detailed results.
        
        Returns:
            Dict with score, matched patterns, and analysis
        """
        self.total_evaluations += 1
        
        if not text:
            return {
                'score': 0.0,
                'is_crisis': False,
                'matched_patterns': [],
                'sparse_terms': [],
                'threshold':  self.current_threshold
            }
        
        text_lower = text.lower()
        matched_patterns = []
        max_score = 0.0
        
        # Check regex patterns
        for compiled, pattern, weight in self.compiled_patterns:
            match = compiled.search(text_lower)
            if match:
                matched_patterns.append({
                    'pattern': pattern,
                    'match': match. group(),
                    'weight':  weight
                })
                max_score = max(max_score, weight)
        
        # Check sparse encoder for additional terms
        sparse_terms = []
        if self.sparse_encoder:
            sparse_severity = self.sparse_encoder.get_crisis_severity(text)
            sparse_found = self.sparse_encoder.get_crisis_terms_found(text)
            
            for term, weight in sparse_found:
                sparse_terms.append({'term': term, 'weight':  weight})
                max_score = max(max_score, weight)
        
        # Check entity graph for triggers
        trigger_boost = 0.0
        trigger_entities = []
        if self.entity_graph and entities:
            trigger_alerts = self.entity_graph.check_trigger_entities(entities)
            if trigger_alerts:
                for alert in trigger_alerts:
                    trigger_entities.append(alert['entity'])
                    trigger_boost = max(trigger_boost, alert['trigger_score'] * 0.2)
                # Lower threshold if triggers present
                self.current_threshold = max(0.5, self.base_threshold - trigger_boost)
        else:
            self.current_threshold = self.base_threshold
        
        # Check conversation context for escalation
        escalation_detected = False
        if conversation_context and len(conversation_context) >= 2:
            prev_scores = []
            for prev_text in conversation_context[-3:]:
                prev_score = self._quick_score(prev_text)
                prev_scores.append(prev_score)
            
            if len(prev_scores) >= 2:
                if all(s > 0.3 for s in prev_scores) and max_score > prev_scores[-1]:
                    escalation_detected = True
                    max_score = min(1.0, max_score + 0.1)
        
        # Check for protective factors
        protective_count = sum(
            1 for p in self.compiled_protective
            if p.search(text_lower)
        )
        
        # Apply protective reduction (but never below 0.5 for serious patterns)
        reduction = min(0.1, protective_count * 0.03)
        if max_score >= 0.7: 
            final_score = max(0.5, max_score - reduction)
        else:
            final_score = max(0.0, max_score - reduction)
        
        # Add trigger boost
        final_score = min(1.0, final_score + trigger_boost)
        
        # Determine if crisis
        is_crisis = final_score >= self.current_threshold
        
        if is_crisis:
            self. crisis_count += 1
        
        # Track recent scores
        self.recent_scores.append(final_score)
        self.recent_scores = self.recent_scores[-10:]
        
        return {
            'score': final_score,
            'is_crisis': is_crisis,
            'matched_patterns': matched_patterns,
            'sparse_terms': sparse_terms,
            'trigger_entities': trigger_entities,
            'trigger_boost': trigger_boost,
            'escalation_detected': escalation_detected,
            'protective_factors': protective_count,
            'threshold': self.current_threshold
        }
    
    def _quick_score(self, text:  str) -> float:
        """Quick crisis score without full analysis."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        max_score = 0.0
        
        for compiled, pattern, weight in self.compiled_patterns:
            if compiled.search(text_lower):
                max_score = max(max_score, weight)
        
        return max_score
    
    def get_crisis_response_elements(self, score: float) -> Dict[str, Any]:
        """Get crisis response elements based on severity."""
        if score >= 0.85:
            return {
                "severity": "CRITICAL",
                "requires_resources": True,
                "tone": "immediate care, direct, warm",
                "must_include":  [
                    "Express immediate care",
                    "Take words seriously",
                    "Ask about safety",
                    "988 Lifeline",
                    "Crisis Text Line:  HOME to 741741"
                ],
                "must_avoid": [
                    "Dismissing feelings",
                    "Generic responses",
                    "Long responses"
                ]
            }
        elif score >= 0.7:
            return {
                "severity": "HIGH",
                "requires_resources": True,
                "tone": "deeply caring, present",
                "must_include": [
                    "Validate pain",
                    "Express concern",
                    "Mention support available"
                ],
                "must_avoid": [
                    "Minimizing",
                    "Toxic positivity"
                ]
            }
        elif score >= 0.5:
            return {
                "severity": "MODERATE",
                "requires_resources": False,
                "tone": "warm, attentive",
                "must_include": [
                    "Acknowledge difficulty"
                ],
                "must_avoid": [
                    "Being dismissive"
                ]
            }
        else:
            return {
                "severity": "LOW",
                "requires_resources":  False,
                "tone": "warm, engaged",
                "must_include":  [],
                "must_avoid": []
            }
    
    def needs_immediate_intervention(self, text: str, context: List[str] = None) -> bool:
        """Check if immediate crisis intervention is needed."""
        result = self.evaluate_with_details(text, context)
        return result['is_crisis']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety checker statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "crisis_count": self.crisis_count,
            "crisis_rate": self.crisis_count / max(1, self.total_evaluations),
            "current_threshold": self.current_threshold,
            "base_threshold": self.base_threshold,
            "recent_scores":  self.recent_scores[-5: ],
            "patterns_count": len(self.compiled_patterns)
        }
    
    def reset_threshold(self):
        """Reset threshold to base value."""
        self.current_threshold = self.base_threshold