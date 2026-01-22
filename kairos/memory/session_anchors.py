"""
Session Anchors
Stores the first few turns of each session for O(1) retrieval.

The "reason for the visit" - what brought someone to therapy - is CRITICAL context
that should ALWAYS be available, regardless of what topics come up later.  

Session anchors are retrieved by exact ID, not vector similarity, ensuring
they're always in context.  
"""
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib


class SessionAnchors:
    """
    Manages session anchor points - the first N interactions
    that establish context for the session.
    
    Features:
    - Deterministic ID generation for O(1) lookup
    - Automatic capture of first N turns
    - Rich metadata for context
    - Session-start summarization
    
    These are retrieved via exact ID lookup, not vector search,
    ensuring they're always available regardless of semantic drift.
    """
    
    def __init__(self, session_id: str, max_anchors: int = 3):
        """
        Initialize session anchors. 
        
        Args:
            session_id: Unique session identifier
            max_anchors:   Maximum number of turns to anchor (default 3)
        """
        self.session_id = session_id
        self.max_anchors = max_anchors
        
        # Anchor storage
        self.anchors: List[Dict[str, Any]] = []
        self.anchor_ids: List[str] = []
        
        # Map from deterministic anchor key to UUID
        self.anchor_key_to_uuid: Dict[str, str] = {}
        
        # Lock flag - once we have max_anchors, stop adding
        self.anchors_locked = False
        
        # Session metadata
        self.session_start_time = datetime.now()
        self.primary_concern:   Optional[str] = None  # Extracted from first turn
        self.initial_emotion: Optional[str] = None
        self.initial_crisis:   bool = False
        
        # Summary of session start (generated after anchors locked)
        self.session_start_summary:   Optional[str] = None
    
    def should_be_anchor(self, turn_number: int) -> bool:
        """
        Check if this turn should be anchored.
        
        Args:
            turn_number: Current turn number (0-indexed)
            
        Returns: 
            True if this turn should be an anchor
        """
        if self.anchors_locked:
            return False
        return turn_number < self.max_anchors
    
    def generate_anchor_id(self, turn_number: int) -> str:
        """
        Generate a valid UUID for anchor storage.
        
        We use UUID5 with a namespace derived from session_id and turn_number
        to create deterministic but valid UUIDs.
        """
        # Create a deterministic key
        anchor_key = f"anchor_{self.session_id}_{turn_number}"
        
        # Check if we already have a UUID for this key
        if anchor_key in self.anchor_key_to_uuid:
            return self.anchor_key_to_uuid[anchor_key]
        
        # Generate a deterministic UUID using UUID5
        # Use a namespace UUID (we'll use the DNS namespace as a base)
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace DNS
        
        # Create UUID5 from namespace and anchor key
        anchor_uuid = str(uuid.uuid5(namespace, anchor_key))
        
        # Store the mapping
        self.anchor_key_to_uuid[anchor_key] = anchor_uuid
        
        return anchor_uuid
    
    def get_anchor_key(self, turn_number: int) -> str:
        """Get the deterministic anchor key (for display/debugging)."""
        return f"anchor_{self.session_id}_{turn_number}"
    
    def register_anchor(
        self,
        turn_number: int,
        user_text: str,
        system_response: str,
        emotion: str,
        emotional_intensity: float,
        is_crisis: bool,
        crisis_score: float,
        salience_score: float,
        entities: List[str] = None,
        modality:   str = "text",
        biomarker_summary: str = None
    ) -> Optional[str]:
        """
        Register a new anchor point.
        
        Args:
            turn_number:  Turn number (0-indexed)
            user_text: User's message
            system_response: System's response
            emotion:   Detected emotion
            emotional_intensity:   Intensity score
            is_crisis: Crisis flag
            crisis_score: Crisis score
            salience_score:   Salience score
            entities:   Mentioned entities
            modality: Input modality
            biomarker_summary: Summary of biomarker signals
            
        Returns:
            Anchor ID (UUID) if registered, None if anchors locked
        """
        if self.anchors_locked:
            return None
        
        if turn_number >= self.max_anchors:
            self.anchors_locked = True
            self._generate_session_summary()
            return None
        
        anchor_id = self.generate_anchor_id(turn_number)
        anchor_key = self.get_anchor_key(turn_number)
        
        anchor_data = {
            "id": anchor_id,
            "key": anchor_key,
            "turn_number": turn_number,
            "user_text": user_text,
            "system_response": system_response,
            "emotion": emotion,
            "emotional_intensity": emotional_intensity,
            "is_crisis": is_crisis,
            "crisis_score": crisis_score,
            "salience_score": salience_score,
            "entities": entities or [],
            "modality": modality,
            "biomarker_summary": biomarker_summary,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(user_text.split()) if user_text else 0,
        }
        
        self.anchors.append(anchor_data)
        self.anchor_ids. append(anchor_id)
        
        # Track initial state from first anchor
        if turn_number == 0:
            self.initial_emotion = emotion
            self.initial_crisis = is_crisis
            self.primary_concern = self._extract_primary_concern(user_text)
        
        # Lock if we've reached max
        if len(self.anchors) >= self.max_anchors:
            self.anchors_locked = True
            self._generate_session_summary()
        
        return anchor_id
    
    def _extract_primary_concern(self, text: str) -> Optional[str]:
        """
        Extract the primary concern from first message.
        
        This is a simplified extraction - in production, this could
        use the LLM for better summarization.
        """
        if not text:
            return None
        
        # Take first 100 characters or first sentence
        text = text. strip()
        
        # Find first sentence
        for end_char in ['.', '!', '?']:
            idx = text.find(end_char)
            if 0 < idx < 150:  
                return text[:idx + 1]
        
        # Otherwise truncate
        if len(text) > 100:
            return text[:100] + "..."
        
        return text
    
    def _generate_session_summary(self):
        """Generate summary of session start from anchors."""
        if not self.anchors:
            return
        
        parts = []
        
        # Initial state
        if self.initial_crisis:
            parts.append("⚠️ Session began with crisis indicators")
        
        if self.primary_concern:
            parts.append(f"Primary concern: \"{self.primary_concern}\"")
        
        if self.initial_emotion:
            parts.append(f"Initial emotion: {self.initial_emotion}")
        
        # Entities mentioned early
        all_entities = set()
        for anchor in self.anchors:
            all_entities.update(anchor. get('entities', []))
        
        if all_entities:
            parts.append(f"Key entities: {', '.join(list(all_entities)[:5])}")
        
        # Modalities used
        modalities = set(a['modality'] for a in self. anchors)
        if 'audio' in modalities or 'video' in modalities:  
            parts.append(f"Modalities:  {', '.join(modalities)}")
        
        self.session_start_summary = " | ".join(parts) if parts else "Session started normally"
    
    def get_anchor_ids(self) -> List[str]:
        """Get all anchor IDs (UUIDs) for direct retrieval."""
        return self. anchor_ids. copy()
    
    def get_anchor_by_id(self, anchor_id: str) -> Optional[Dict[str, Any]]:  
        """Get specific anchor by ID."""
        for anchor in self.anchors:
            if anchor['id'] == anchor_id:
                return anchor
        return None
    
    def get_anchor_by_turn(self, turn_number: int) -> Optional[Dict[str, Any]]:  
        """Get anchor by turn number."""
        for anchor in self.anchors:
            if anchor['turn_number'] == turn_number:  
                return anchor
        return None
    
    def get_all_anchors(self) -> List[Dict[str, Any]]:  
        """Get all anchor data."""
        return self.anchors.copy()
    
    def get_anchor_context_for_llm(self) -> str:
        """
        Get formatted anchor context for LLM.
        
        Returns a structured summary of how the session started.
        """
        if not self.anchors:
            return ""
        
        lines = ["=== SESSION START (Anchors) ==="]
        
        if self.session_start_summary:
            lines.append(self.session_start_summary)
            lines.append("")
        
        for anchor in self.anchors:
            turn = anchor['turn_number'] + 1
            crisis_marker = "🚨 " if anchor. get('is_crisis') else ""
            emotion = anchor.get('emotion', 'unknown')
            intensity = anchor.get('emotional_intensity', 0)
            
            lines.append(f"[Turn {turn}] {crisis_marker}{emotion} ({int(intensity * 100)}%)")
            
            # Truncate long text
            user_text = anchor['user_text']
            if len(user_text) > 150:
                user_text = user_text[:150] + "..."
            lines.append(f"  User: \"{user_text}\"")
            
            # Show entities if present
            if anchor.get('entities'):
                lines.append(f"  Entities: {', '.join(anchor['entities'][:5])}")
            
            # Show biomarker summary for audio/video
            if anchor.get('modality') != 'text' and anchor.get('biomarker_summary'):
                lines.append(f"  Biomarkers: {anchor['biomarker_summary'][: 80]}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_anchor_texts(self) -> List[str]:
        """Get just the user texts from anchors."""
        return [a['user_text'] for a in self.anchors]
    
    def had_early_crisis(self) -> bool:
        """Check if there was a crisis in early turns."""
        return any(a.get('is_crisis', False) for a in self.anchors)
    
    def get_primary_concern(self) -> Optional[str]:
        """Get the primary concern identified at session start."""
        return self.primary_concern
    
    def get_stats(self) -> Dict[str, Any]:
        """Get anchor statistics."""
        return {
            "session_id": self.session_id,
            "anchor_count": len(self.anchors),
            "max_anchors": self.max_anchors,
            "anchors_locked": self.anchors_locked,
            "anchor_ids": self.anchor_ids,
            "primary_concern": self.primary_concern,
            "initial_emotion":  self.initial_emotion,
            "initial_crisis": self.initial_crisis,
            "session_start_summary": self.session_start_summary,
            "session_start_time": self.session_start_time. isoformat(),
            "anchors":  [
                {
                    "turn":  a['turn_number'],
                    "user_text_preview": a['user_text'][:50] + "..." if len(a['user_text']) > 50 else a['user_text'],
                    "emotion": a. get('emotion', 'unknown'),
                    "is_crisis": a.get('is_crisis', False),
                    "salience": a.get('salience_score', 0),
                    "entities": a.get('entities', [])[: 3]
                }
                for a in self.anchors
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize anchors to dict for storage."""
        return {
            "session_id": self.session_id,
            "max_anchors": self.max_anchors,
            "anchors": self.anchors,
            "anchor_ids": self.anchor_ids,
            "anchor_key_to_uuid": self.anchor_key_to_uuid,
            "anchors_locked": self.anchors_locked,
            "primary_concern": self.primary_concern,
            "initial_emotion": self.initial_emotion,
            "initial_crisis": self.initial_crisis,
            "session_start_summary": self.session_start_summary,
            "session_start_time": self.session_start_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionAnchors':
        """Deserialize anchors from dict."""
        instance = cls(
            session_id=data['session_id'],
            max_anchors=data. get('max_anchors', 3)
        )
        instance.anchors = data. get('anchors', [])
        instance.anchor_ids = data.get('anchor_ids', [])
        instance.anchor_key_to_uuid = data.get('anchor_key_to_uuid', {})
        instance.anchors_locked = data.get('anchors_locked', False)
        instance.primary_concern = data.get('primary_concern')
        instance.initial_emotion = data.get('initial_emotion')
        instance.initial_crisis = data.get('initial_crisis', False)
        instance.session_start_summary = data.get('session_start_summary')
        
        if data.get('session_start_time'):
            instance.session_start_time = datetime.fromisoformat(data['session_start_time'])
        
        return instance