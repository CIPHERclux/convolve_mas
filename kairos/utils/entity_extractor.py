"""
Entity Extractor
Extracts and normalizes entities from text with alias mapping
"""
import re
from typing import List, Dict, Set

try:
    from config import ENTITY_ALIASES
except ImportError:
    ENTITY_ALIASES = {
        'mom': 'mother', 'mum': 'mother', 'mommy': 'mother', 'mama': 'mother',
        'dad': 'father', 'daddy': 'father', 'papa': 'father',
        'bro': 'brother', 'sis': 'sister',
        'bf': 'boyfriend', 'gf': 'girlfriend', 'bestie': 'best_friend',
        'boss': 'manager', 'supervisor': 'manager',
    }


class EntityExtractor:
    """
    Extracts entities from text and normalizes aliases. 
    Implements Entity Drift Normalization from the architecture. 
    """
    
    def __init__(self):
        self.alias_map = ENTITY_ALIASES
        
        # Common relationship terms
        self.relationship_patterns = [
            r'\b(mother|mom|mum|mommy|mama)\b',
            r'\b(father|dad|daddy|papa)\b',
            r'\b(brother|bro)\b',
            r'\b(sister|sis)\b',
            r'\b(husband|wife|spouse|partner)\b',
            r'\b(boyfriend|girlfriend|bf|gf)\b',
            r'\b(friend|best friend|bestie)\b',
            r'\b(boss|manager|supervisor)\b',
            r'\b(coworker|colleague)\b',
            r'\b(therapist|counselor|doctor|psychiatrist)\b',
            r'\b(child|son|daughter|kid)\b',
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.relationship_patterns]
    
    def extract(self, text: str) -> List[str]:
        """
        Extract and normalize entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of normalized entity strings
        """
        if not text:
            return []
        
        entities = set()
        text_lower = text.lower()
        
        # Extract relationship entities
        for pattern in self. compiled_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                # Normalize using alias map
                normalized = self._normalize_entity(match. lower())
                entities.add(normalized)
        
        # Extract proper nouns (capitalized words that aren't sentence starters)
        words = text. split()
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and word and word[0].isupper():
                clean_word = re.sub(r'[^\w]', '', word)
                if len(clean_word) > 1:
                    entities.add(clean_word. lower())
        
        return list(entities)
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity using alias map."""
        return self.alias_map.get(entity, entity)
    
    def add_alias(self, alias: str, canonical: str):
        """Add a new alias mapping."""
        self.alias_map[alias. lower()] = canonical.lower()
    
    def get_canonical(self, entity: str) -> str:
        """Get canonical form of an entity."""
        return self.alias_map.get(entity.lower(), entity.lower())