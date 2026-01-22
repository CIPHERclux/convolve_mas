"""
Sparse Encoder for Hybrid Search
Implements BM25/SPLADE-style sparse vectors for keyword matching
Critical for crisis term detection that semantic search might miss
"""
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import math


class SparseEncoder:
    """
    Sparse vector encoder for hybrid search. 
    
    Ensures exact keyword matches for critical safety terms
    that might be "smoothed out" by semantic embeddings. 
    """
    
    def __init__(self):
        # Crisis terms with weights (higher = more critical)
        self.crisis_terms = {
            # Direct suicide terms (weight 1.0)
            'kill myself': 1.0,
            'kill me': 1.0,
            'end my life': 1.0,
            'take my life': 1.0,
            'want to die': 1.0,
            'wanna die': 1.0,
            'suicide': 1.0,
            'suicidal': 1.0,
            'unalive': 1.0,
            'kms': 1.0,
            'kys': 1.0,
            
            # Methods (weight 0.95)
            'overdose': 0.95,
            'hang myself': 0.95,
            'shoot myself': 0.95,
            'jump off':  0.95,
            'slit wrist': 0.95,
            'cut myself': 0.9,
            'hurt myself': 0.9,
            
            # Hopelessness (weight 0.85)
            'no point': 0.85,
            'no reason to live': 0.9,
            'better off dead': 0.95,
            'world without me': 0.9,
            'burden':  0.8,
            'worthless': 0.8,
            'hopeless': 0.85,
            'no hope': 0.85,
            'give up': 0.75,
            'cant go on': 0.85,
            'cant take it': 0.8,
            
            # Intent signals (weight 0.8)
            'goodbye forever': 0.9,
            'final goodbye': 0.9,
            'end it all': 0.9,
            'not be here':  0.8,
            'disappear': 0.7,
            'sleep forever': 0.85,
            'never wake up': 0.85,
        }
        
        # Build vocabulary index
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = defaultdict(lambda: 1.0)
        self._build_vocab()
        
        # Compile patterns for efficient matching
        self.compiled_patterns: List[Tuple[re.Pattern, str, float]] = []
        self._compile_patterns()
    
    def _build_vocab(self):
        """Build vocabulary from crisis terms."""
        idx = 0
        for term in self.crisis_terms:
            words = term.lower().split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
            # Also add the full phrase
            phrase_key = term.lower().replace(' ', '_')
            if phrase_key not in self.vocab:
                self.vocab[phrase_key] = idx
                idx += 1
    
    def _compile_patterns(self):
        """Compile regex patterns for crisis terms."""
        for term, weight in self.crisis_terms. items():
            # Escape special regex characters and create word boundary pattern
            escaped = re.escape(term)
            # Replace escaped spaces with flexible whitespace
            pattern_str = escaped.replace(r'\ ', r'\s+')
            try:
                pattern = re.compile(r'\b' + pattern_str + r'\b', re.IGNORECASE)
                self.compiled_patterns.append((pattern, term, weight))
            except re.error as e:
                print(f"  [SparseEncoder] Warning: Invalid pattern '{term}': {e}")
    
    def encode(self, text: str) -> Dict[int, float]:
        """
        Encode text into sparse vector. 
        
        Returns:
            Dict mapping vocabulary indices to weights
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        sparse_vector = {}
        
        # Check for crisis term matches
        for pattern, term, weight in self.compiled_patterns:
            if pattern.search(text_lower):
                phrase_key = term.lower().replace(' ', '_')
                if phrase_key in self.vocab:
                    idx = self.vocab[phrase_key]
                    sparse_vector[idx] = max(sparse_vector.get(idx, 0), weight)
        
        # Also encode individual words with TF-IDF style weights
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                # TF component (log-scaled)
                tf = 1 + math.log(count) if count > 0 else 0
                # IDF component
                idf = self.idf. get(word, 1.0)
                weight = tf * idf
                sparse_vector[idx] = max(sparse_vector.get(idx, 0), weight * 0.5)
        
        return sparse_vector
    
    def encode_for_qdrant(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Encode text into Qdrant sparse vector format.
        
        Returns:
            Tuple of (indices, values)
        """
        sparse_dict = self.encode(text)
        
        if not sparse_dict:
            return [], []
        
        indices = list(sparse_dict.keys())
        values = list(sparse_dict.values())
        
        return indices, values
    
    def get_crisis_severity(self, text: str) -> float:
        """
        Get overall crisis severity from text.
        
        Returns:
            Float 0-1 indicating crisis severity
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        max_severity = 0.0
        
        for pattern, term, weight in self. compiled_patterns:
            if pattern.search(text_lower):
                max_severity = max(max_severity, weight)
        
        return max_severity
    
    def get_crisis_terms_found(self, text: str) -> List[Tuple[str, float]]: 
        """
        Get list of crisis terms found in text.
        
        Returns:
            List of (term, weight) tuples
        """
        if not text: 
            return []
        
        text_lower = text.lower()
        found = []
        
        for pattern, term, weight in self.compiled_patterns:
            if pattern.search(text_lower):
                found.append((term, weight))
        
        # Sort by weight descending
        found.sort(key=lambda x: x[1], reverse=True)
        return found
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_stats(self) -> Dict:
        """Get encoder statistics."""
        return {
            "vocab_size": len(self.vocab),
            "crisis_terms_count": len(self.crisis_terms),
            "compiled_patterns": len(self.compiled_patterns)
        }