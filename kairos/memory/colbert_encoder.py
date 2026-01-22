"""
ColBERT Encoder - Phase 3: Late Interaction / Multivector Support

Implements token-level embeddings for preserving nuance in user text. 
Detects contradictions like "I love him" vs "I'm scared of him".
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class ColBERTEncoder:  
    """
    Late Interaction encoder for token-level vectors. 
    
    Unlike single-vector encoding which compresses an entire paragraph
    into one 384-dim vector, ColBERT preserves per-token embeddings. 
    
    This enables:
    - Detection of contradictions within text
    - Fine-grained semantic matching
    - Better handling of complex emotional statements
    """
    
    def __init__(self, max_tokens: int = 64, token_dim: int = 384):
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            print("  [ColBERTEncoder] Loading model...")
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Get tokenizer from model
            self._tokenizer = self._model.tokenizer
            
            self._initialized = True
            print("  [ColBERTEncoder] Model loaded successfully")
            
        except Exception as e:
            print(f"  [ColBERTEncoder] Failed to load model: {e}")
            self._initialized = True  # Prevent repeated attempts
    
    def encode(self, text: str) -> List[List[float]]:
        """
        Encode text to multi-vector representation (one vector per token).
        
        Args:
            text: Input text
        
        Returns:
            List of token vectors, each of dimension token_dim
        """
        self._load_model()
        
        if self._model is None or not text or not text.strip():
            return []
        
        try:
            import torch
            
            # Tokenize
            encoded = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors='pt'
            )
            
            # Get token embeddings from model
            with torch.no_grad():
                model_output = self._model._first_module().auto_model(
                    **encoded
                )
                
                # Get last hidden state (token embeddings)
                token_embeddings = model_output. last_hidden_state[0]  # [seq_len, dim]
                
                # Get attention mask to filter padding
                attention_mask = encoded['attention_mask'][0]
                
                # Filter out padding tokens
                valid_embeddings = []
                for i, mask in enumerate(attention_mask):
                    if mask == 1:  # Valid token
                        valid_embeddings.append(
                            token_embeddings[i]. cpu().numpy().tolist()
                        )
                
                # Limit to max_tokens
                if len(valid_embeddings) > self.max_tokens:
                    valid_embeddings = valid_embeddings[:self.max_tokens]
                
                return valid_embeddings
                
        except Exception as e:
            print(f"  [ColBERTEncoder] Encoding failed: {e}")
            # Fallback:  return single sentence embedding as one "token"
            try:
                embedding = self._model.encode(text, convert_to_numpy=True)
                return [embedding. tolist()]
            except:
                return []
    
    def encode_query(self, text: str) -> List[List[float]]: 
        """
        Encode query text for late interaction matching.
        Same as encode but can be extended for query-specific processing.
        """
        return self.encode(text)
    
    def compute_similarity(
        self,
        query_vectors: List[List[float]],
        doc_vectors: List[List[float]]
    ) -> float:
        """
        Compute MaxSim similarity between query and document token vectors.
        
        This is the core ColBERT late interaction mechanism: 
        For each query token, find the maximum similarity to any document token,
        then sum these maximum similarities.
        
        Args:
            query_vectors: List of query token vectors
            doc_vectors:  List of document token vectors
        
        Returns:
            Similarity score (higher = more similar)
        """
        if not query_vectors or not doc_vectors:
            return 0.0
        
        query_arr = np.array(query_vectors)
        doc_arr = np.array(doc_vectors)
        
        # Normalize vectors
        query_norm = query_arr / (np.linalg.norm(query_arr, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_arr / (np.linalg.norm(doc_arr, axis=1, keepdims=True) + 1e-8)
        
        # Compute all pairwise similarities
        similarity_matrix = np.dot(query_norm, doc_norm.T)  # [query_len, doc_len]
        
        # MaxSim:  for each query token, take max similarity across all doc tokens
        max_sims = np.max(similarity_matrix, axis=1)  # [query_len]
        
        # Sum of max similarities
        total_similarity = np.sum(max_sims)
        
        # Normalize by query length
        normalized_similarity = total_similarity / len(query_vectors)
        
        return float(normalized_similarity)
    
    def detect_contradictions(self, text:  str) -> List[Dict[str, Any]]:
        """
        Detect potential contradictions within text.
        
        Uses token-level analysis to find opposing sentiment patterns.
        
        Args:
            text: Input text
        
        Returns:
            List of detected contradictions with evidence
        """
        self._load_model()
        
        if self._model is None or not text:
            return []
        
        contradictions = []
        
        # Split into clauses/sentences
        import re
        clauses = re.split(r'[.,;!?]+\s*|\s+but\s+|\s+however\s+|\s+although\s+', text. lower())
        clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 3]
        
        if len(clauses) < 2:
            return []
        
        # Define contradiction patterns
        positive_words = {'love', 'like', 'happy', 'good', 'great', 'wonderful', 'safe', 'trust', 'care'}
        negative_words = {'hate', 'fear', 'scared', 'afraid', 'hurt', 'angry', 'sad', 'bad', 'dangerous', 'distrust'}
        
        # Check for emotional contradictions about same subject
        for i, clause1 in enumerate(clauses):
            words1 = set(clause1.split())
            has_positive1 = bool(words1 & positive_words)
            has_negative1 = bool(words1 & negative_words)
            
            for j, clause2 in enumerate(clauses[i+1:], start=i+1):
                words2 = set(clause2.split())
                has_positive2 = bool(words2 & positive_words)
                has_negative2 = bool(words2 & negative_words)
                
                # Check for opposing sentiments
                if (has_positive1 and has_negative2) or (has_negative1 and has_positive2):
                    # Check for shared subjects (pronouns or nouns)
                    subjects = {'he', 'she', 'him', 'her', 'they', 'them', 'it', 'i', 'me', 'my'}
                    shared_subjects = (words1 & subjects) & (words2 & subjects)
                    
                    # Also check for repeated content words
                    content_words1 = {w for w in words1 if len(w) > 3 and w not in subjects}
                    content_words2 = {w for w in words2 if len(w) > 3 and w not in subjects}
                    shared_content = content_words1 & content_words2
                    
                    if shared_subjects or shared_content:
                        contradictions.append({
                            "clause1": clause1,
                            "clause2": clause2,
                            "type": "emotional_contradiction",
                            "shared_reference": list(shared_subjects | shared_content),
                            "sentiment1": "positive" if has_positive1 else "negative",
                            "sentiment2": "positive" if has_positive2 else "negative"
                        })
        
        return contradictions
    
    def get_token_importance(self, text: str) -> List[Tuple[str, float]]: 
        """
        Get importance scores for each token based on embedding magnitude.
        
        Useful for understanding which words carry the most semantic weight.
        
        Args:
            text: Input text
        
        Returns:
            List of (token, importance_score) tuples
        """
        self._load_model()
        
        if self._model is None or not text:
            return []
        
        try:
            # Get tokens
            tokens = self._tokenizer. tokenize(text)[:self.max_tokens]
            
            # Get embeddings
            embeddings = self. encode(text)
            
            if len(tokens) != len(embeddings):
                # Alignment issue, return empty
                return []
            
            # Calculate importance as L2 norm of each embedding
            importance_scores = []
            for i, (token, embedding) in enumerate(zip(tokens, embeddings)):
                norm = np.linalg.norm(embedding)
                importance_scores.append((token, float(norm)))
            
            # Sort by importance
            importance_scores.sort(key=lambda x: x[1], reverse=True)
            
            return importance_scores
            
        except Exception as e:
            print(f"  [ColBERTEncoder] Token importance failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
            "max_tokens": self.max_tokens,
            "token_dim":  self.token_dim
        }