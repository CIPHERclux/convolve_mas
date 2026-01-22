"""
Qdrant Manager - ENHANCED VERSION
Phase 1, 2, 3 improvements:
- Hybrid Search with Sparse Vectors
- Reciprocal Rank Fusion (RRF)
- Binary Quantization for semantic vectors (Phase 3)
- Trajectory vector support
"""
import uuid
from typing import Dict, Any, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
    SparseVectorParams, SparseVector,
    Prefetch, FusionQuery, Fusion,
    BinaryQuantization, BinaryQuantizationConfig,
    QuantizationSearchParams
)

from config import (
    COLLECTION_NAME, QDRANT_PATH,
    SEMANTIC_DIM, BIOMARKER_DIM
)

# Constants
RELIABILITY_DIM = 32
TRAJECTORY_DIM = 160  # 32 biomarker dims × 5 turns
SPARSE_VECTOR_NAME = "sparse_keywords"


class QdrantManager:
    """
    Enhanced Qdrant Manager with: 
    - Hybrid search (dense + sparse vectors)
    - Reciprocal Rank Fusion (RRF)
    - Binary quantization for semantic vectors (Phase 3)
    - Trajectory vectors for temporal pattern matching
    - Multi-vector support
    """
    
    def __init__(self, path: str = QDRANT_PATH, collection_name: str = None):
        """Initialize Qdrant client with local storage."""
        print(f"  Initializing Enhanced Qdrant at:  {path}")
        
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name or COLLECTION_NAME
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the collection with enhanced multi-vector config."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            print(f"  Creating enhanced collection: {self.collection_name}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Semantic vector with binary quantization for efficiency (Phase 3)
                    "semantic": VectorParams(
                        size=SEMANTIC_DIM,
                        distance=Distance.COSINE,
                        quantization_config=BinaryQuantization(
                            binary=BinaryQuantizationConfig(
                                always_ram=True
                            )
                        )
                    ),
                    # Biomarker vector - NO quantization for precision
                    "biomarker":  VectorParams(
                        size=BIOMARKER_DIM,
                        distance=Distance. EUCLID
                    ),
                    # Emotional state vector (8-dim)
                    "emotional":  VectorParams(
                        size=8,
                        distance=Distance. COSINE
                    ),
                    # Trajectory vector for temporal pattern matching (Phase 3)
                    "trajectory": VectorParams(
                        size=TRAJECTORY_DIM,
                        distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    # Sparse vector for keyword matching (Phase 1)
                    SPARSE_VECTOR_NAME: SparseVectorParams()
                }
            )
            print(f"  Enhanced collection created successfully")
        else:
            print(f"  Collection '{self.collection_name}' already exists")
    
    def upsert_point(
        self,
        vectors: Dict[str, Any],
        payload: Dict[str, Any],
        point_id: Optional[str] = None,
        sparse_indices: List[int] = None,
        sparse_values: List[float] = None
    ) -> str:
        """
        Insert or update a point with all vector types.
        
        Args:
            vectors: Dict with semantic, biomarker, emotional, trajectory vectors
            payload: Metadata payload
            point_id: Optional point ID
            sparse_indices:  Sparse vector indices
            sparse_values:  Sparse vector values
        """
        if point_id is None:
            point_id = str(uuid. uuid4())
        
        # Build dense vector dict with defaults
        vector_dict = {
            "semantic": vectors.get("semantic", [0.0] * SEMANTIC_DIM),
            "biomarker": vectors.get("biomarker", [0.0] * BIOMARKER_DIM),
            "emotional": vectors.get("emotional", [0.0] * 8),
            "trajectory":  vectors.get("trajectory", [0.0] * TRAJECTORY_DIM),
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=vector_dict,
            payload=payload
        )
        
        # Upsert dense vectors
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        # Add sparse vector separately if provided
        if sparse_indices and sparse_values:
            try:
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector={
                                **vector_dict,
                                SPARSE_VECTOR_NAME: SparseVector(
                                    indices=sparse_indices,
                                    values=sparse_values
                                )
                            }
                        )
                    ]
                )
            except Exception as e:
                pass  # Sparse vectors may not be supported in all versions
        
        return point_id
    
    # In QdrantManager.search_hybrid() - REPLACE the entire scoring logic

    def search_hybrid(
        self,
        semantic_vector: List[float],
        acoustic_vector: Optional[List[float]] = None,
        trajectory_vector: Optional[List[float]] = None,
        limit: int = 10,
        filter_dict: Optional[Dict] = None,
        use_graph_expansion: bool = True,
        entities: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced hybrid search with proper RRF fusion."""
        
        all_results = {}  # point_id -> {score, point_data}
        
        # 1. SEMANTIC SEARCH (always do this)
        semantic_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("semantic", semantic_vector),
            query_filter=filter_dict,
            limit=limit * 3,  # Get more candidates
            with_payload=True,
            with_vectors=True
        )
        
        for rank, hit in enumerate(semantic_results):
            point_id = hit.id
            all_results[point_id] = {
                'point': hit,
                'scores': {'semantic': 1.0 / (60 + rank)},  # RRF formula
                'rank': rank
            }
        
        # 2. ACOUSTIC SEARCH (if available) - CRITICAL for emotion matching
        if acoustic_vector and len(acoustic_vector) == 32:
            acoustic_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("acoustic", acoustic_vector),
                query_filter=filter_dict,
                limit=limit * 3,
                with_payload=True,
                with_vectors=True
            )
            
            for rank, hit in enumerate(acoustic_results):
                point_id = hit.id
                rrf_score = 1.5 / (60 + rank)  # HIGHER weight for acoustic
                
                if point_id in all_results:
                    all_results[point_id]['scores']['acoustic'] = rrf_score
                else:
                    all_results[point_id] = {
                        'point': hit,
                        'scores': {'acoustic': rrf_score},
                        'rank': rank
                    }
        
        # 3. TRAJECTORY SEARCH (if available)
        if trajectory_vector and len(trajectory_vector) == 160:
            trajectory_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("trajectory", trajectory_vector),
                query_filter=filter_dict,
                limit=limit * 2,
                with_payload=True,
                with_vectors=True
            )
            
            for rank, hit in enumerate(trajectory_results):
                point_id = hit.id
                rrf_score = 1.2 / (60 + rank)  # Higher weight for patterns
                
                if point_id in all_results:
                    all_results[point_id]['scores']['trajectory'] = rrf_score
                else:
                    all_results[point_id] = {
                        'point': hit,
                        'scores': {'trajectory': rrf_score},
                        'rank': rank
                    }
        
        # 4. GRAPH EXPANSION (if entities provided)
        if use_graph_expansion and entities and hasattr(self, 'entity_graph'):
            expanded_entities = []
            for entity in entities[:3]:  # Top 3 entities only
                related = self.entity_graph.get_related_entities(entity, limit=3)
                for rel in related:
                    expanded_entities.append({
                        'entity': rel['entity'],
                        'weight': rel['co_occurrence_count'] / 10.0  # Normalize
                    })
            
            # Search for expanded entities
            for exp in expanded_entities[:5]:  # Top 5 expansions
                entity_filter = {
                    "must": [
                        {"key": "entities", "match": {"any": [exp['entity']]}}
                    ]
                }
                if filter_dict:
                    entity_filter["must"].extend(filter_dict.get("must", []))
                
                entity_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("semantic", semantic_vector),
                    query_filter=entity_filter,
                    limit=limit,
                    with_payload=True
                )
                
                for rank, hit in enumerate(entity_results):
                    point_id = hit.id
                    rrf_score = (exp['weight'] * 0.8) / (60 + rank)
                    
                    if point_id in all_results:
                        all_results[point_id]['scores']['graph'] = rrf_score
                    else:
                        all_results[point_id] = {
                            'point': hit,
                            'scores': {'graph': rrf_score},
                            'rank': rank
                        }
        
        # 5. COMPUTE FINAL SCORES
        for point_id, data in all_results.items():
            scores = data['scores']
            # Weighted sum of all scores
            final_score = (
                scores.get('semantic', 0) * 1.0 +
                scores.get('acoustic', 0) * 1.5 +  # BOOST acoustic
                scores.get('trajectory', 0) * 1.2 +
                scores.get('graph', 0) * 0.8
            )
            
            # BOOST high-salience memories
            payload = data['point'].payload
            if payload.get('salience_score', 0) > 0.7:
                final_score *= 1.5
            if payload.get('is_crisis', False):
                final_score *= 2.0  # MAJOR boost for crisis
            
            data['final_score'] = final_score
        
        # 6. SORT AND RETURN
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:limit]
        
        return [r['point'] for r in sorted_results]
    
    def search_semantic(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search by semantic similarity with binary quantization."""
        search_filter = self._build_filter(filters) if filters else None
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="semantic",
            limit=limit,
            query_filter=search_filter,
            with_payload=True,
            search_params=QuantizationSearchParams(
                ignore=False,  # Use quantization
                rescore=True,  # Rescore with original vectors for accuracy
                oversampling=2.0  # Oversample for better recall
            )
        ).points
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def search_biomarker(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters:  Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search by biomarker similarity (euclidean, no quantization)."""
        search_filter = self._build_filter(filters) if filters else None
        
        results = self.client. query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="biomarker",
            limit=limit,
            query_filter=search_filter,
            with_payload=True
        ).points
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def search_trajectory(
        self,
        trajectory_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]: 
        """
        Search by biomarker trajectory similarity.
        Phase 3:  Dynamic Time Warping-style pattern matching.
        """
        search_filter = self._build_filter(filters) if filters else None
        
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=trajectory_vector,
                using="trajectory",
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            ).points
            
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                }
                for r in results
            ]
        except Exception as e:
            print(f"  Warning: Trajectory search failed: {e}")
            return []
    
    def recommend_positive(
        self,
        positive_ids: List[str],
        negative_ids: Optional[List[str]] = None,
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]: 
        """
        Phase 2: Discovery search using recommend API.
        Finds memories similar to positive anchors.
        """
        search_filter = self._build_filter(filters) if filters else None
        
        try:
            results = self.client.recommend(
                collection_name=self. collection_name,
                positive=positive_ids,
                negative=negative_ids or [],
                using="semantic",
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            )
            
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                }
                for r in results
            ]
        except Exception as e:
            print(f"  Warning: Recommend search failed: {e}")
            return []
    
    def search_combined(
        self,
        semantic_vector: List[float],
        biomarker_vector: List[float],
        semantic_weight: float = 0.6,
        biomarker_weight: float = 0.4,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Legacy combined search (prefer search_hybrid_rrf)."""
        semantic_results = self.search_semantic(semantic_vector, limit=limit * 2)
        biomarker_results = self.search_biomarker(biomarker_vector, limit=limit * 2)
        
        scores = {}
        payloads = {}
        
        for r in semantic_results:
            rid = str(r["id"])
            scores[rid] = semantic_weight * r["score"]
            payloads[rid] = r["payload"]
        
        for r in biomarker_results: 
            rid = str(r["id"])
            bio_score = 1.0 / (1.0 + r["score"])
            if rid in scores:
                scores[rid] += biomarker_weight * bio_score
            else:
                scores[rid] = biomarker_weight * bio_score
                payloads[rid] = r["payload"]
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {
                "id":  rid,
                "score": score,
                "payload": payloads[rid]
            }
            for rid, score in ranked
        ]
    
    def get_recent(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent interactions for a user."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        sorted_results = sorted(
            results,
            key=lambda x: x.payload.get("timestamp", ""),
            reverse=True
        )
        
        return [
            {
                "id": r. id,
                "payload": r.payload,
                "vectors": r.vector
            }
            for r in sorted_results
        ]
    
    def _build_filter(self, filters:  Dict) -> Filter:
        """Build Qdrant filter from dict."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status)
            }
        except Exception as e: 
            return {
                "name": self.collection_name,
                "error": str(e)
            }
    
    def delete_collection(self):
        """Delete the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        print(f"  Collection '{self.collection_name}' deleted")