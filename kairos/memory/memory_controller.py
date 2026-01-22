"""
Memory Controller - COMPLETE ENHANCED VERSION
All features properly integrated:  
1. User Profile (World State)
2. Salience Tracking (Core Memories)
3. Session Anchors (Forced Early Retrieval)
4. Graph-RAG (Associative Link Spreading)
5. Modality Tracking (Audio/Video/Text progression)
6. Intervention Tracking
7. Trajectory Matching
"""
import os
import json
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from .qdrant_manager import QdrantManager

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client. models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SparseVectorParams, SparseVector,
    Prefetch, FusionQuery, Fusion,
    PayloadSchemaType,
    QuantizationSearchParams
)
from sentence_transformers import SentenceTransformer

from config import (
    QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    EPISODIC_TOP_K, SEMANTIC_DIM, BIOMARKER_DIM
)
from .sparse_encoder import SparseEncoder
from . entity_graph import EntityGraph
from .intervention_tracker import InterventionTracker
from .trajectory_matcher import TrajectoryMatcher
from .user_profile import UserProfile
from . salience_tracker import SalienceTracker
from .session_anchors import SessionAnchors
from .graph_rag import GraphRAG
from .modality_tracker import ModalityTracker


TRAJECTORY_DIM = 160
SPARSE_VECTOR_NAME = "sparse_keywords"


class MemoryController:
    """
    Fully Enhanced Memory System with all features integrated. 
    """
    
    def __init__(self, qdrant_path: str = QDRANT_PATH, user_id: str = "default", session_id: str = None):
        self.qdrant_path = qdrant_path
        self. user_id = user_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collection_name = f"{COLLECTION_NAME}_{user_id}"
        self.interventions_collection = f"{COLLECTION_NAME}_{user_id}_interventions"
        
        os.makedirs(qdrant_path, exist_ok=True)
        
        print(f"  Initializing Enhanced Memory Controller...")
        self.client = QdrantClient(path=qdrant_path)
        
        print(f"  Loading sentence transformer...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = SEMANTIC_DIM
        
        # Phase 1 components
        print(f"  Initializing sparse encoder...")
        self.sparse_encoder = SparseEncoder()
        
        print(f"  Initializing entity graph...")
        self.entity_graph = EntityGraph(user_id, qdrant_path)
        
        # Phase 2 components
        print(f"  Initializing intervention tracker...")
        self.intervention_tracker = InterventionTracker(
            user_id=user_id,
            storage_path=os.path.join(qdrant_path, "interventions")
        )
        
        # Phase 3 components
        print(f"  Initializing trajectory matcher...")
        self.trajectory_matcher = TrajectoryMatcher(
            biomarker_dim=BIOMARKER_DIM,
            trajectory_window=5
        )
        
        # NEW: World State (User Profile)
        print(f"  Initializing user profile...")
        self.user_profile = UserProfile(user_id, os.path.join(qdrant_path, "profiles"))
        
        # NEW:  Salience Tracker
        print(f"  Initializing salience tracker...")
        self.salience_tracker = SalienceTracker()
        
        # NEW: Session Anchors
        print(f"  Initializing session anchors...")
        self.session_anchors = SessionAnchors(self.session_id, max_anchors=3)
        
        # NEW:  Graph-RAG
        print(f"  Initializing Graph-RAG...")
        self.graph_rag = GraphRAG(user_id, os.path.join(qdrant_path, "graphs"))
        
        # NEW:  Modality Tracker
        print(f"  Initializing modality tracker...")
        self.modality_tracker = ModalityTracker(user_id, os.path.join(qdrant_path, "modality"))
        
        # Create collections
        self._ensure_collection()
        
        # Cross-session memory
        self. session_memory_path = Path(qdrant_path) / f"session_memory_{user_id}. pkl"
        self.cross_session_data = self._load_cross_session_memory()
        
        # Debug tracking
        self.last_query_debug = {}
        self.total_memories = 0
        self.session_memories = 0
        self.debug_mode = True
        
        print(f"  ✅ Memory Controller initialized successfully")

    def _ensure_collection(self):
        """Create collection with all indexes."""
        collections = [c. name for c in self.client. get_collections().collections]
        
        if self.collection_name not in collections:
            print(f"  Creating collection: {self.collection_name}")
            
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "semantic": VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                        "emotional": VectorParams(size=8, distance=Distance.COSINE),
                        "biomarker": VectorParams(size=BIOMARKER_DIM, distance=Distance.EUCLID),
                        "trajectory": VectorParams(size=TRAJECTORY_DIM, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams()
                    }
                )
                
                # Create payload indexes for filtering
                index_fields = [
                    ("salience_score", PayloadSchemaType.FLOAT),
                    ("memory_type", PayloadSchemaType. KEYWORD),
                    ("modality", PayloadSchemaType. KEYWORD),
                    ("is_anchor", PayloadSchemaType. BOOL),
                    ("is_crisis", PayloadSchemaType. BOOL),
                    ("user_id", PayloadSchemaType.KEYWORD),
                    ("session_id", PayloadSchemaType.KEYWORD),
                ]
                
                for field, schema in index_fields:
                    try:
                        self.client. create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=schema
                        )
                    except Exception: 
                        pass
                
                print(f"  ✅ Collection created with indexes")
                
            except Exception as e: 
                print(f"  Warning: Collection creation issue: {e}")
                # Fallback without sparse vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "semantic": VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                        "emotional": VectorParams(size=8, distance=Distance.COSINE),
                        "biomarker": VectorParams(size=BIOMARKER_DIM, distance=Distance.EUCLID),
                        "trajectory": VectorParams(size=TRAJECTORY_DIM, distance=Distance.COSINE)
                    }
                )
        else:
            info = self.client.get_collection(self.collection_name)
            self.total_memories = info.points_count
            print(f"  Collection exists with {self.total_memories} memories")
    
    def _load_cross_session_memory(self) -> Dict:
        """Load cross-session memory."""
        if self.session_memory_path.exists():
            try:
                with open(self. session_memory_path, 'rb') as f:
                    return pickle.load(f)
            except Exception: 
                pass
        
        return {
            "user_id": self.user_id,
            "sessions": [],
            "user_profile": {
                "common_topics": [],
                "crisis_history": [],
                "positive_moments": [],
                "positive_moment_ids": []
            }
        }
    
    def _save_cross_session_memory(self):
        """Save cross-session memory."""
        try:
            with open(self.session_memory_path, 'wb') as f:
                pickle.dump(self.cross_session_data, f)
        except Exception as e: 
            print(f"  Warning: Could not save cross-session:  {e}")
    
    def save_intervention_outcome(
        self,
        pre_biomarker:  List[float],
        post_biomarker: List[float],
        intervention_type: str,
        user_emotion: str,
        session_id: str
    ) -> Optional[str]:
        """Save intervention outcome for learning."""
        try:
            return self.intervention_tracker.record_intervention(
                pre_state_vector=pre_biomarker,
                post_state_vector=post_biomarker,
                intervention_type=intervention_type,
                user_emotion=user_emotion,
                session_id=session_id
            )
        except Exception as e:
            if self.debug_mode:
                print(f"  [Intervention] Error:  {e}")
            return None
    
    def save_interaction(
        self,
        user_text: str,
        system_response: str,
        vectors: Dict[str, Any],
        payload:  Dict[str, Any],
        user_id: str = None,
        session_id: str = None
    ) -> str:
        """Save interaction with all enhancements."""
        turn_number = payload.get("turn_number", 0)
        modality = payload.get("modality_type", "text")
        
        # Determine if this should be an anchor
        is_anchor = self.session_anchors.should_be_anchor(turn_number)
        if is_anchor:
            point_id = self.session_anchors.generate_anchor_id(turn_number)
        else:
            point_id = str(uuid.uuid4())
        
        if self.debug_mode:
            print(f"  [Memory] Saving turn {turn_number}, modality={modality}, anchor={is_anchor}")
        
        facts_found = self.user_profile.extract_facts_from_text(user_text, turn_number)
        if facts_found and self.debug_mode:
            print(f"  [Memory] Extracted facts: {[f.get('type') for f in facts_found]}")
            # CRITICAL FIX: Force immediate save if name extracted
            for fact in facts_found:
                if fact.get('type') == 'name':
                    self.user_profile.force_save()
                    print(f"  [Memory] 🔒 Name saved immediately: {fact.get('value')}")
        
        # Calculate salience
        salience_data = self.salience_tracker. calculate_salience(
            text=user_text,
            emotional_intensity=payload.get("emotional_intensity", 0.5),
            crisis_score=payload.get("crisis_score", 0.0),
            is_crisis=payload.get("is_crisis", False),
            detected_emotion=payload.get("detected_emotion", "neutral"),
            turn_number=turn_number,
            is_first_turn=(turn_number == 0),
            contains_new_facts=len(facts_found) > 0,
            new_facts_count=len(facts_found),
            trigger_entities=payload.get("trigger_entities", []),
            modality=modality
        )
        
        if self.debug_mode:
            print(f"  [Memory] Salience:  {salience_data['salience_score']:.2f} ({salience_data['category']})")
        
        # Create vectors
        semantic_embedding = self.encoder.encode(user_text).tolist()
        
        biomarker = vectors.get('biomarker', [0] * BIOMARKER_DIM)
        if isinstance(biomarker, np.ndarray):
            biomarker = biomarker.tolist()
        while len(biomarker) < BIOMARKER_DIM: 
            biomarker.append(0.0)
        biomarker = biomarker[: BIOMARKER_DIM]
        
        emotional_vector = self._create_emotional_vector(biomarker, payload)
        
        # Update trajectory matcher and get trajectory vector
        self.trajectory_matcher.add_biomarker(
            biomarker=biomarker,
            turn_number=turn_number,
            emotion=payload.get("detected_emotion")
        )
        trajectory_vector = self.trajectory_matcher.get_trajectory_vector()
        
        # Build biomarker summary for storage
        biomarker_summary = {
            "jitter": float(biomarker[0]) if len(biomarker) > 0 else 0,
            "shimmer": float(biomarker[1]) if len(biomarker) > 1 else 0,
            "f0_variance": float(biomarker[2]) if len(biomarker) > 2 else 0,
            "loudness": float(biomarker[3]) if len(biomarker) > 3 else 0,
            "sentiment": float(biomarker[22]) if len(biomarker) > 22 else 0,
            "crying":  float(biomarker[25]) if len(biomarker) > 25 else 0,
            "laughter": float(biomarker[24]) if len(biomarker) > 24 else 0
        }
        
        # Determine memory type
        memory_type = "interaction"
        if facts_found:
            memory_type = "fact"
        if payload.get("is_crisis"):
            memory_type = "crisis"
        if is_anchor:
            memory_type = "anchor"
        
        # Build payload
        memory_payload = {
            "user_text": user_text,
            "system_response": system_response,
            "user_id": user_id or self.user_id,
            "session_id": session_id or self.session_id,
            "timestamp": datetime.now().isoformat(),
            "turn_number":  turn_number,
            
            # Modality info
            "modality":  modality,
            "modality_type": modality,
            "biomarker_summary": biomarker_summary,
            "feature_insights": payload.get("feature_insights", ""),
            
            # Emotional state
            "detected_emotion": payload.get("detected_emotion", "unknown"),
            "emotional_intensity":  payload.get("emotional_intensity", 0.5),
            "risk_level": payload.get("risk_level", "LOW"),
            "is_crisis": payload.get("is_crisis", False),
            "crisis_score": payload.get("crisis_score", 0.0),
            "intervention_type": payload.get("intervention_type", "SUPPORT"),
            
            # Salience
            "salience_score": salience_data["salience_score"],
            "salience_category": salience_data["category"],
            "should_always_retrieve": salience_data["should_always_retrieve"],
            
            # Memory metadata
            "memory_type": memory_type,
            "is_anchor": is_anchor,
            "contains_facts": len(facts_found) > 0,
            
            # Entity tracking
            "entities": payload.get("entities", []),
            "trigger_entities": payload.get("trigger_entities", [])
        }
        
        # Build vectors dict
        vector_dict = {
            "semantic": semantic_embedding,
            "emotional": emotional_vector,
            "biomarker": biomarker,
            "trajectory": trajectory_vector
        }
        
        # Store point
        point = PointStruct(
            id=point_id,
            vector=vector_dict,
            payload=memory_payload
        )
        
        try:
            self.client.upsert(
                collection_name=self. collection_name,
                points=[point]
            )
        except Exception as e:
            print(f"  [Memory] Error storing:  {e}")
        
        # Register anchor if applicable
        if is_anchor: 
            self.session_anchors. register_anchor(
                turn_number=turn_number,
                user_text=user_text,
                system_response=system_response,
                emotion=payload.get("detected_emotion", "unknown"),
                emotional_intensity=payload.get("emotional_intensity", 0.5),
                is_crisis=payload.get("is_crisis", False),
                crisis_score=payload.get("crisis_score", 0.0),
                salience_score=salience_data["salience_score"],
                entities=payload.get("entities", []),
                modality=modality,
                biomarker_summary=payload.get("feature_insights", "")
            )
        
        # Update Graph-RAG
        entities = payload.get("entities", [])
        self.graph_rag.update_graph(
            entities=entities,
            emotion=payload.get("detected_emotion", "neutral"),
            emotional_intensity=payload.get("emotional_intensity", 0.5),
            is_crisis=payload.get("is_crisis", False),
            text=user_text[: 200]
        )
        
        # Update Modality Tracker
        self. modality_tracker.record_interaction(
            modality=modality,
            text=user_text,
            emotion=payload.get("detected_emotion", "neutral"),
            emotional_intensity=payload.get("emotional_intensity", 0.5),
            is_crisis=payload.get("is_crisis", False),
            biomarker=biomarker,
            turn_number=turn_number,
            feature_insights=payload.get("feature_insights", "")
        )
        
        # Update entity graph
        for entity in entities:
            self. entity_graph.update_entity(
                entity=entity,
                emotion=payload. get("detected_emotion", "neutral"),
                is_crisis=payload.get("is_crisis", False),
                is_positive=payload.get("emotional_intensity", 0.5) < 0.5 and 
                           payload.get("detected_emotion", "") in ['joy', 'contentment', 'happiness'],
                context=user_text[:200],
                co_occurring_entities=[e for e in entities if e != entity]
            )
        
        # Update cross-session data
        self._update_cross_session_data(memory_payload, point_id)
        
        self.session_memories += 1
        self.total_memories += 1
        
        return point_id
    def build_rich_context(
        self,
        query_text: str,
        biomarker: np.ndarray,
        entities: List[str],
        session_id: str,
        limit: int = 8
    ) -> str:
        """Build RICH context that actually uses biomarker data."""
        
        context_parts = []
        
        # 1. SESSION ANCHORS (always include)
        if self.session_anchors and self.session_anchors.session_id == session_id:
            anchor_context = self.session_anchors.get_anchor_context_for_llm()
            if anchor_context:
                context_parts.append(anchor_context)
        
        # 2. RETRIEVE RELEVANT MEMORIES using direct client calls
        semantic_embedding = self.encoder.encode(query_text).tolist()
        
        # Convert biomarker to proper format
        biomarker_list = biomarker.tolist() if isinstance(biomarker, np.ndarray) else biomarker
        if biomarker_list and len(biomarker_list) >= BIOMARKER_DIM:
            biomarker_list = biomarker_list[:BIOMARKER_DIM]
        else:
            biomarker_list = None
        
        # Get trajectory vector
        trajectory_vector = self.trajectory_matcher.get_trajectory_vector()
        
        # Perform hybrid search using prefetch + RRF fusion
        try:
            prefetch_queries = [
                Prefetch(query=semantic_embedding, using="semantic", limit=limit * 2)
            ]
            
            # Add biomarker search if available
            if biomarker_list:
                emotional_vector = self._create_emotional_vector(biomarker_list, {})
                prefetch_queries.append(
                    Prefetch(query=emotional_vector, using="emotional", limit=limit)
                )
            
            # Add trajectory search
            if trajectory_vector and len(trajectory_vector) == TRAJECTORY_DIM:
                prefetch_queries.append(
                    Prefetch(query=trajectory_vector, using="trajectory", limit=limit)
                )
            
            # Execute fusion search
            fusion_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=True,
                search_params=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0
                )
            ).points
            
            retrieved = fusion_results
            
        except Exception as e:
            print(f"  [Memory] Hybrid search failed: {e}")
            # Fallback to simple semantic search
            try:
                retrieved = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("semantic", semantic_embedding),
                    limit=limit,
                    with_payload=True,
                    with_vectors=True
                )
            except Exception as e2:
                print(f"  [Memory] Fallback search also failed: {e2}")
                retrieved = []
        
        if not retrieved:
            return "\n\n".join(context_parts) if context_parts else "No previous context available."
        
        # 3. BUILD MEMORY CONTEXT
        context_parts.append("\n=== RELEVANT PAST MOMENTS ===")
        
        for i, point in enumerate(retrieved[:6], 1):  # Top 6
            payload = point.payload
            
            # Extract key info
            emotion = payload.get('detected_emotion', 'unknown')
            intensity = payload.get('emotional_intensity', 0)
            crisis = payload.get('is_crisis', False)
            salience = payload.get('salience_score', 0)
            user_text = payload.get('user_text', '')[:200]
            timestamp = payload.get('timestamp', '')
            
            # Get biomarker summary
            bio_summary = payload.get('feature_insights', '') or payload.get('biomarker_summary', '')
            
            crisis_marker = "🚨 CRISIS " if crisis else ""
            salience_marker = "⭐" if salience > 0.7 else ""
            
            context_parts.append(
                f"\n[Memory {i}] {crisis_marker}{salience_marker}{emotion} ({int(intensity*100)}%)"
            )
            context_parts.append(f"When: {timestamp}")
            context_parts.append(f"Said: \"{user_text}...\"")
            
            if bio_summary:
                context_parts.append(f"Biomarkers then: {bio_summary[:150]}")
            
            # COMPARE to current biomarkers
            if biomarker_list and hasattr(point, 'vector') and 'biomarker' in point.vector:
                past_biomarker = point.vector['biomarker']
                if isinstance(past_biomarker, list) and len(past_biomarker) >= 8:
                    current_bio_slice = biomarker_list[:8]
                    past_bio_slice = past_biomarker[:8]
                    
                    # Calculate similarity
                    try:
                        similarity = np.dot(current_bio_slice, past_bio_slice) / (
                            np.linalg.norm(current_bio_slice) * np.linalg.norm(past_bio_slice) + 1e-8
                        )
                        
                        if similarity > 0.85:
                            context_parts.append(
                                f"⚠️ PATTERN MATCH: Current voice/behavior matches this memory (similarity: {similarity:.2f})"
                            )
                    except:
                        pass
            
            context_parts.append("")  # Spacing
        
        # 4. TRAJECTORY PATTERNS
        trajectory_pattern = self.trajectory_matcher.get_trajectory_pattern()
        if trajectory_pattern['pattern'] != 'INSUFFICIENT_DATA':
            context_parts.append("\n=== EMOTIONAL TRAJECTORY ===")
            context_parts.append(f"Pattern: {trajectory_pattern['pattern']}")
            context_parts.append(f"Description: {trajectory_pattern['description']}")
            context_parts.append(f"Current distress: {trajectory_pattern.get('current_distress', 0):.2f}")
            
            if trajectory_pattern.get('alert_level') in ['HIGH', 'CRITICAL']:
                context_parts.append(f"⚠️ ALERT: {trajectory_pattern['recommendation']}")
        
        # 5. TRIGGER ENTITIES
        if entities:
            trigger_alerts = self.entity_graph.check_trigger_entities(entities, threshold=0.3)
            if trigger_alerts:
                context_parts.append("\n=== TRIGGER ENTITIES DETECTED ===")
                for alert in trigger_alerts[:3]:
                    context_parts.append(
                        f"⚠️ {alert['entity']}: trigger_score={alert['trigger_score']:.2f} "
                        f"(mentioned {alert['mention_count']}x, crisis {alert['crisis_count']}x)"
                    )
        
        return "\n".join(context_parts)
    def _create_emotional_vector(self, biomarker: List[float], payload: Dict) -> List[float]:
        """Create 8-dim emotional state vector."""
        return [
            biomarker[0] if len(biomarker) > 0 else 0,  # jitter
            biomarker[1] if len(biomarker) > 1 else 0,  # shimmer
            biomarker[2] if len(biomarker) > 2 else 0,  # f0_variance
            biomarker[22] if len(biomarker) > 22 else 0,  # sentiment
            float(payload.get("emotional_intensity", 0.5)),
            1.0 if payload.get("is_crisis") else 0.0,
            biomarker[25] if len(biomarker) > 25 else 0,  # crying
            biomarker[24] if len(biomarker) > 24 else 0,  # laughter
        ]
    
    def _update_cross_session_data(self, payload: Dict, point_id: str):
        """Update cross-session data."""
        profile = self.cross_session_data["user_profile"]
        
        if payload.get("is_crisis"):
            profile["crisis_history"].append({
                "timestamp": payload["timestamp"],
                "text_snippet": payload["user_text"][:100],
                "modality": payload. get("modality", "text"),
                "point_id": point_id
            })
        
        # Bound lists
        profile["crisis_history"] = profile["crisis_history"][-20:]
        profile["positive_moments"] = profile["positive_moments"][-20:]
        profile["positive_moment_ids"] = profile["positive_moment_ids"][-20:]
        
        if self.session_memories % 5 == 0:
            self._save_cross_session_memory()
    
    def get_context(
        self,
        vectors: Dict[str, Any],
        modality: str,
        entities: List[str],
        user_id: str,
        current_text: str = None,
        top_k: int = EPISODIC_TOP_K,
        debug:  bool = True
    ) -> Tuple[str, Dict]: 
        """
        Enhanced context retrieval with Graph-RAG and modality awareness.
        """
        self.last_query_debug = {
            "query_type": [],
            "user_profile_facts": {},
            "anchor_memories": [],
            "high_salience_memories": [],
            "graph_rag_expansion": {},
            "modality_context": {},
            "semantic_memories": [],
            "total_searched": 0
        }
        
        if self.debug_mode:
            print(f"  [Memory] Starting retrieval...")
        
        all_results = []
        context_parts = []
        

        # 1. USER PROFILE (World State) - Deterministic facts
        
        user_facts = self.user_profile.get_facts_for_llm()
        if user_facts:
            context_parts.append(user_facts)
            self.last_query_debug["user_profile_facts"] = self.user_profile.get_all_facts()
            if self.debug_mode:
                print(f"  [Memory] User facts loaded")
        
        
        # 2. SESSION ANCHORS - O(1) lookup for "reason for visit"
        
        anchor_ids = self.session_anchors. get_anchor_ids()
        if anchor_ids: 
            try:
                anchor_points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=anchor_ids,
                    with_payload=True
                )
                
                anchor_context = self.session_anchors.get_anchor_context_for_llm()
                if anchor_context: 
                    context_parts.append(anchor_context)
                
                self.last_query_debug["anchor_memories"] = [
                    {"id": p.id, "text": p. payload.get("user_text", "")[:80]}
                    for p in anchor_points
                ]
                
                if self.debug_mode:
                    print(f"  [Memory] Retrieved {len(anchor_points)} anchors")
            except Exception as e:
                if self.debug_mode:
                    print(f"  [Memory] Anchor retrieval warning: {e}")
        
        
        # 3. HIGH SALIENCE MEMORIES - Always retrieve important moments
       
        try:
            high_salience_results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="salience_score", range=Range(gte=0.4)),  # LOWERED from 0.7
                        FieldCondition(key="user_id", match=MatchValue(value=user_id or self.user_id))
                    ]
                ),
                limit=10,  # INCREASED from 5
                with_payload=True
            )
            
            for r in high_salience_results: 
                if r.id not in anchor_ids:
                    all_results.append({
                        "type": "high_salience",
                        "score": r.payload.get("salience_score", 0.7),
                        "payload": r.payload,
                        "id": r.id
                    })
            
            if self.debug_mode and high_salience_results:
                print(f"  [Memory] Retrieved {len(high_salience_results)} high-salience memories")
                
            self.last_query_debug["high_salience_memories"] = [
                {"text": r.payload.get("user_text", "")[:80], "salience": r.payload.get("salience_score")}
                for r in high_salience_results
            ]
        except Exception as e:
            if self.debug_mode:
                print(f"  [Memory] High salience warning: {e}")
        
        
        # 3b. EMOTIONAL MEMORIES - Retrieve memories with strong emotions
        
        if current_text:
            emotional_keywords = ['sad', 'angry', 'hurt', 'upset', 'hate', 'scared', 'anxious']
            text_lower = current_text.lower()
            
            if any(kw in text_lower for kw in emotional_keywords):
                try:
                    emotional_results, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="emotional_intensity", range=Range(gte=0.4)),
                                FieldCondition(key="user_id", match=MatchValue(value=user_id or self.user_id))
                            ]
                        ),
                        limit=5,
                        with_payload=True
                    )
                    
                    for r in emotional_results:
                        if r.id not in [res.get("id") for res in all_results] and r.id not in anchor_ids:
                            all_results.append({
                                "type": "emotional_memory",
                                "score": r.payload.get("emotional_intensity", 0.5),
                                "payload": r.payload,
                                "id": r.id
                            })
                    
                    if self.debug_mode and emotional_results:
                        print(f"  [Memory] Retrieved {len(emotional_results)} emotional memories")
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"  [Memory] Emotional retrieval warning: {e}")
        
        # 4. GRAPH-RAG:  Associative Link Spreading
        
        graph_expansion = None
        expanded_terms = entities. copy() if entities else []
        
        if entities:
            graph_expansion = self.graph_rag.get_query_expansion(entities)
            expanded_terms = graph_expansion.get('expansion_terms', entities)
            
            self.last_query_debug["graph_rag_expansion"] = {
                "original":  entities,
                "expanded": expanded_terms,
                "related_with_scores": graph_expansion.get('related_with_scores', [])[:5],
                "emotional_context": graph_expansion.get('emotional_context', {}),
                "topics": graph_expansion.get('topic_context', [])
            }
            
            if self.debug_mode:
                related = [r['entity'] for r in graph_expansion. get('related_with_scores', [])[:5]]
                if related:
                    print(f"  [Memory] Graph-RAG expanded:  {entities} -> +{related}")
        
        
        # 5. MODALITY-SPECIFIC CONTEXT
       
        mentioned_modality = None
        if current_text: 
            text_lower = current_text.lower()
            if any(w in text_lower for w in ['audio', 'voice', 'recording', 'sound']):
                mentioned_modality = 'audio'
            elif any(w in text_lower for w in ['video', 'face', 'camera', 'see me']):
                mentioned_modality = 'video'
        
        if mentioned_modality: 
            modality_summary = self.modality_tracker. get_modality_progression_for_llm(mentioned_modality)
            if modality_summary:
                context_parts.append(f"\n{modality_summary}")
                self.last_query_debug["modality_context"] = {
                    "mentioned": mentioned_modality,
                    "summary": self.modality_tracker.get_modality_progression(mentioned_modality)
                }
                
                if self.debug_mode:
                    print(f"  [Memory] Added {mentioned_modality} modality context")
        
        
        # 6. GRAPH-RAG ENHANCED SEMANTIC SEARCH
        
        if current_text:
            semantic_embedding = self.encoder.encode(current_text).tolist()
            biomarker = vectors.get('biomarker', [0] * BIOMARKER_DIM)
            if isinstance(biomarker, np.ndarray):
                biomarker = biomarker.tolist()
            emotional_vector = self._create_emotional_vector(biomarker, {})
            
            # OPTIMIZATION: Use Graph-RAG Qdrant expansion if entities present
            if entities:
                graph_results = self._graph_rag_qdrant_expansion(
                    entities=entities,
                    semantic_vector=semantic_embedding,
                    limit=top_k
                )
                
                existing_ids = {r.get("id") for r in all_results if "id" in r}
                for r in graph_results:
                    if r["id"] not in existing_ids and r["id"] not in anchor_ids:
                        all_results.append(r)
                
                if self.debug_mode and graph_results:
                    print(f"  [Memory] Graph-RAG Qdrant expansion: {len(graph_results)} results")
            
            # Standard RRF fusion as fallback/supplement
            try:
                prefetch_queries = [
                    Prefetch(query=semantic_embedding, using="semantic", limit=top_k * 2),
                    Prefetch(query=emotional_vector, using="emotional", limit=top_k)
                ]
                
                fusion_results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch_queries,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                    search_params=QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0
                    )
                ).points
                
                existing_ids = {r.get("id") for r in all_results if "id" in r}
                for r in fusion_results:
                    if r.id not in existing_ids and r.id not in anchor_ids:
                        all_results.append({
                            "type": "semantic_fusion",
                            "score": r.score,
                            "payload": r.payload,
                            "id": r.id
                        })
                
                if self.debug_mode:
                    print(f"  [Memory] Semantic fusion returned {len(fusion_results)} results")
                
            except Exception as e:
                if self.debug_mode:
                    print(f"  [Memory] RRF fusion warning: {e}")
        
        self.last_query_debug["total_searched"] = len(all_results)
        self.last_query_debug["query_type"] = ["user_profile", "anchors", "high_salience", "graph_rag", "modality", "fusion"]
        
        # Sort by relevance
        all_results.sort(key=lambda x: x. get("score", 0), reverse=True)
        
        
        # BUILD CONTEXT STRING
        
        
        # Entity alerts from entity graph
        entity_alerts = self.entity_graph.check_trigger_entities(entities) if entities else []
        if entity_alerts:
            context_parts.append("\n=== ⚠️ TRIGGER ALERTS ===")
            for alert in entity_alerts:
                context_parts.append(f"'{alert['entity']}' is a TRIGGER (score: {alert['trigger_score']:.2f})")
        
        # Graph-RAG emotional context
        if graph_expansion and graph_expansion.get('emotional_context'):
            emotions = graph_expansion['emotional_context']
            if emotions:
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                context_parts.append(f"\n=== EMOTIONAL ASSOCIATIONS ===")
                for emotion, strength in top_emotions:
                    context_parts.append(f"• Related to '{emotion}' (strength: {strength:.2f})")
        
        # Related memories
        
        if all_results:
            context_parts.append("\n=== RELATED MEMORIES ===")
            for i, result in enumerate(all_results[:top_k]):
                payload = result["payload"]
                match_type = result["type"]
                
                salience = payload.get("salience_score", 0)
                salience_marker = "⭐ " if salience >= 0.7 else ""
                crisis_marker = "🚨 " if payload.get("is_crisis") else ""
                modality_marker = f"[{payload.get('modality', 'text').upper()}] " if payload.get('modality') != 'text' else ""
                turn_num = payload.get('turn_number', '?')
                
                # CRITICAL FIX: Include FULL text so LLM can actually see what happened
                user_text = payload.get('user_text', '')
                system_response = payload.get('system_response', '')
                
                context_parts.append(
                    f"[Memory {i+1} - Turn {turn_num}] {salience_marker}{crisis_marker}{modality_marker}({match_type})\n"
                    f"  User said: \"{user_text}\"\n"
                    f"  Kairos replied: \"{system_response[:200]}...\"\n"
                    f"  Emotion: {payload.get('detected_emotion', 'unknown')} | Intensity: {payload.get('emotional_intensity', 0):.2f}"
                )
            
            # Include biomarker summary for audio/video memories
            if payload.get('modality') in ['audio', 'video'] and payload.get('biomarker_summary'):
                bio = payload['biomarker_summary']
                if isinstance(bio, dict):
                    context_parts.append(
                        f"  Voice/Visual: jitter={bio.get('jitter', 0):.2f}, "
                        f"sentiment={bio.get('sentiment', 0):.2f}"
                    )
        
        context_string = "\n".join(context_parts) if context_parts else "No previous context."
        
        if not self.last_query_debug. get('query_type'):
            self.last_query_debug['query_type'] = ["attempted"]
            self.last_query_debug['total_searched'] = len(all_results)
            
        return context_string, self.last_query_debug
    
    def _graph_rag_qdrant_expansion(
        self,
        entities: List[str],
        semantic_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Graph-RAG using Qdrant prefetch for associative expansion.
        
        Instead of just semantic search, this:
        1. Searches for direct entity mentions
        2. Expands via Graph-RAG associations
        3. Uses prefetch to combine both
        """
        if not entities:
            return []
        
        # Get Graph-RAG expansion
        graph_expansion = self.graph_rag.get_query_expansion(entities)
        expanded_terms = graph_expansion.get('expansion_terms', entities)
        
        if not expanded_terms:
            return []
        
        # Build prefetch queries for each expanded term
        prefetch_queries = []
        
        # Main semantic query
        prefetch_queries.append(
            Prefetch(
                query=semantic_vector,
                using="semantic",
                limit=limit * 2
            )
        )
        
        # Add queries for each expanded entity
        for term in expanded_terms[:5]:  # Top 5 related entities
            try:
                term_embedding = self.encoder.encode(term).tolist()
                prefetch_queries.append(
                    Prefetch(
                        query=term_embedding,
                        using="semantic",
                        limit=limit
                    )
                )
            except:
                pass
        
        # Execute fusion query
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True
            ).points
            
            return [
                {
                    "type": "graph_rag_expansion",
                    "score": r.score,
                    "payload": r.payload,
                    "id": r.id
                }
                for r in results
            ]
        except Exception as e:
            if self.debug_mode:
                print(f"  [Memory] Graph-RAG Qdrant expansion failed: {e}")
            return []
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_user_name(self) -> Optional[str]:
        """Get user's name."""
        return self.user_profile.get_name()
    
    def get_user_facts(self) -> Dict[str, Any]:
        """Get all user facts."""
        return self. user_profile.get_all_facts()
    
    def get_user_facts_for_llm(self) -> str:
        """Get formatted user facts for LLM."""
        return self.user_profile. get_facts_for_llm()
    
    def get_modality_summary(self, modality: str) -> Dict[str, Any]:
        """Get summary for a specific modality."""
        return self.modality_tracker.get_modality_progression(modality)
    
    def get_modality_progression_for_llm(self, modality: str) -> str:
        """Get modality progression for LLM."""
        return self.modality_tracker.get_modality_progression_for_llm(modality)
    
    def get_effective_intervention_recommendation(
        self,
        current_emotion: str,
        current_biomarker: List[float]
    ) -> Optional[str]:
        """Get recommended intervention based on past effectiveness."""
        result = self.intervention_tracker.get_recommended_intervention(
            current_emotion=current_emotion,
            current_biomarker=current_biomarker
        )
        if result:
            return result.get('intervention_type')
        return None
    
    def get_trajectory_alert(self, turn_number: int) -> Optional[Dict[str, Any]]:
        """Get trajectory alert if warranted."""
        return self.trajectory_matcher.get_alert(turn_number)
    
    def get_trajectory_pattern(self) -> Dict[str, Any]:
        """Get current trajectory pattern."""
        return self. trajectory_matcher.get_trajectory_pattern()
    
    def save_session_summary(self, session_id: str, summary: Dict):
        """Save session summary."""
        self.cross_session_data["sessions"].append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        })
        self.cross_session_data["sessions"] = self.cross_session_data["sessions"][-50:]
        
        self. entity_graph.force_save()
        self.graph_rag.force_save()
        self.modality_tracker.force_save()
        self.intervention_tracker.force_save()
        self._save_cross_session_memory()
    
    def get_debug_info(self) -> Dict:
        """Get comprehensive debug info."""
        return {
            "total_memories": self.total_memories,
            "session_memories": self.session_memories,
            "user_profile": self.user_profile. get_stats(),
            "salience_tracker": self.salience_tracker. get_session_stats(),
            "session_anchors": self.session_anchors.get_stats(),
            "entity_graph": self.entity_graph.get_stats(),
            "graph_rag": self.graph_rag.get_stats(),
            "modality_tracker":  self.modality_tracker.get_stats(),
            "intervention_tracker": self.intervention_tracker. get_stats(),
            "trajectory_matcher": self.trajectory_matcher.get_stats(),
            "last_query":  self.last_query_debug
        }
    
    def set_debug_mode(self, enabled: bool):
        """Set debug mode."""
        self.debug_mode = enabled
    
    def force_save_all(self):
        """Force save all components."""
        self.entity_graph.force_save()
        self.graph_rag.force_save()
        self.modality_tracker.force_save()
        self.intervention_tracker.force_save()
        self.user_profile.force_save()
        self._save_cross_session_memory()
        print("  ✅ All memory components saved")