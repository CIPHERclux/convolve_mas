"""
Graph-RAG (Retrieval Augmented Generation with Graph)
Implements spreading activation for associative memory retrieval. 

When user mentions "mom", we don't just retrieve memories about "mom" -
we also retrieve related concepts (family, home, childhood) based on
learned associations. 

This enables more human-like associative recall. 
"""
import json
import os
import math
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation. 
    
    Features:
    - Entity relationship tracking
    - Spreading activation for query expansion
    - Emotional context propagation
    - Topic clustering
    - Co-occurrence strength learning
    
    The graph captures: 
    - Which entities appear together (co-occurrence)
    - What emotions are associated with entities
    - Topic clusters (work, family, health, etc.)
    - Trigger relationships (entity X triggers thoughts of Y)
    """
    
    def __init__(self, user_id: str, storage_path: str = "./graph_data"):
        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.graph_file = self.storage_path / f"{user_id}_graph_rag.json"
        
        # Core graph structures
        self.nodes:  Dict[str, Dict[str, Any]] = {}  # entity -> node data
        self.edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # entity -> {related_entity -> weight}
        
        # Emotional associations
        self.entity_emotions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Topic assignments
        self.entity_topics: Dict[str, Set[str]] = defaultdict(set)
        self.topic_entities: Dict[str, Set[str]] = defaultdict(set)
        
        # Predefined topic keywords for classification
        self.topic_keywords = {
            'family': {'mom', 'mother', 'dad', 'father', 'brother', 'sister', 'parent', 'family', 'son', 'daughter', 'grandma', 'grandpa', 'aunt', 'uncle', 'cousin'},
            'relationship': {'boyfriend', 'girlfriend', 'partner', 'spouse', 'husband', 'wife', 'dating', 'relationship', 'love', 'breakup', 'ex'},
            'work': {'job', 'work', 'boss', 'coworker', 'office', 'career', 'salary', 'promotion', 'fired', 'quit', 'interview'},
            'school': {'school', 'college', 'university', 'class', 'teacher', 'professor', 'exam', 'test', 'homework', 'grades', 'student'},
            'health': {'doctor', 'therapist', 'medication', 'diagnosis', 'hospital', 'sick', 'pain', 'anxiety', 'depression', 'therapy'},
            'friendship': {'friend', 'bestie', 'buddy', 'friendship', 'friends'},
            'self': {'myself', 'me', 'self', 'i'},
        }
        
        # Activation decay for spreading
        self.activation_decay = 0.5
        self.max_spread_depth = 3
        self.min_activation_threshold = 0.1
        
        # Statistics
        self.total_updates = 0
        
        # Load existing graph
        self._load_graph()
    
    def _load_graph(self):
        """Load graph from disk."""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                
                self.nodes = data.get('nodes', {})
                
                # Reconstruct defaultdicts
                self.edges = defaultdict(lambda: defaultdict(float))
                for e1, related in data.get('edges', {}).items():
                    for e2, weight in related.items():
                        self.edges[e1][e2] = weight
                
                self.entity_emotions = defaultdict(lambda: defaultdict(float))
                for entity, emotions in data.get('entity_emotions', {}).items():
                    for emotion, strength in emotions.items():
                        self.entity_emotions[entity][emotion] = strength
                
                self.entity_topics = defaultdict(set)
                for entity, topics in data.get('entity_topics', {}).items():
                    self.entity_topics[entity] = set(topics)
                
                self.topic_entities = defaultdict(set)
                for topic, entities in data.get('topic_entities', {}).items():
                    self.topic_entities[topic] = set(entities)
                
                self.total_updates = data.get('total_updates', 0)
                
                print(f"  [GraphRAG] Loaded graph:  {len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())} edges")
                
            except Exception as e: 
                print(f"  [GraphRAG] Warning: Could not load graph: {e}")
    
    def _save_graph(self):
        """Save graph to disk."""
        try:
            data = {
                'user_id': self.user_id,
                'nodes': self. nodes,
                'edges': {k: dict(v) for k, v in self.edges.items()},
                'entity_emotions': {k: dict(v) for k, v in self.entity_emotions.items()},
                'entity_topics': {k: list(v) for k, v in self.entity_topics.items()},
                'topic_entities':  {k: list(v) for k, v in self.topic_entities.items()},
                'total_updates': self.total_updates,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.graph_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [GraphRAG] Warning: Could not save graph: {e}")
    
    def update_graph(
        self,
        entities: List[str],
        emotion: str = None,
        emotional_intensity: float = 0.5,
        is_crisis: bool = False,
        text: str = None
    ):
        """
        Update graph with new entity mentions.
        
        Args:
            entities: List of entities mentioned
            emotion: Detected emotion
            emotional_intensity: Intensity of emotion
            is_crisis: Whether this is a crisis context
            text: Original text for topic extraction
        """
        if not entities:
            return
        
        self.total_updates += 1
        entities_lower = [e.lower() for e in entities]
        
        # Update/create nodes
        for entity in entities_lower:
            if entity not in self.nodes:
                self. nodes[entity] = {
                    'mention_count': 0,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': None,
                    'crisis_count': 0,
                    'total_intensity': 0.0,
                }
            
            self.nodes[entity]['mention_count'] += 1
            self. nodes[entity]['last_seen'] = datetime.now().isoformat()
            self.nodes[entity]['total_intensity'] += emotional_intensity
            
            if is_crisis:
                self. nodes[entity]['crisis_count'] += 1
            
            # Update emotional association
            if emotion:
                current = self.entity_emotions[entity]. get(emotion. lower(), 0)
                # Exponential moving average
                alpha = 0.3
                self.entity_emotions[entity][emotion. lower()] = (
                    (1 - alpha) * current + alpha * emotional_intensity
                )
            
            # Assign topics
            self._assign_topics(entity)
        
        # Update co-occurrence edges
        for i, e1 in enumerate(entities_lower):
            for e2 in entities_lower[i+1:]:
                # Bidirectional edge update
                weight_boost = 1.0
                if is_crisis:
                    weight_boost = 1.5  # Stronger association in crisis
                if emotional_intensity > 0.7:
                    weight_boost *= 1.2  # Stronger association with intense emotion
                
                self.edges[e1][e2] += weight_boost
                self.edges[e2][e1] += weight_boost
        
        # Periodic save
        if self.total_updates % 10 == 0:
            self._save_graph()
    
    def _assign_topics(self, entity: str):
        """Assign entity to topic clusters."""
        entity_lower = entity.lower()
        
        for topic, keywords in self.topic_keywords.items():
            if entity_lower in keywords:
                self.entity_topics[entity_lower].add(topic)
                self.topic_entities[topic]. add(entity_lower)
    
    def get_query_expansion(
        self,
        query_entities: List[str],
        max_expansions: int = 10,
        include_topics: bool = True
    ) -> Dict[str, Any]:
        """
        Expand query entities using spreading activation.
        
        Args:
            query_entities: Initial entities from query
            max_expansions:  Maximum entities to add
            include_topics: Whether to include topic-based expansion
            
        Returns: 
            Dict with expanded entities, scores, and context
        """
        if not query_entities:
            return {
                'original_entities': [],
                'expansion_terms': [],
                'related_with_scores': [],
                'emotional_context': {},
                'topic_context': []
            }
        
        query_lower = [e.lower() for e in query_entities]
        
        # Spreading activation
        activations = self._spread_activation(query_lower)
        
        # Get top expansions (excluding original query terms)
        expansions = []
        for entity, score in sorted(activations.items(), key=lambda x: x[1], reverse=True):
            if entity not in query_lower and score >= self.min_activation_threshold:
                expansions.append({
                    'entity': entity,
                    'activation_score': round(score, 4),
                    'mention_count': self.nodes. get(entity, {}).get('mention_count', 0),
                    'emotions': dict(self.entity_emotions. get(entity, {}))
                })
                if len(expansions) >= max_expansions:
                    break
        
        # Aggregate emotional context
        emotional_context = defaultdict(float)
        for entity in query_lower:
            for emotion, strength in self.entity_emotions. get(entity, {}).items():
                emotional_context[emotion] += strength
        
        # Normalize emotional context
        if emotional_context:
            max_emotion = max(emotional_context.values())
            if max_emotion > 0:
                emotional_context = {k: round(v/max_emotion, 3) for k, v in emotional_context.items()}
        
        # Topic context
        topic_context = []
        if include_topics:
            topic_scores = defaultdict(int)
            for entity in query_lower:
                for topic in self.entity_topics. get(entity, []):
                    topic_scores[topic] += 1
            topic_context = sorted(topic_scores.keys(), key=lambda t: topic_scores[t], reverse=True)
        
        return {
            'original_entities':  query_entities,
            'expansion_terms': [e['entity'] for e in expansions],
            'related_with_scores': expansions,
            'emotional_context': dict(emotional_context),
            'topic_context': topic_context[: 5]
        }
    
    def _spread_activation(
        self,
        source_entities: List[str],
        initial_activation: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform spreading activation from source entities.
        
        Args:
            source_entities: Starting nodes
            initial_activation: Initial activation level
            
        Returns:
            Dict of entity -> activation score
        """
        activations:  Dict[str, float] = {}
        
        # Initialize source activations
        for entity in source_entities:
            activations[entity] = initial_activation
        
        # Spread activation iteratively
        current_frontier = set(source_entities)
        
        for depth in range(self.max_spread_depth):
            next_frontier = set()
            decay = self.activation_decay ** (depth + 1)
            
            for entity in current_frontier: 
                current_activation = activations.get(entity, 0)
                
                if current_activation < self.min_activation_threshold:
                    continue
                
                # Spread to neighbors
                neighbors = self. edges.get(entity, {})
                
                for neighbor, edge_weight in neighbors.items():
                    # Normalize edge weight
                    total_out_weight = sum(self.edges[entity].values())
                    if total_out_weight > 0:
                        normalized_weight = edge_weight / total_out_weight
                    else: 
                        normalized_weight = 0.5
                    
                    # Calculate spread activation
                    spread_activation = current_activation * decay * normalized_weight
                    
                    if spread_activation >= self.min_activation_threshold:
                        # Accumulate (don't overwrite)
                        activations[neighbor] = activations.get(neighbor, 0) + spread_activation
                        next_frontier.add(neighbor)
            
            current_frontier = next_frontier
            
            if not current_frontier:
                break
        
        return activations
    
    def get_related_entities(
        self,
        entity: str,
        limit: int = 5,
        min_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Get directly related entities."""
        entity_lower = entity.lower()
        
        if entity_lower not in self.edges:
            return []
        
        related = []
        for other_entity, weight in sorted(
            self.edges[entity_lower].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if weight >= min_weight:
                related.append({
                    'entity': other_entity,
                    'weight': round(weight, 2),
                    'emotions': dict(self.entity_emotions.get(other_entity, {})),
                    'topics': list(self.entity_topics.get(other_entity, []))
                })
                if len(related) >= limit:
                    break
        
        return related
    
    def get_entities_by_topic(self, topic: str, limit: int = 10) -> List[str]:
        """Get entities belonging to a topic."""
        entities = list(self.topic_entities.get(topic. lower(), []))
        
        # Sort by mention count
        entities.sort(
            key=lambda e: self.nodes. get(e, {}).get('mention_count', 0),
            reverse=True
        )
        
        return entities[:limit]
    
    def get_entities_by_emotion(
        self,
        emotion: str,
        limit: int = 5,
        min_strength: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get entities most associated with an emotion."""
        emotion_lower = emotion.lower()
        results = []
        
        for entity, emotions in self.entity_emotions.items():
            if emotion_lower in emotions and emotions[emotion_lower] >= min_strength:
                results.append({
                    'entity': entity,
                    'emotion_strength': round(emotions[emotion_lower], 3),
                    'mention_count': self.nodes.get(entity, {}).get('mention_count', 0)
                })
        
        results.sort(key=lambda x: x['emotion_strength'], reverse=True)
        return results[:limit]
    
    def get_emotional_profile(self, entity: str) -> Dict[str, Any]:
        """Get emotional profile for an entity."""
        entity_lower = entity.lower()
        
        if entity_lower not in self.nodes:
            return {'exists': False}
        
        node_data = self.nodes[entity_lower]
        emotions = dict(self.entity_emotions.get(entity_lower, {}))
        
        # Calculate average intensity
        avg_intensity = 0
        if node_data['mention_count'] > 0:
            avg_intensity = node_data['total_intensity'] / node_data['mention_count']
        
        # Determine dominant emotion
        dominant_emotion = None
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        return {
            'exists': True,
            'entity': entity_lower,
            'mention_count': node_data['mention_count'],
            'avg_intensity': round(avg_intensity, 3),
            'crisis_rate': node_data['crisis_count'] / max(1, node_data['mention_count']),
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'topics': list(self.entity_topics.get(entity_lower, [])),
            'first_seen': node_data. get('first_seen'),
            'last_seen': node_data.get('last_seen')
        }
    
    def find_bridge_entities(self, entity1: str, entity2: str) -> List[str]:
        """Find entities that connect two entities."""
        e1_lower = entity1.lower()
        e2_lower = entity2.lower()
        
        e1_neighbors = set(self.edges.get(e1_lower, {}).keys())
        e2_neighbors = set(self.edges. get(e2_lower, {}).keys())
        
        bridges = e1_neighbors.intersection(e2_neighbors)
        
        # Sort by combined weight
        bridges_with_weight = []
        for bridge in bridges: 
            weight = (
                self.edges[e1_lower]. get(bridge, 0) +
                self.edges[e2_lower].get(bridge, 0)
            )
            bridges_with_weight.append((bridge, weight))
        
        bridges_with_weight.sort(key=lambda x: x[1], reverse=True)
        
        return [b[0] for b in bridges_with_weight[:5]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total_edges = sum(len(v) for v in self.edges.values()) // 2  # Bidirectional
        
        # Find top entities
        top_entities = sorted(
            self.nodes.items(),
            key=lambda x: x[1]. get('mention_count', 0),
            reverse=True
        )[:5]
        
        # Count triggers (high crisis rate)
        trigger_count = sum(
            1 for node in self.nodes.values()
            if node. get('crisis_count', 0) / max(1, node.get('mention_count', 1)) > 0.3
        )
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': total_edges,
            'total_topics': len(self.topic_entities),
            'total_updates': self.total_updates,
            'trigger_count': trigger_count,
            'top_entities': [(e, d. get('mention_count', 0)) for e, d in top_entities],
            'topics_used': list(self.topic_entities.keys()),
            'emotions_tracked': len(set(
                emotion 
                for emotions in self.entity_emotions.values() 
                for emotion in emotions. keys()
            ))
        }
    
    def force_save(self):
        """Force save to disk."""
        self._save_graph()
    
    def clear_graph(self):
        """Clear all graph data."""
        self.nodes = {}
        self.edges = defaultdict(lambda: defaultdict(float))
        self.entity_emotions = defaultdict(lambda: defaultdict(float))
        self.entity_topics = defaultdict(set)
        self.topic_entities = defaultdict(set)
        self.total_updates = 0
        self._save_graph()