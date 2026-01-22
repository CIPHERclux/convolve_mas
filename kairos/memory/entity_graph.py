"""
Entity Graph
Tracks entities, their emotional associations, and trigger status. 
Used for: 
1. Identifying trigger entities that lower safety thresholds
2. Understanding relationships between entities
3. Associative retrieval
"""
import json
import os
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class EntityGraph:
    """
    Maintains a graph of entities mentioned by the user.
    
    Tracks:
    - Mention frequency
    - Emotional associations
    - Trigger status (entities associated with crisis/trauma)
    - Co-occurrence relationships
    """
    
    def __init__(self, user_id: str, storage_path: str = "./entity_data"):
        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.graph_file = self.storage_path / f"{user_id}_entities.json"
        
        # Entity data
        self.entities: Dict[str, Dict[str, Any]] = {}
        
        # Co-occurrence tracking
        self.co_occurrences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Emotional associations
        self.emotional_associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        self._load_graph()
    
    def _load_graph(self):
        """Load entity graph from disk."""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                
                self.entities = data.get('entities', {})
                
                # Reconstruct defaultdicts
                self.co_occurrences = defaultdict(lambda: defaultdict(int))
                for e1, related in data.get('co_occurrences', {}).items():
                    for e2, count in related.items():
                        self. co_occurrences[e1][e2] = count
                
                self.emotional_associations = defaultdict(lambda: defaultdict(float))
                for entity, emotions in data.get('emotional_associations', {}).items():
                    for emotion, strength in emotions.items():
                        self.emotional_associations[entity][emotion] = strength
                
                print(f"  [EntityGraph] Loaded {len(self.entities)} entities")
            except Exception as e:
                print(f"  [EntityGraph] Warning: Could not load graph: {e}")
    
    def _save_graph(self):
        """Save entity graph to disk."""
        try:
            data = {
                'entities': self.entities,
                'co_occurrences': {k: dict(v) for k, v in self.co_occurrences. items()},
                'emotional_associations': {k: dict(v) for k, v in self.emotional_associations.items()}
            }
            with open(self.graph_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [EntityGraph] Warning: Could not save graph: {e}")
    
    def update_entity(
        self,
        entity:  str,
        emotion: str,
        is_crisis: bool = False,
        is_positive: bool = False,
        context: str = "",
        co_occurring_entities: List[str] = None
    ):
        """Update entity with new interaction data."""
        entity_lower = entity.lower()
        
        # Initialize entity if new
        if entity_lower not in self.entities:
            self.entities[entity_lower] = {
                'mention_count': 0,
                'crisis_count': 0,
                'positive_count': 0,
                'first_seen': datetime.now().isoformat(),
                'last_seen': None,
                'trigger_score': 0.0,
                'positive_score': 0.0,
                'contexts': []
            }
        
        # Update counts
        self.entities[entity_lower]['mention_count'] += 1
        self.entities[entity_lower]['last_seen'] = datetime.now().isoformat()
        
        if is_crisis:
            self. entities[entity_lower]['crisis_count'] += 1
        if is_positive:
            self. entities[entity_lower]['positive_count'] += 1
        
        # Update trigger/positive scores using exponential moving average
        mention_count = self.entities[entity_lower]['mention_count']
        crisis_rate = self.entities[entity_lower]['crisis_count'] / mention_count
        positive_rate = self.entities[entity_lower]['positive_count'] / mention_count
        
        alpha = 0.3
        self.entities[entity_lower]['trigger_score'] = (
            (1 - alpha) * self.entities[entity_lower]['trigger_score'] +
            alpha * crisis_rate
        )
        self.entities[entity_lower]['positive_score'] = (
            (1 - alpha) * self.entities[entity_lower]['positive_score'] +
            alpha * positive_rate
        )
        
        # Store context snippet
        if context:
            self.entities[entity_lower]['contexts'].append(context[: 100])
            self.entities[entity_lower]['contexts'] = self.entities[entity_lower]['contexts'][-5:]
        
        # Update emotional association
        if emotion: 
            current = self.emotional_associations[entity_lower]. get(emotion. lower(), 0)
            self.emotional_associations[entity_lower][emotion. lower()] = (
                (1 - alpha) * current + alpha * 1.0
            )
        
        # Update co-occurrences
        if co_occurring_entities:
            for other_entity in co_occurring_entities:
                other_lower = other_entity.lower()
                if other_lower != entity_lower:
                    self.co_occurrences[entity_lower][other_lower] += 1
                    self.co_occurrences[other_lower][entity_lower] += 1
        
        # Periodic save
        if mention_count % 5 == 0:
            self._save_graph()
    
    def check_trigger_entities(self, entities: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Check if any entities are triggers. 
        
        Returns list of trigger alerts with scores.
        """
        alerts = []
        
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self.entities:
                entity_data = self.entities[entity_lower]
                trigger_score = entity_data. get('trigger_score', 0)
                
                if trigger_score >= threshold:
                    alerts.append({
                        'entity': entity,
                        'trigger_score': trigger_score,
                        'crisis_count': entity_data.get('crisis_count', 0),
                        'mention_count': entity_data.get('mention_count', 0),
                        'emotions': dict(self.emotional_associations. get(entity_lower, {}))
                    })
        
        # Sort by trigger score
        alerts. sort(key=lambda x: x['trigger_score'], reverse=True)
        return alerts
    
    def get_entity_info(self, entity: str) -> Optional[Dict[str, Any]]: 
        """Get information about an entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.entities:
            return None
        
        entity_data = self.entities[entity_lower]. copy()
        entity_data['emotions'] = dict(self.emotional_associations.get(entity_lower, {}))
        entity_data['co_occurring'] = dict(self.co_occurrences.get(entity_lower, {}))
        
        return entity_data
    
    def get_related_entities(self, entity: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get entities that co-occur with the given entity."""
        entity_lower = entity.lower()
        co_occurring = self.co_occurrences.get(entity_lower, {})
        
        related = []
        for other_entity, count in sorted(co_occurring.items(), key=lambda x: x[1], reverse=True)[:limit]:
            related.append({
                'entity':  other_entity,
                'co_occurrence_count': count,
                'entity_data': self.entities. get(other_entity, {})
            })
        
        return related
    
    def get_entities_by_emotion(self, emotion: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get entities most associated with an emotion."""
        emotion_lower = emotion.lower()
        results = []
        
        for entity, emotions in self.emotional_associations.items():
            if emotion_lower in emotions: 
                results.append({
                    'entity': entity,
                    'emotion_strength': emotions[emotion_lower],
                    'entity_data': self.entities.get(entity, {})
                })
        
        results.sort(key=lambda x: x['emotion_strength'], reverse=True)
        return results[:limit]
    
    def get_all_triggers(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get all trigger entities above threshold."""
        triggers = []
        
        for entity, data in self.entities.items():
            if data.get('trigger_score', 0) >= threshold:
                triggers.append({
                    'entity': entity,
                    'trigger_score': data['trigger_score'],
                    'crisis_count': data.get('crisis_count', 0),
                    'mention_count': data.get('mention_count', 0)
                })
        
        triggers.sort(key=lambda x: x['trigger_score'], reverse=True)
        return triggers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get entity graph statistics."""
        trigger_count = sum(1 for e in self.entities.values() if e.get('trigger_score', 0) >= 0.3)
        positive_count = sum(1 for e in self.entities.values() if e.get('positive_score', 0) >= 0.3)
        
        top_triggers = self.get_all_triggers()[: 3]
        
        return {
            'total_entities': len(self.entities),
            'trigger_count':  trigger_count,
            'positive_count': positive_count,
            'top_triggers': [(t['entity'], round(t['trigger_score'], 2)) for t in top_triggers],
            'total_co_occurrences': sum(sum(v.values()) for v in self.co_occurrences.values()) // 2
        }
    
    def force_save(self):
        """Force save to disk."""
        self._save_graph()