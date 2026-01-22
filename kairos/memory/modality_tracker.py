"""
Modality Tracker
Tracks user's history across different input modalities (text, audio, video).

This enables: 
- Understanding how user's voice/face signals change over time
- Comparing text-only vs multimodal interactions
- Detecting patterns specific to modalities
- Providing context about user's multimodal history
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np


class ModalityTracker: 
    """
    Tracks user interactions across different modalities.
    
    Features:
    - Per-modality interaction history
    - Biomarker progression tracking
    - Modality-specific pattern detection
    - Cross-modality comparison
    - Session modality summary
    """
    
    def __init__(self, user_id: str, storage_path: str = "./modality_data"):
        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.tracker_file = self.storage_path / f"{user_id}_modality. json"
        
        # Per-modality tracking
        self.modality_history: Dict[str, List[Dict[str, Any]]] = {
            'text': [],
            'audio': [],
            'video': []
        }
        
        # Aggregated statistics
        self.modality_stats: Dict[str, Dict[str, Any]] = {
            'text': self._init_modality_stats(),
            'audio': self._init_modality_stats(),
            'video': self._init_modality_stats()
        }
        
        # Biomarker baselines per modality
        self.modality_baselines: Dict[str, Dict[str, float]] = {
            'audio': {},
            'video': {}
        }
        
        # Session tracking
        self.session_modalities: List[str] = []
        self.session_start = datetime.now()
        
        # Load existing data
        self._load_data()
    
    def _init_modality_stats(self) -> Dict[str, Any]: 
        """Initialize statistics structure for a modality."""
        return {
            'count': 0,
            'total_intensity': 0.0,
            'crisis_count': 0,
            'emotions': defaultdict(int),
            'avg_word_count': 0.0,
            'total_word_count': 0,
            'first_use':  None,
            'last_use':  None,
            # Audio/video specific
            'avg_jitter': 0.0,
            'avg_shimmer': 0.0,
            'total_jitter': 0.0,
            'total_shimmer': 0.0,
            'crying_detected_count': 0,
            'laughter_detected_count': 0,
        }
    
    def _load_data(self):
        """Load modality data from disk."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                
                # Load history (keep last 100 per modality)
                for modality in ['text', 'audio', 'video']:
                    self.modality_history[modality] = data.get('history', {}).get(modality, [])[-100:]
                
                # Load stats
                for modality in ['text', 'audio', 'video']:
                    saved_stats = data.get('stats', {}).get(modality, {})
                    if saved_stats:
                        self.modality_stats[modality]. update(saved_stats)
                        # Reconstruct defaultdict for emotions
                        if 'emotions' in saved_stats: 
                            self.modality_stats[modality]['emotions'] = defaultdict(int, saved_stats['emotions'])
                
                # Load baselines
                self.modality_baselines = data.get('baselines', {'audio': {}, 'video':  {}})
                
                print(f"  [ModalityTracker] Loaded:  text={len(self.modality_history['text'])}, "
                      f"audio={len(self.modality_history['audio'])}, video={len(self.modality_history['video'])}")
                
            except Exception as e:
                print(f"  [ModalityTracker] Warning: Could not load data: {e}")
    
    def _save_data(self):
        """Save modality data to disk."""
        try:
            # Convert defaultdicts to regular dicts for JSON
            stats_for_save = {}
            for modality, stats in self.modality_stats.items():
                stats_copy = dict(stats)
                if isinstance(stats_copy. get('emotions'), defaultdict):
                    stats_copy['emotions'] = dict(stats_copy['emotions'])
                stats_for_save[modality] = stats_copy
            
            data = {
                'user_id': self.user_id,
                'history': {
                    modality: history[-100:]  # Keep last 100
                    for modality, history in self.modality_history.items()
                },
                'stats': stats_for_save,
                'baselines': self.modality_baselines,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.tracker_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"  [ModalityTracker] Warning: Could not save data: {e}")
    
    def record_interaction(
        self,
        modality:  str,
        text: str,
        emotion: str,
        emotional_intensity: float,
        is_crisis: bool,
        biomarker:  List[float] = None,
        turn_number: int = 0,
        feature_insights: str = None
    ):
        """
        Record an interaction for a specific modality.
        
        Args:
            modality: 'text', 'audio', or 'video'
            text:  User's message
            emotion: Detected emotion
            emotional_intensity:  Intensity score
            is_crisis:  Crisis flag
            biomarker:  Biomarker vector (for audio/video)
            turn_number: Turn number
            feature_insights: Summary of notable features
        """
        modality = modality.lower()
        if modality not in self.modality_history:
            modality = 'text'  # Default
        
        # Track session modality
        if modality not in self.session_modalities:
            self.session_modalities.append(modality)
        
        word_count = len(text.split()) if text else 0
        
        # Create interaction record
        interaction = {
            'turn_number': turn_number,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'emotional_intensity': emotional_intensity,
            'is_crisis': is_crisis,
            'word_count': word_count,
            'text_preview': text[:100] if text else '',
            'feature_insights': feature_insights
        }
        
        # Add biomarker data for audio/video
        if modality in ['audio', 'video'] and biomarker:
            interaction['biomarker_summary'] = {
                'jitter': biomarker[0] if len(biomarker) > 0 else 0,
                'shimmer': biomarker[1] if len(biomarker) > 1 else 0,
                'f0_variance': biomarker[2] if len(biomarker) > 2 else 0,
                'sentiment':  biomarker[22] if len(biomarker) > 22 else 0,
                'laughter':  biomarker[24] if len(biomarker) > 24 else 0,
                'crying': biomarker[25] if len(biomarker) > 25 else 0,
            }
        
        # Store interaction
        self.modality_history[modality].append(interaction)
        
        # Update statistics
        stats = self.modality_stats[modality]
        stats['count'] += 1
        stats['total_intensity'] += emotional_intensity
        stats['total_word_count'] += word_count
        stats['avg_word_count'] = stats['total_word_count'] / stats['count']
        
        if is_crisis:
            stats['crisis_count'] += 1
        
        if emotion: 
            stats['emotions'][emotion. lower()] += 1
        
        if not stats['first_use']:
            stats['first_use'] = datetime. now().isoformat()
        stats['last_use'] = datetime. now().isoformat()
        
        # Audio/video specific stats
        if modality in ['audio', 'video'] and biomarker:
            if len(biomarker) > 0:
                stats['total_jitter'] += max(0, biomarker[0])
                stats['avg_jitter'] = stats['total_jitter'] / stats['count']
            
            if len(biomarker) > 1:
                stats['total_shimmer'] += max(0, biomarker[1])
                stats['avg_shimmer'] = stats['total_shimmer'] / stats['count']
            
            if len(biomarker) > 24 and biomarker[24] > 0.3:
                stats['laughter_detected_count'] += 1
            
            if len(biomarker) > 25 and biomarker[25] > 0.3:
                stats['crying_detected_count'] += 1
            
            # Update baselines
            self._update_baseline(modality, biomarker)
        
        # Periodic save
        if stats['count'] % 5 == 0:
            self._save_data()
    
    def _update_baseline(self, modality: str, biomarker: List[float]):
        """Update running baseline for modality."""
        if modality not in self.modality_baselines:
            self.modality_baselines[modality] = {}
        
        baseline = self.modality_baselines[modality]
        alpha = 0.1  # Slow adaptation
        
        features = {
            'jitter': biomarker[0] if len(biomarker) > 0 else 0,
            'shimmer': biomarker[1] if len(biomarker) > 1 else 0,
            'f0_variance': biomarker[2] if len(biomarker) > 2 else 0,
        }
        
        for feature, value in features.items():
            if feature not in baseline:
                baseline[feature] = value
            else:
                baseline[feature] = (1 - alpha) * baseline[feature] + alpha * value
    
    def get_modality_progression(
        self,
        modality:  str,
        n_recent: int = 10
    ) -> Dict[str, Any]:
        """
        Get progression of biomarker signals over time for a modality.
        
        Args:
            modality: 'audio' or 'video'
            n_recent: Number of recent interactions to analyze
            
        Returns:
            Dict with progression data and trends
        """
        modality = modality.lower()
        history = self.modality_history. get(modality, [])
        
        if not history: 
            return {
                'has_data': False,
                'modality': modality,
                'message': f"No {modality} interactions recorded yet"
            }
        
        recent = history[-n_recent:]
        
        # Extract biomarker progressions
        progressions = {
            'jitter': [],
            'shimmer': [],
            'sentiment': [],
            'emotional_intensity': [],
            'emotions': [],
            'timestamps': []
        }
        
        for interaction in recent:
            progressions['emotional_intensity'].append(interaction. get('emotional_intensity', 0))
            progressions['emotions']. append(interaction.get('emotion', 'unknown'))
            progressions['timestamps'].append(interaction.get('timestamp', ''))
            
            bio_summary = interaction.get('biomarker_summary', {})
            progressions['jitter'].append(bio_summary.get('jitter', 0))
            progressions['shimmer'].append(bio_summary.get('shimmer', 0))
            progressions['sentiment']. append(bio_summary.get('sentiment', 0))
        
        # Calculate trends
        trends = {}
        for feature in ['jitter', 'shimmer', 'sentiment', 'emotional_intensity']:
            values = progressions[feature]
            if len(values) >= 2:
                trend = values[-1] - values[0]
                if trend > 0.1:
                    trends[feature] = 'INCREASING'
                elif trend < -0.1:
                    trends[feature] = 'DECREASING'
                else:
                    trends[feature] = 'STABLE'
        
        # Compare to baseline
        baseline_comparison = {}
        if modality in self.modality_baselines and recent:
            baseline = self.modality_baselines[modality]
            latest_bio = recent[-1].get('biomarker_summary', {})
            
            for feature in ['jitter', 'shimmer']: 
                if feature in baseline and feature in latest_bio:
                    diff = latest_bio[feature] - baseline[feature]
                    if diff > 0.15:
                        baseline_comparison[feature] = 'ABOVE_BASELINE'
                    elif diff < -0.15:
                        baseline_comparison[feature] = 'BELOW_BASELINE'
                    else:
                        baseline_comparison[feature] = 'AT_BASELINE'
        
        return {
            'has_data': True,
            'modality': modality,
            'interactions_analyzed': len(recent),
            'total_interactions': len(history),
            'progressions': progressions,
            'trends': trends,
            'baseline_comparison':  baseline_comparison,
            'stats': {
                'avg_intensity': np.mean(progressions['emotional_intensity']),
                'max_intensity': max(progressions['emotional_intensity']) if progressions['emotional_intensity'] else 0,
                'avg_jitter': np.mean(progressions['jitter']) if progressions['jitter'] else 0,
                'avg_shimmer': np.mean(progressions['shimmer']) if progressions['shimmer'] else 0,
            },
            'dominant_emotion': max(set(progressions['emotions']), key=progressions['emotions'].count) if progressions['emotions'] else None
        }
    
    def get_modality_progression_for_llm(self, modality: str) -> str:
        """Get formatted progression for LLM context."""
        progression = self.get_modality_progression(modality)
        
        if not progression['has_data']:
            return ""
        
        lines = [f"=== {modality. upper()} HISTORY ==="]
        lines.append(f"Total {modality} interactions: {progression['total_interactions']}")
        
        # Trends
        if progression. get('trends'):
            lines. append("Recent trends:")
            for feature, trend in progression['trends'].items():
                lines.append(f"  • {feature}: {trend}")
        
        # Baseline comparison
        if progression.get('baseline_comparison'):
            lines.append("Compared to baseline:")
            for feature, comparison in progression['baseline_comparison'].items():
                lines.append(f"  • {feature}: {comparison}")
        
        # Stats
        stats = progression. get('stats', {})
        if stats:
            lines.append(f"Average emotional intensity: {stats.get('avg_intensity', 0):.2f}")
            if modality in ['audio', 'video']:
                lines.append(f"Average voice tremor (jitter): {stats.get('avg_jitter', 0):.2f}")
        
        return "\n".join(lines)
    
    def get_cross_modality_comparison(self) -> Dict[str, Any]:
        """Compare user's behavior across modalities."""
        comparison = {}
        
        for modality in ['text', 'audio', 'video']:
            stats = self.modality_stats[modality]
            if stats['count'] > 0:
                avg_intensity = stats['total_intensity'] / stats['count']
                crisis_rate = stats['crisis_count'] / stats['count']
                
                # Find dominant emotion
                emotions = stats. get('emotions', {})
                dominant_emotion = max(emotions. items(), key=lambda x: x[1])[0] if emotions else None
                
                comparison[modality] = {
                    'interaction_count': stats['count'],
                    'avg_intensity': round(avg_intensity, 3),
                    'crisis_rate': round(crisis_rate, 3),
                    'dominant_emotion': dominant_emotion,
                    'avg_word_count': round(stats. get('avg_word_count', 0), 1),
                }
                
                if modality in ['audio', 'video']: 
                    comparison[modality]['avg_jitter'] = round(stats.get('avg_jitter', 0), 3)
                    comparison[modality]['avg_shimmer'] = round(stats. get('avg_shimmer', 0), 3)
                    comparison[modality]['crying_detected'] = stats.get('crying_detected_count', 0)
                    comparison[modality]['laughter_detected'] = stats.get('laughter_detected_count', 0)
        
        # Find which modality shows most distress
        most_distressed_modality = None
        max_crisis_rate = 0
        for modality, data in comparison.items():
            if data['crisis_rate'] > max_crisis_rate:
                max_crisis_rate = data['crisis_rate']
                most_distressed_modality = modality
        
        return {
            'modalities': comparison,
            'most_used_modality': max(comparison.items(), key=lambda x: x[1]['interaction_count'])[0] if comparison else None,
            'most_distressed_modality':  most_distressed_modality,
            'total_interactions': sum(d['interaction_count'] for d in comparison.values())
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of modalities used in current session."""
        return {
            'session_start': self.session_start. isoformat(),
            'modalities_used': self.session_modalities,
            'multimodal_session': len(self.session_modalities) > 1,
            'primary_modality': self.session_modalities[0] if self. session_modalities else 'text'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive modality statistics."""
        stats = {}
        
        for modality in ['text', 'audio', 'video']:
            mod_stats = self.modality_stats[modality]
            if mod_stats['count'] > 0:
                stats[modality] = {
                    'count': mod_stats['count'],
                    'avg_intensity': round(mod_stats['total_intensity'] / mod_stats['count'], 3),
                    'crisis_count': mod_stats['crisis_count'],
                    'crisis_rate': round(mod_stats['crisis_count'] / mod_stats['count'], 3),
                    'avg_word_count': round(mod_stats.get('avg_word_count', 0), 1),
                    'first_use': mod_stats. get('first_use'),
                    'last_use':  mod_stats.get('last_use'),
                    'dominant_emotions': dict(sorted(
                        mod_stats. get('emotions', {}).items(),
                        key=lambda x:  x[1],
                        reverse=True
                    )[:3])
                }
                
                if modality in ['audio', 'video']:
                    stats[modality]['avg_jitter'] = round(mod_stats.get('avg_jitter', 0), 3)
                    stats[modality]['avg_shimmer'] = round(mod_stats.get('avg_shimmer', 0), 3)
                    stats[modality]['crying_detected'] = mod_stats. get('crying_detected_count', 0)
                    stats[modality]['laughter_detected'] = mod_stats.get('laughter_detected_count', 0)
        
        stats['total_interactions'] = sum(
            self.modality_stats[m]['count'] for m in ['text', 'audio', 'video']
        )
        stats['session_modalities'] = self.session_modalities
        
        return stats
    
    def force_save(self):
        """Force save to disk."""
        self._save_data()
    
    def reset_session(self):
        """Reset session tracking for new session."""
        self.session_modalities = []
        self.session_start = datetime.now()