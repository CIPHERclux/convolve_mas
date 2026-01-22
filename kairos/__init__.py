"""
Kairos - Mental Health Support System

Enhanced with Phase 1, 2, 3 improvements:
- Hybrid search with RRF and sparse vectors
- Entity graph for trigger detection
- Intervention outcome tracking
- Trajectory pattern matching
- ColBERT late interaction
"""

__version__ = "2. 0.0"

from kairos.orchestration.orchestrator import KairosOrchestrator
from kairos.extraction.feature_engine import FeatureEngine
from kairos.memory.memory_controller import MemoryController
from kairos.memory.baseline_manager import BaselineManager

__all__ = [
    'KairosOrchestrator',
    'FeatureEngine', 
    'MemoryController',
    'BaselineManager'
]