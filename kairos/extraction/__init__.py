"""
Kairos Feature Extraction Module
Converts raw multimodal buffers into the 32-dimensional phenotype vector
"""

from .feature_engine import FeatureEngine
from .acoustic_engine import AcousticEngine
from .visual_engine import VisualEngine
from .linguistic_engine import LinguisticEngine
from .special_signals import SpecialSignalsEngine

__all__ = ['FeatureEngine', 'AcousticEngine', 'VisualEngine', 'LinguisticEngine', 'SpecialSignalsEngine']