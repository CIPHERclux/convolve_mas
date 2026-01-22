"""
Kairos Configuration - Enhanced Version
Includes all Phase 1, 2, 3 parameters
"""
import os

# API Configuration
GROQ_API_KEY = os. environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"

# Vector Dimensions
SEMANTIC_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2
BIOMARKER_DIM = 32  # 8 acoustic + 8 visual + 8 linguistic + 8 special
RELIABILITY_DIM = 32
TRAJECTORY_DIM = 160  # BIOMARKER_DIM * 5 (trajectory window)

# Qdrant Configuration
QDRANT_PATH = "./kairos_memory"
COLLECTION_NAME = "kairos_episodic"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Memory Retrieval Configuration
EPISODIC_TOP_K = 5  # Number of memories to retrieve
TRAJECTORY_WINDOW = 5  # Number of turns for trajectory matching

# Phase 1: RRF Configuration
RRF_K = 60  # RRF ranking constant (standard value)
SPARSE_CRISIS_BOOST = 3.0  # Multiplier for crisis terms in sparse search

# Phase 2: Intervention Tracking
MIN_SUCCESS_SCORE = 0.1  # Minimum success score to consider intervention effective
INTERVENTION_HISTORY_LIMIT = 100  # Max interventions to store per user

# Phase 3: Binary Quantization
USE_BINARY_QUANTIZATION = True  # Enable for semantic vectors
QUANTIZATION_RESCORE = True  # Rescore with original vectors
QUANTIZATION_OVERSAMPLING = 2.0  # Oversampling factor for recall


# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 0.5  # Minimum audio length in seconds

# Video Configuration
VIDEO_FPS = 5  # Target frames per second
VIDEO_FRAME_SIZE = (224, 224)  # Target frame size (width, height)


# Safety Configuration
MIN_SNR_THRESHOLD = 10  # Minimum Signal-to-Noise Ratio in dB
MIN_FACE_CONFIDENCE = 0.5  # Minimum face detection confidence

# Crisis detection thresholds
CRISIS_THRESHOLD_BASE = 0.7  # Base threshold for crisis detection
CRISIS_THRESHOLD_WITH_TRIGGER = 0.5  # Lowered threshold when triggers present

# Feature Extraction Weights
STATIC_WEIGHTS = {
    "acoustic": 0.85,  # Weight for acoustic features when available
    "visual": 0.80,    # Weight for visual features when available
    "linguistic": 1.0,  # Linguistic always available
    "latency": 0.7     # Weight for response latency feature
}

# Linguistic Analysis Configuration
ABSOLUTIST_WORDS = [
    'always', 'never', 'completely', 'totally', 'absolutely',
    'entirely', 'constantly', 'nothing', 'everything', 'everyone',
    'nobody', 'forever', 'impossible', 'definitely', 'certainly',
    'all', 'none', 'must', 'every', 'only', 'just'
]

FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'i mean', 'kind of',
    'sort of', 'basically', 'actually', 'literally', 'honestly',
    'right', 'so', 'well', 'anyway', 'whatever'
]


# Entity Graph Configuration
TRIGGER_THRESHOLD = 0.6  # Score threshold to classify as trigger
SUPPORT_THRESHOLD = 0.6  # Score threshold to classify as support
MIN_MENTIONS_FOR_CLASSIFICATION = 3  # Minimum mentions before auto-classification

# Entity aliases (default mappings)
ENTITY_ALIASES = {
    'mom': 'mother', 'mum': 'mother', 'mommy': 'mother', 'mama': 'mother',
    'dad': 'father', 'daddy': 'father', 'papa': 'father',
    'bro': 'brother', 'sis': 'sister',
    'bf': 'boyfriend', 'gf': 'girlfriend', 'bestie': 'best_friend',
    'boss': 'manager', 'supervisor': 'manager',
}

# Trajectory Matcher Configuration

TRAJECTORY_PATTERN_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for pattern detection
TRAJECTORY_ALERT_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for crisis alert


# ColBERT Configuration

COLBERT_MAX_TOKENS = 64  # Maximum tokens per document
COLBERT_TOKEN_DIM = 384  # Token embedding dimension


# User Baselines Configuration

USER_BASELINES_DIR = "./user_baselines"


# Session Configuration
MAX_CONVERSATION_HISTORY = 20  # Maximum turns to keep in memory
MAX_CROSS_SESSION_HISTORY = 50  # Maximum sessions to store