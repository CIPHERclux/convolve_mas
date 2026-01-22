import os
import sys
import shutil
import zipfile
import uuid
from datetime import datetime

# Import Colab files module at the top level (CRITICAL for upload to work)
from google.colab import files as colab_files

print("=" * 60)
print("KAIROS MENTAL SUPPORT SYSTEM - SETUP")
print("=" * 60)

# CLEANUP PREVIOUS INSTALLATION
print("\nCleaning up previous installation...")

workspace_dir = "/content/kairos_workspace"
qdrant_data_dir = "/content/qdrant_data"

if os.path.exists(workspace_dir):
    shutil.rmtree(workspace_dir)
    print(f"  Removed:  {workspace_dir}")

if os.path.exists(qdrant_data_dir):
    shutil.rmtree(qdrant_data_dir)
    print(f"  Removed:  {qdrant_data_dir}")

# Clean up old zip files and media files
for f in os.listdir("/content"):
    if f.endswith(". zip") or f.endswith(".m4a") or f.endswith(".mp3") or f.endswith(".wav") or f.endswith(".mp4"):
        try:
            os.remove(os.path.join("/content", f))
        except:
            pass

modules_to_remove = [key for key in list(sys.modules.keys()) if 'kairos' in key. lower() or 'config' in key.lower()]
for mod in modules_to_remove:
    del sys.modules[mod]

os.chdir("/content")
print("Cleanup complete!\n")


# UPLOAD REPOSITORY ZIP and extraction
print("=" * 60)
print("Please upload your Kairos repository ZIP file...")
print("=" * 60)

uploaded = colab_files.upload()
zip_filename = list(uploaded.keys())[0]
print(f"\nReceived:  {zip_filename}")

print("\nExtracting repository...")
os.makedirs(workspace_dir, exist_ok=True)

zip_path = os.path.join("/content", zip_filename)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(workspace_dir)

def find_repo_root(base_path):
    # Walk the directory tree
    for root, dirs, files_list in os.walk(base_path):
        # 1. CRITICAL: Ignore Mac metadata folders
        if "__MACOSX" in root:
            continue

        # 2. Verify it's the real code folder
        # We check for BOTH main.py and config.py to ensure it's not a ghost folder
        if 'main.py' in files_list and 'config.py' in files_list:
            return root

    # Fallback
    return base_path

repo_root = find_repo_root(workspace_dir)
print(f"✅ Repository root found: {repo_root}")

os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("\n" + "=" * 60)
print("INSTALLING DEPENDENCIES...")
print("=" * 60)

# Install main dependencies
!pip install -q qdrant-client>=1.7.0 sentence-transformers>=2.2.0 \
    faster-whisper>=0.10.0 librosa>=0.10.0 \
    moviepy>=1.0.3 groq>=0.4.0 nltk>=3.8.0 numpy>=1.24.0 \
    scipy>=1.10.0 tqdm>=4.65.0 pydub>=0.25.0

# Install mediapipe with specific version that works in Colab
!pip install -q mediapipe==0.10.14

# Download Haar Cascade for fallback face detection
import urllib.request
import os

cascade_dir = "/tmp"
cascade_file = f"{cascade_dir}/haarcascade_frontalface_default. xml"
if not os.path.exists(cascade_file):
    print("Downloading Haar Cascade for face detection...")
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(cascade_url, cascade_file)
    print(f"  Downloaded to {cascade_file}")

print("\nDownloading NLTK data...")
import nltk

# Download all required NLTK resources
nltk_resources = [
    'punkt',
    'punkt_tab',  # NEW - required for newer NLTK versions
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',  # NEW - English specific
    'vader_lexicon',
    'stopwords',
    'wordnet'
]

for resource in nltk_resources:
    try:
        nltk.download(resource, quiet=True)
        print(f"  ✓ {resource}")
    except Exception as e:
        print(f"  ⚠ {resource} (optional): {e}")

print("NLTK data downloaded!")
# Test MediaPipe
print("\nTesting MediaPipe...")
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        print("  ✅ MediaPipe FaceMesh is available")
    else:
        print("  ⚠️ MediaPipe installed but FaceMesh not found in expected location")
        print("  Will use OpenCV fallback for visual features")
except Exception as e:
    print(f"  ⚠️ MediaPipe test failed: {e}")
    print("  Will use OpenCV fallback for visual features")

# ============================================================
# INITIALIZE SYSTEM - CORRECTED VERSION
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING KAIROS...")
print("=" * 60)

# Clear and reimport
modules_to_remove = [key for key in list(sys.modules.keys()) if 'kairos' in key. lower() or 'config' in key.lower()]
for mod in modules_to_remove:
    del sys.modules[mod]

import config
from config import GROQ_API_KEY, QDRANT_PATH

# Get API key
api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
if not api_key:
    api_key = input("Please enter your Groq API key:  ").strip()
    os.environ["GROQ_API_KEY"] = api_key

from kairos.orchestration. orchestrator import KairosOrchestrator
from kairos.memory. qdrant_manager import QdrantManager
from kairos.memory.memory_controller import MemoryController
from kairos.memory.baseline_manager import BaselineManager
from kairos.extraction.feature_engine import FeatureEngine
from kairos.ingestion.file_loader import FileLoader

session_id = str(uuid.uuid4())[:8]
user_id = "default_user"

print(f"Session ID: {session_id}")

# 1. BaselineManager needs user_id
baseline_manager = BaselineManager(user_id)

# 2. MemoryController needs qdrant_path (string) and user_id
memory_controller = MemoryController(qdrant_path=QDRANT_PATH, user_id=user_id)

# 3. FeatureEngine needs baseline_manager
feature_engine = FeatureEngine(baseline_manager)

# 4. FileLoader has no dependencies
file_loader = FileLoader()


# 5. Orchestrator ties everything together
orchestrator = KairosOrchestrator(
    feature_engine=feature_engine,
    memory_controller=memory_controller,
    api_key=api_key,
    user_id=user_id,
    session_id=session_id
)

print("\n✅ System initialized!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_uploaded_filepath(uploaded_dict):
    """Get the actual filepath of an uploaded file."""
    if not uploaded_dict:
        return None

    filename = list(uploaded_dict.keys())[0]
    file_content = uploaded_dict[filename]

    clean_filename = filename.replace(" ", "_")
    filepath = f"/content/{clean_filename}"

    with open(filepath, 'wb') as f:
        f.write(file_content)

    print(f"  📁 File saved to: {filepath}")
    print(f"  📊 File size: {len(file_content)} bytes")

    if os.path.exists(filepath):
        print(f"  ✅ File verified at: {filepath}")
        return filepath
    else:
        print(f"  ❌ File not found at: {filepath}")
        return None

def print_memory_debug(debug_info):
    """Display detailed memory/Qdrant debug information."""
    print("\n" + "─" * 60)
    print("🗄️ MEMORY & QDRANT DEBUG")
    print("─" * 60)

    memory_debug = debug_info.get('memory_debug', {})

    if not memory_debug:
        print("  No memory debug info available")
        return

    # Query info
    print(f"\n  📊 QUERY INFO:")
    print(f"     Query types used: {memory_debug.get('query_type', [])}")
    print(f"     Total results: {memory_debug.get('total_searched', 0)}")

    # Semantic matches
    semantic = memory_debug.get('semantic_matches', [])
    print(f"\n  🔍 SEMANTIC MATCHES ({len(semantic)}):")
    if semantic:
        for i, match in enumerate(semantic[:3]):
            print(f"     [{i+1}] Score: {match. get('score', 0):.3f}")
            print(f"         Text: \"{match.get('text', '')[:60]}... \"")
            print(f"         Emotion: {match.get('emotion', 'unknown')}")
    else:
        print("     No semantic matches found")

    # Emotional matches
    emotional = memory_debug.get('emotional_matches', [])
    print(f"\n  💭 EMOTIONAL MATCHES ({len(emotional)}):")
    if emotional:
        for i, match in enumerate(emotional[:3]):
            print(f"     [{i+1}] Score:  {match.get('score', 0):.3f}")
            print(f"         Text: \"{match.get('text', '')[:60]}...\"")
            print(f"         Emotion:  {match.get('emotion', 'unknown')}")
    else:
        print("     No emotional matches found")

    # Cross-session info
    cross = memory_debug.get('cross_session_info', {})
    print(f"\n  🔄 CROSS-SESSION INFO:")
    print(f"     Total previous sessions: {cross.get('total_sessions', 0)}")
    print(f"     Common topics: {cross.get('common_topics', [])}")
    print(f"     Crisis history count: {cross.get('crisis_count', 0)}")
    print(f"     Positive moments count: {cross.get('positive_count', 0)}")

    print("─" * 60)


def print_biomarker_delta(orchestrator):
    """Display biomarker delta/trend information."""
    print("\n" + "─" * 60)
    print("📈 BIOMARKER TRENDS & DELTAS")
    print("─" * 60)

    if not hasattr(orchestrator, 'biomarker_tracker'):
        print("  Biomarker tracker not available")
        return

    tracker = orchestrator.biomarker_tracker

    # Delta from last turn
    delta = tracker. get_delta()
    if delta. get('has_delta'):
        print(f"\n  📊 CHANGES FROM LAST TURN:")

        changes = delta.get('significant_changes', [])
        if changes:
            for change in changes[:5]: 
                # FIX: Use 'direction' with fallback, and 'change' not 'delta'
                change_value = change.get('change', 0)
                direction = change. get('direction', 'increased' if change_value > 0 else 'decreased')
                direction_icon = "↑" if direction == 'increased' else "↓"
                feature = change.get('feature', 'unknown')
                interpretation = change.get('interpretation', '')
                
                # FIX: Use 'change' key instead of 'delta'
                print(f"     {direction_icon} {feature}:  {change_value:+.3f}")
                if interpretation:
                    print(f"        └─ {interpretation}")
        else:
            print("     No significant changes detected")

        # FIX: These keys don't exist in the return dict, use what's available
        print(f"\n     Delta magnitude: {delta.get('delta_magnitude', 0):.3f}")
        print(f"     Total changes detected: {delta.get('total_changes', 0)}")
    else:
        print(f"\n  {delta.get('message', 'No delta available')}")

    # Session trend
    try:
        session_trend = tracker.get_session_trend()
        if session_trend.get('has_trend'):
            print(f"\n  📈 SESSION TREND:")
            print(f"     Distress trend: {session_trend.get('distress_trend', 'unknown')}")
            print(f"     Turns analyzed: {session_trend.get('turns_analyzed', 0)}")
            
            # Show feature trends if available
            feature_trends = session_trend.get('feature_trends', {})
            if feature_trends:
                print(f"     Feature trends:")
                for feature, trend_data in list(feature_trends.items())[:3]:
                    direction = trend_data.get('direction', 'unknown')
                    print(f"        • {feature}: {direction}")
    except Exception as e:
        print(f"  Session trend not available: {e}")

    print("─" * 60)


# ============================================================
# HELPER FUNCTIONS (FIXED VISUALS & MENTAL STATE)
# ============================================================

def get_uploaded_filepath(uploaded_dict):
    """Get the actual filepath of an uploaded file."""
    if not uploaded_dict:
        return None
    filename = list(uploaded_dict.keys())[0]
    file_content = uploaded_dict[filename]
    clean_filename = filename.replace(" ", "_")
    filepath = f"/content/{clean_filename}"
    with open(filepath, 'wb') as f:
        f.write(file_content)
    print(f"  📁 File saved to: {filepath}")
    if os.path.exists(filepath):
        return filepath
    return None

def print_biomarker_features(debug_info):
    """Display extracted features using the visual dict from Orchestrator"""
    print("\n" + "─" * 60)
    print("🎤 EXTRACTED BIOMARKER FEATURES")
    print("─" * 60)

    # 1. Get feature extraction block from debug info
    fe = debug_info.get('feature_extraction', {})
    
    # 2. Get input modality
    modality = fe.get('modality', 'text')
    print(f"\n  📥 Input Modality: {modality.upper()}")

    # 3. Acoustic Features (Use the insights string if vector is truncated)
    insights = fe.get('feature_insights', '')
    biomarker_preview = fe.get('biomarker_first_8', [])
    
    if modality in ['audio', 'video'] and biomarker_preview:
        print(f"\n  🎤 ACOUSTIC SUMMARY:")
        if insights:
            # Split insights by semicolon for cleaner list
            for insight in insights.split(';'):
                if insight.strip():
                    print(f"     • {insight.strip()}")
        else:
            print("     No specific acoustic anomalies detected.")
            
        # Print raw values for first 8 if available
        print(f"\n     Raw Vector (first 8): {[round(x, 2) for x in biomarker_preview]}")
    else:
        print("\n  🎤 ACOUSTIC FEATURES: Not available (text-only)")

    # 4. Visual Features (Use the dictionary explicitly passed by Orchestrator)
    # The Orchestrator passes 'visual_features' dict directly in debug_info['feature_extraction']
    visual_dict = fe.get('visual_features', {})
    
    if modality == 'video' and visual_dict:
        print(f"\n  👁️ VISUAL FEATURES (Computer Vision):")
        
        # Define interpretation thresholds
        thresholds = {
            'masking_score': (0.6, "Possible masking/fake expression"),
            'brow_tension': (0.5, "High concern/worry"),
            'gaze_aversion': (0.5, "Avoidance/discomfort"),
            'facial_dynamism': (0.4, "High engagement"),
            'stare_duration': (0.5, "Dissociation indicator")
        }

        for k, v in visual_dict.items():
            # Create a visual bar
            val = float(v)
            bar_len = min(int(abs(val) * 10), 10)
            bar = "▓" * bar_len + "░" * (10 - bar_len)
            
            # Check for meaning
            meaning = ""
            thresh, msg = thresholds.get(k, (1.0, ""))
            if abs(val) >= thresh:
                meaning = f"→ {msg}"
                
            print(f"     • {k.replace('_', ' ').title()}:")
            print(f"       [{bar}] {val:.2f} {meaning}")
            
        # Check for notable visual alerts
        notable = fe.get('notable_visual', [])
        if notable:
            print(f"\n     ⚠️ NOTABLE ALERTS:")
            for note in notable:
                print(f"       • {note}")
                
    elif modality == 'video':
        print("\n  👁️ VISUAL FEATURES: Processing... (No face detected or low confidence)")
    else:
        print("\n  👁️ VISUAL FEATURES: Not available")

    print("─" * 60)

def print_mental_state(orchestrator):
    """Display current mental state assessment (FIXED KEY ACCESS)"""
    print("\n" + "─" * 60)
    print("📊 MENTAL STATE ASSESSMENT")
    print("─" * 60)

    if not orchestrator.conversation_history:
        print("  No data yet")
        return

    # Get the last turn directly
    latest = orchestrator.conversation_history[-1]
    
    # 1. Attempt to get data from flat structure (Standard Kairos Orchestrator)
    emotion = latest.get('emotion', 'unknown')
    intensity = latest.get('intensity', 0.5)
    risk = latest.get('risk_level', 'LOW') # Note: Orchestrator usually saves this as 'risk_level'
    if not risk: risk = 'LOW' # Fallback
    
    is_crisis = latest.get('is_crisis', False)
    modality = latest.get('modality', 'text')

    # 2. Display
    print(f"\n  🎭 Emotion: {str(emotion).upper()}")

    # Intensity bar
    try:
        intensity_val = float(intensity) if intensity is not None else 0.5
        intensity_pct = int(intensity_val * 100)
        bar_filled = int(intensity_val * 10)
        bar = '█' * bar_filled + '░' * (10 - bar_filled)
        print(f"  📈 Intensity: [{bar}] {intensity_pct}%")
    except:
        print(f"  📈 Intensity: {intensity}")

    # Risk level
    risk_icons = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
    icon = risk_icons.get(risk, '⚪')
    print(f"  {icon} Risk: {risk}")

    if is_crisis:
        print(f"  🚨 CRISIS FLAG ACTIVE")

    # Session trend
    if len(orchestrator.conversation_history) >= 2:
        recent = orchestrator.conversation_history[-3:]
        trend = " → ".join([str(h.get('emotion', '?')).upper() for h in recent])
        print(f"\n  📈 Recent Trend: {trend}")

    print("─" * 60)

def print_full_debug(debug_info, orchestrator):
    """Print all debug info"""
    if not debug_info: return

    # 1. Visual/Audio Features
    print_biomarker_features(debug_info)
    
    # 2. Memory
    mem = debug_info.get('memory', {})
    print("\n" + "─" * 60)
    print("🗄️ MEMORY & CONTEXT")
    print("─" * 60)
    print(f"  User Facts: {list(mem.get('user_facts', {}).keys())}")
    print(f"  Anchors: {len(mem.get('anchor_memories', []))}")
    
    # 3. Routing/Reasoning
    print("\n" + "─" * 60)
    print("🧠 AI REASONING")
    print("─" * 60)
    
    routing = debug_info.get('routing', {})
    print(f"  📍 Routing: {routing.get('message_type', 'UNKNOWN')} " 
          f"({'Deep Assessment' if routing.get('needs_deep_assessment') else 'Simple'})")
    
    reasoning = debug_info.get('reasoning', {})
    
    # Check for understanding
    if 'understanding' in reasoning:
        u = reasoning['understanding']
        print(f"  🧐 Analysis: {u.get('primary_emotion', 'unknown')} "
              f"(Subtext: {u.get('what_theyre_not_saying', 'none')})")
        
    # Check for masking
    if 'signal_analysis' in reasoning:
        s = reasoning['signal_analysis']
        if s.get('is_masking'):
            print(f"  🎭 Masking Detected: {s.get('masking_type')}")
            
    # Check for response plan
    if 'response_plan' in reasoning:
        p = reasoning['response_plan']
        print(f"  📝 Plan: {p.get('response_type')} ({p.get('tone')} tone)")

    print("─" * 60)

def print_biomarker_features(debug_info):
    """Display extracted features - FIXED VERSION"""
    print("\n" + "─" * 60)
    print("🎤 EXTRACTED BIOMARKER FEATURES")
    print("─" * 60)

    # FIX: Get from feature_extraction, not feature_data
    fe = debug_info.get('feature_extraction', {})
    
    if not fe:
        print("  No feature data available")
        return

    modality = fe.get('modality', 'text')
    biomarker = fe.get('biomarker_first_8', [])
    insights = fe.get('feature_insights', '')
    
    print(f"\n  📥 Input Modality: {modality.upper()}")

    # Acoustic Features
    if modality in ['audio', 'video'] and biomarker and any(b != 0 for b in biomarker):
        print(f"\n  🎤 ACOUSTIC FEATURES:")
        
        feature_names = [
            'Jitter (voice tremor)',
            'Shimmer (instability)', 
            'F0 Variance (pitch)',
            'Loudness',
            'TEO (vocal energy)',
            'HNR (clarity)',
            'Speech Rate',
            'Pause Frequency'
        ]
        
        for i, (name, val) in enumerate(zip(feature_names, biomarker)):
            if abs(val) > 0.3:  # Only show significant values
                bar = '▓' * int(abs(val) * 10) + '░' * (10 - int(abs(val) * 10))
                status = '⚠️ HIGH' if val > 0.5 else '↑ elevated' if val > 0.3 else '↓ reduced' if val < -0.3 else 'normal'
                print(f"     • {name}: [{bar}] {val:+.2f} {status}")
        
        if insights:
            print(f"\n     💡 Summary: {insights}")
    else:
        print("\n  🎤 ACOUSTIC FEATURES: Not available (text-only)")
    
    # Visual Features
    visual_dict = fe.get('visual_features', {})
    if visual_dict:
        print(f"\n  👁️ VISUAL FEATURES:")
        for k, v in visual_dict.items():
            if abs(v) > 0.4:  # Only show significant
                bar = '▓' * int(abs(v) * 10) + '░' * (10 - int(abs(v) * 10))
                print(f"     • {k.replace('_', ' ').title()}: [{bar}] {v:.2f}")
    
    # Notable changes
    if fe.get('significant_changes'):
        print(f"\n  📊 CHANGES FROM LAST TURN:")
        for change in fe.get('significant_changes', [])[:3]:
            print(f"     • {change}")
    
    print("─" * 60)


def print_mental_state(orchestrator):
    """Display current mental state - FIXED VERSION"""
    print("\n" + "─" * 60)
    print("📊 MENTAL STATE ASSESSMENT")
    print("─" * 60)

    if not orchestrator.conversation_history:
        print("  No data yet")
        return

    latest = orchestrator.conversation_history[-1]
    
    # FIX: Get emotion from the right place
    # Orchestrator stores it at the root level
    emotion = latest.get('emotion', 'unknown')
    
    # Fallback to trying diagnosis if available
    if emotion == 'unknown' and 'diagnosis' in latest:
        diagnosis = latest['diagnosis']
        emotion = diagnosis.get('emotional_state', 
                               diagnosis.get('suspected_emotion', 'unknown'))
    
    # Get other fields
    intensity = latest.get('intensity', 0.5)
    risk = latest.get('risk_level', 'LOW')
    is_crisis = latest.get('is_crisis', False)
    intervention = latest.get('intervention_type', 'SUPPORT')

    # Display
    print(f"\n  🎭 Emotion: {emotion.upper()}")

    # Intensity bar
    try:
        intensity_val = float(intensity)
        intensity_pct = int(intensity_val * 100)
        bar_filled = int(intensity_val * 10)
        bar = '█' * bar_filled + '░' * (10 - bar_filled)
        print(f"  📈 Intensity: [{bar}] {intensity_pct}%")
    except:
        print(f"  📈 Intensity: {intensity}")

    # Risk
    risk_icons = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
    icon = risk_icons.get(risk, '⚪')
    print(f"  {icon} Risk: {risk}")
    
    if is_crisis:
        print(f"  🚨 CRISIS FLAG ACTIVE")
    
    print(f"  💊 Intervention: {intervention}")

    # Session trend (use last 5 turns)
    if len(orchestrator.conversation_history) >= 2:
        recent = orchestrator.conversation_history[-5:]
        emotions = []
        for h in recent:
            e = h.get('emotion', '?')
            if e == 'unknown':
                e = '?'
            emotions.append(e[:3].upper())  # Abbreviate
        
        trend = " → ".join(emotions)
        print(f"\n  📈 Session Trend: {trend}")

    print("─" * 60)


def print_reasoning_chain(debug_info):
    """Display the LLM reasoning chain"""
    print("\n" + "─" * 60)
    print("🧠 REASONING CHAIN")
    print("─" * 60)

    # Understanding
    understanding = debug_info. get('understanding', {})
    if understanding:
        print("\n  📖 UNDERSTANDING:")
        print(f"     Primary emotion: {understanding.get('primary_emotion', 'unknown')}")
        print(f"     Underlying need: {understanding.get('underlying_need', 'unknown')}")
        if understanding.get('concerning_language'):
            print(f"     ⚠️ Concerning language: {understanding.get('concerning_language')}")
        if understanding. get('euphemisms_detected'):
            print(f"     🔍 Euphemisms:  {understanding.get('euphemisms_detected')}")

    # Signal Analysis
    signal_analysis = debug_info.get('signal_analysis', {})
    if signal_analysis:
        print("\n  📡 SIGNAL ANALYSIS:")
        text_signals = signal_analysis.get('text_signals', {})
        if text_signals:
            print(f"     Text sentiment: {text_signals.get('sentiment', 'unknown')}")
        print(f"     Most reliable: {signal_analysis.get('most_reliable_signals', ['text'])}")
        if signal_analysis.get('overall_signal_summary'):
            summary = signal_analysis.get('overall_signal_summary', '')[:350]
            print(f"     Summary: {summary}...")

    # Inconsistency Check
    inconsistency = debug_info.get('inconsistency_check', {})
    if inconsistency:
        if inconsistency.get('inconsistencies_found'):
            print("\n  ⚡ INCONSISTENCIES DETECTED:")
            for inc in inconsistency.get('inconsistencies_found', [])[:2]:
                print(f"     - {inc. get('type', 'unknown')}: {inc.get('resolution', '')[: 200]}")
        if inconsistency.get('is_masking'):
            print(f"     🎭 Masking detected:  {inconsistency.get('masking_type', 'unknown')}")

    # Risk Assessment
    risk = debug_info.get('risk_assessment', {})
    if risk:
        print("\n  ⚠️ RISK ASSESSMENT:")
        print(f"     Level: {risk.get('risk_level', 'LOW')}")
        print(f"     Crisis: {'YES' if risk.get('is_crisis') else 'No'}")
        if risk.get('explicit_indicators'):
            print(f"     Explicit indicators: {risk.get('explicit_indicators')}")
        if risk.get('reasoning'):
            reason = risk.get('reasoning', '')[:150]
            print(f"     Reasoning: {reason}...")

    # Response Plan
    plan = debug_info.get('response_plan', {})
    if plan:
        print("\n  📋 RESPONSE PLAN:")
        print(f"     Strategy: {plan.get('primary_strategy', 'unknown')}")
        print(f"     Tone: {plan.get('tone', 'warm')}")
        if plan.get('must_include'):
            print(f"     Must include: {plan.get('must_include')[:3]}")

    print("─" * 60)

def print_memory_info(debug_info):
    """Display memory retrieval details"""
    print("\n" + "─" * 60)
    print("🧠 MEMORY RETRIEVAL")
    print("─" * 60)

    entities = debug_info.get('entities', [])
    context = debug_info.get('memory_context', '')

    if entities:
        print(f"  🏷️ Entities: {', '.join(entities)}")

    if context and "No previous context" not in context:
        print(f"  📚 Context ({len(context)} chars):")
        for line in context.split('\n')[:10]:
            if line.strip():
                print(f"     {line[: 75]}{'...' if len(line) > 75 else ''}")
    else:
        print("  📭 No relevant memories found")

    print("─" * 60)

def print_llm_context(debug_info):
    """Show what the LLM sees"""
    print("\n" + "─" * 60)
    print("🤖 LLM CONTEXT (What Kairos Sees)")
    print("─" * 60)

    diagnosis = debug_info.get('diagnosis', {})
    feature_data = debug_info.get('feature_data', {})

    biomarker = feature_data.get('vectors', {}).get('biomarker', [])

    # Show key signals the LLM receives
    print("\n  Key signals passed to reasoning:")

    if len(biomarker) >= 8:
        jitter = biomarker[0]
        shimmer = biomarker[1]
        f0_var = biomarker[2]

        print(f"    • Voice Tremor (Jitter): {jitter:+.3f}", end="")
        if jitter > 0.3:
            print(" → Anxiety/stress detected")
        else:
            print(" → Normal")

        print(f"    • Voice Stability (Shimmer): {shimmer:+.3f}", end="")
        if shimmer > 0.3:
            print(" → Unstable voice")
        else:
            print(" → Stable")

        print(f"    • Expressiveness (F0 Var): {f0_var:+.3f}", end="")
        if f0_var < -0.3:
            print(" → Monotone (possible depression)")
        elif f0_var > 0.3:
            print(" → Highly expressive")
        else:
            print(" → Normal range")

    if len(biomarker) > 22:
        sentiment = biomarker[22]
        print(f"    • Sentiment: {sentiment:+.3f}", end="")
        if sentiment < -0.3:
            print(" → Negative")
        elif sentiment > 0.3:
            print(" → Positive")
        else:
            print(" → Neutral")

    if len(biomarker) > 24:
        laughter = biomarker[24]
        if laughter > 0.3:
            print(f"    • Laughter: {laughter:.3f} → Joy/humor detected")

    print("─" * 60)

# ============================================================
# WELCOME
# ============================================================
print("\n" + "=" * 60)
print("  KAIROS - Your Mental Support Companion")
print("=" * 60)
print("""
I'm here to listen and support you.

Input options:
  [T] Type a message
  [A] Upload audio
  [V] Upload video
  [D] Toggle debug
  [Q] End session
""")
print("=" * 60)


# MAIN INTERACTION LOOP 
turn_count = 0
last_system_end_time = datetime.now()
show_debug = True

while True:
    print("\n" + "-" * 40)
    print(f"Turn {turn_count + 1} | Debug: {'ON' if show_debug else 'OFF'}")
    print("  [T] Text  [A] Audio  [V] Video  [D] Debug  [Q] Quit")
    print("-" * 40)

    choice = input("Choice: ").strip().upper()

    if choice == 'Q':
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        if orchestrator.conversation_history:
            print_mental_state(orchestrator)
        else:
            print("  No conversation data recorded.")

        # Save session
        try:
            session_summary = orchestrator.get_session_summary()
            orchestrator.memory. save_session_summary(session_id, session_summary)
            print("\n✅ Session saved to memory for future reference.")
        except Exception as e:
            print(f"\n⚠️ Could not save session: {e}")

        print("\nThank you for sharing today.  Take care.  💙")
        print("=" * 60)
        break

    if choice == 'D':
        show_debug = not show_debug
        print(f"✅ Debug {'ON' if show_debug else 'OFF'}")
        continue

    # Build input object
    input_object = {
        "session_id": session_id,
        "turn_id": turn_count,
        "timestamp": datetime.now().isoformat(),
        "last_system_end_time": last_system_end_time.isoformat()
    }

    if choice == 'T':
        print("\nType message (Enter twice to send):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            elif line:
                lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            print("No message entered.")
            continue

        input_object["modality"] = "text"
        input_object["modality_type"] = "text"  # Add this!
        input_object["text"] = text
        input_object["filepath"] = None

    elif choice in ['A', 'V']:
        print(f"\n📁 Upload your {'audio' if choice == 'A' else 'video'} file...")
        try:
            uploaded = colab_files.upload()
            if not uploaded:
                print("No file uploaded.")
                continue

            filepath = get_uploaded_filepath(uploaded)

            if not filepath:
                print("Failed to process uploaded file.")
                continue

            modality = "audio" if choice == 'A' else "video"
            input_object["modality"] = modality
            input_object["modality_type"] = modality  # Add this!
            input_object["text"] = None
            input_object["filepath"] = filepath

        except Exception as e:
            print(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    else:
        print("Invalid choice.")
        continue

    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)

    try:
        # Load the file if audio/video
        if input_object. get("filepath"):
            print("  Loading file...")
            loaded_data = file_loader.load(input_object)
            input_object. update(loaded_data)

        # Process the turn
        response, debug_info = orchestrator.process_turn_with_debug(input_object)

        # Update timing
        last_system_end_time = datetime.now()
        turn_count += 1

        # Display debug info if enabled
        if show_debug and debug_info:
            print_full_debug(debug_info, orchestrator)

        # Display response
        print("\n" + "=" * 60)
        print("KAIROS:")
        print("=" * 60)
        print(f"\n{response}\n")

        # Display mental state
        if show_debug:
            print_mental_state(orchestrator)

    except Exception as e:
        print(f"\n❌ Error processing turn: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease try again.")