"""
Main entry point for Kairos - Enhanced Version
Demonstrates all Phase 1, 2, 3 features. 
"""
import os
from datetime import datetime

from kairos.orchestration.orchestrator import KairosOrchestrator
from kairos.extraction. feature_engine import FeatureEngine
from kairos.memory.memory_controller import MemoryController
from kairos.memory.baseline_manager import BaselineManager
from config import GROQ_API_KEY, QDRANT_PATH


def create_kairos(user_id: str = "default_user", api_key: str = None):
    """
    Create and initialize a Kairos instance with all enhancements.
    
    Args:
        user_id:  Unique user identifier
        api_key:  Groq API key (uses env var if not provided)
    
    Returns:
        Configured KairosOrchestrator instance
    """
    api_key = api_key or GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in environment or passed as argument")
    
    print("=" * 60)
    print("🧠 Initializing Kairos Mental Health Support System v2.0")
    print("=" * 60)
    
    # Initialize baseline manager for personalized calibration
    print("\n📊 Setting up user baseline manager...")
    baseline_manager = BaselineManager(user_id=user_id)
    
    # Initialize feature extraction engine
    print("\n🎤 Initializing feature extraction engine...")
    feature_engine = FeatureEngine(baseline_manager=baseline_manager)
    
    # Initialize enhanced memory controller with all Phase 1, 2, 3 components
    print("\n💾 Initializing enhanced memory system...")
    memory_controller = MemoryController(
        qdrant_path=QDRANT_PATH,
        user_id=user_id
    )
    
    # Create session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize orchestrator with all integrations
    print("\n🎯 Initializing orchestrator with all enhancements...")
    orchestrator = KairosOrchestrator(
        feature_engine=feature_engine,
        memory_controller=memory_controller,
        api_key=api_key,
        user_id=user_id,
        session_id=session_id
    )
    
    print("\n" + "=" * 60)
    print("✅ Kairos initialized successfully!")
    print("=" * 60)
    print("\nEnhancements active:")
    print("  Phase 1: Hybrid Search + RRF + Entity Graph + Sparse Vectors")
    print("  Phase 2: Intervention Tracking + Discovery Search")
    print("  Phase 3: Trajectory Matching + ColBERT + Binary Quantization")
    print("=" * 60 + "\n")
    
    return orchestrator


def run_interactive_session(orchestrator: KairosOrchestrator):
    """Run an interactive chat session."""
    print("\n🗣️ Starting interactive session...")
    print("Type 'quit' to exit, 'debug' to see last turn's debug info")
    print("Type 'stats' to see system statistics")
    print("-" * 40 + "\n")
    
    last_debug = {}
    
    while True: 
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n💾 Saving session...")
                orchestrator.save_session()
                print("👋 Goodbye!  Take care of yourself.")
                break
            
            if user_input. lower() == 'debug':
                if last_debug:
                    print("\n📊 Last turn debug info:")
                    _print_debug_summary(last_debug)
                else:
                    print("No debug info available yet.")
                continue
            
            if user_input.lower() == 'stats':
                stats = orchestrator.get_debug_info()
                _print_stats(stats)
                continue
            
            # Process the turn
            input_object = {
                "text": user_input,
                "modality": "text",
                "modality_type": "text"
            }
            
            response, debug_info = orchestrator.process_turn_with_debug(input_object)
            last_debug = debug_info
            
            print(f"\nKairos: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n💾 Saving session...")
            orchestrator.save_session()
            print("👋 Session saved.  Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def _print_debug_summary(debug_info: dict):
    """Print a summary of debug information."""
    print("-" * 40)
    
    # Diagnosis
    if 'diagnosis' in debug_info: 
        d = debug_info['diagnosis']
        print(f"  Emotion: {d.get('emotional_state', 'unknown')}")
        print(f"  Intensity: {d.get('emotional_intensity', 0):.2f}")
        print(f"  Risk:  {d.get('risk_level', 'unknown')}")
        print(f"  Crisis: {d.get('is_crisis', False)}")
        print(f"  Masking:  {d.get('is_masking', False)}")
        if d.get('trigger_entities'):
            print(f"  Triggers: {d.get('trigger_entities')}")
        if d.get('trajectory_pattern'):
            print(f"  Trajectory: {d.get('trajectory_pattern')}")
    
    # Safety details
    if 'safety_details' in debug_info:
        s = debug_info['safety_details']
        print(f"\n  Safety Score: {s.get('score', 0):.2f}")
        if s.get('sparse_terms'):
            print(f"  Crisis Terms: {[t['term'] for t in s['sparse_terms'][:3]]}")
    
    # Memory debug
    if 'memory_debug' in debug_info:
        m = debug_info['memory_debug']
        print(f"\n  Memory Queries: {m.get('query_type', [])}")
        print(f"  Results Found: {m.get('total_searched', 0)}")
        if m.get('crisis_mode'):
            print("  ⚠️ Crisis mode active")
        if m.get('entity_alerts'):
            print(f"  Entity Alerts: {len(m.get('entity_alerts', []))}")
    
    # Intervention
    if 'intervention_type' in debug_info:
        print(f"\n  Intervention:  {debug_info['intervention_type']}")
    
    if 'recommended_intervention' in debug_info and debug_info['recommended_intervention']: 
        print(f"  Recommended: {debug_info['recommended_intervention']}")
    
    print("-" * 40)


def _print_stats(stats: dict):
    """Print system statistics."""
    print("\n📈 System Statistics")
    print("-" * 40)
    
    if 'session_state' in stats:
        s = stats['session_state']
        print(f"  Turn Count: {s.get('turn_count', 0)}")
        print(f"  Crisis Turns: {len(s.get('crisis_turns', []))}")
        print(f"  Trajectory Alerts: {len(s.get('trajectory_alerts', []))}")
        print(f"  Trigger Alerts: {len(s.get('trigger_alerts_this_session', []))}")
    
    if 'memory_stats' in stats:
        m = stats['memory_stats']
        if 'qdrant' in m:
            print(f"\n  Total Memories: {m['qdrant']. get('total_memories', 0)}")
            print(f"  Session Memories: {m['qdrant'].get('session_memories', 0)}")
        if 'entity_graph' in m:
            print(f"  Entities Tracked: {m['entity_graph']. get('total_entities', 0)}")
            print(f"  Triggers Identified: {m['entity_graph']. get('trigger_count', 0)}")
            print(f"  Support Systems: {m['entity_graph']. get('support_count', 0)}")
        if 'intervention_tracker' in m:
            print(f"  Interventions Recorded: {m['intervention_tracker']. get('total_interventions', 0)}")
            print(f"  Success Rate: {m['intervention_tracker'].get('success_rate', 0):.2%}")
    
    if 'safety_checker_stats' in stats:
        sc = stats['safety_checker_stats']
        print(f"\n  Safety Evaluations: {sc.get('total_evaluations', 0)}")
        print(f"  Current Threshold: {sc.get('current_threshold', 0.7):.2f}")
    
    print("-" * 40)


def process_audio_file(orchestrator: KairosOrchestrator, audio_path: str) -> str:
    """Process an audio file through Kairos."""
    from kairos.ingestion.file_loader import FileLoader
    
    loader = FileLoader()
    
    input_object = loader.load({
        "filepath": audio_path,
        "modality": "audio"
    })
    
    response, debug_info = orchestrator.process_turn_with_debug(input_object)
    
    return response


def process_video_file(orchestrator: KairosOrchestrator, video_path: str) -> str:
    """Process a video file through Kairos."""
    from kairos.ingestion.file_loader import FileLoader
    
    loader = FileLoader()
    
    input_object = loader.load({
        "filepath": video_path,
        "modality": "video"
    })
    
    response, debug_info = orchestrator.process_turn_with_debug(input_object)
    
    return response


if __name__ == "__main__": 
    import sys
    
    # Get user ID from command line or use default
    user_id = sys.argv[1] if len(sys.argv) > 1 else "default_user"
    
    # Create Kairos instance
    try:
        kairos = create_kairos(user_id=user_id)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)
    
    # Run interactive session
    run_interactive_session(kairos)