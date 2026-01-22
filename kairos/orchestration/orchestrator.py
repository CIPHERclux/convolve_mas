"""
Kairos Orchestrator - DYNAMIC REASONING VERSION
LLM decides whether to do deep mental assessment or simple response. 
All reasoning steps preserved but triggered dynamically. 
"""
import json
import random
import numpy as np 
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from groq import Groq

from kairos.extraction.feature_engine import FeatureEngine
from kairos.memory.memory_controller import MemoryController
from kairos.segmentation.trigger_engine import TriggerEngine
from kairos.utils.entity_extractor import EntityExtractor
from kairos.utils.safety_checker import SafetyChecker
from kairos.orchestration.biomarker_tracker import BiomarkerTracker
from config import GROQ_MODEL, GROQ_MODEL_FALLBACK


class KairosOrchestrator:
    """
    Kairos Orchestrator - Dynamic Reasoning
    
    The LLM decides: 
    1. Whether deep mental assessment is needed
    2. If yes -> Full reasoning chain (understanding, signals, risk, plan)
    3. If no -> Direct/simple response
    
    All reasoning steps preserved but triggered dynamically. 
    """
    
    def __init__(
        self,
        feature_engine: FeatureEngine,
        memory_controller: MemoryController,
        api_key: str,
        user_id: str = "default_user",
        session_id:  str = None
    ):
        self.features = feature_engine
        self. memory = memory_controller
        self.user_id = user_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # LLM setup
        self.llm = Groq(api_key=api_key)
        self.model = GROQ_MODEL
        self.fallback_model = GROQ_MODEL_FALLBACK
        
        # Core components
        self.trigger_engine = TriggerEngine()
        self.entity_extractor = EntityExtractor()
        
        # Get references to memory sub-components
        self.sparse_encoder = memory_controller.sparse_encoder
        self.entity_graph = memory_controller.entity_graph
        self.graph_rag = memory_controller.graph_rag
        self.modality_tracker = memory_controller. modality_tracker
        self.user_profile = memory_controller.user_profile
        
        # Safety checker
        self.safety_checker = SafetyChecker(
            sparse_encoder=self.sparse_encoder,
            entity_graph=self.entity_graph
        )
        
        # Biomarker tracker
        self. biomarker_tracker = BiomarkerTracker(history_size=15)
        
        # Conversation history
        
        
        # Previous state for intervention tracking
        self.previous_biomarker:  Optional[List[float]] = None
        self.previous_intervention_type: Optional[str] = None
        self.previous_emotion: Optional[str] = None
        
        # Conversation history (list for simple tracking)
        self.conversation_history = []
        
        # Conversation history by session (dict for multi-session tracking)
        self.conversation_history_by_session = {}
        # Debug mode
        self. debug_mode = True
        
        # Session state
        self.session_state = {
            "turn_count": 0,
            "session_start": datetime.now().isoformat(),
            "emotional_arc":  [],
            "intensity_arc": [],
            "risk_arc": [],
            "crisis_turns": [],
            "last_crisis_turn": -999,
            "crisis_acknowledged": False,
            "resources_given_turn": -999,
            "cumulative_signals": {
                "distress_level": 0.0,
                "positive_level": 0.0,
                "anxiety_indicators": 0.0,
                "depression_indicators": 0.0,
                "crying_confidence": 0.0,
                "laughter_confidence":  0.0,
            },
            "signal_history": [],
            "questions_asked_last_3_turns": 0,
            "last_response_had_question": False,
            "topics_user_shared": [],
            "user_language_style": "unknown",
            "user_uses_profanity": False,
            "rapport_level": 0.0,
            "trigger_alerts_this_session": [],
            "interventions_used": [],
            "effective_interventions": [],
            "trajectory_alerts": [],
            "modalities_used": set(),
            "deep_assessments_done": 0,
            "simple_responses_done": 0,
        }
    
    def process_turn(self, input_object: Dict[str, Any]) -> str:
        """Simple interface."""
        response, _ = self.process_turn_with_debug(input_object)
        return response
    
    def process_turn_with_debug(self, input_object: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process turn with dynamic reasoning and rich context building."""
        debug_info = {
            "routing": {},
            "feature_extraction": {},
            "graph_rag": {},
            "modality_tracking": {},
            "safety": {},
            "memory": {},
            "reasoning": {},
            "response": {}
        }
        current_turn = self.session_state["turn_count"]
        
        # =====================================================================
        # STEP 1: INPUT PROCESSING
        # =====================================================================
        self._print_header(f"TURN {current_turn + 1}")
        
        segments = self.trigger_engine.segment(input_object)
        input_object["segments"] = segments
        
        text = input_object.get("text") or self.trigger_engine.get_full_text(segments)
        input_object["text"] = text
        
        if not text or text.strip() in ["[No speech detected]", "[Transcription error]", ""]:
            return "I couldn't quite catch that. Could you try again?", {}
        
        self._print_step("INPUT", f"\"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        
        self._analyze_user_style(text)
        modality = input_object.get("modality_type", input_object.get("modality", "text"))
        self.session_state["modalities_used"].add(modality)
        
        # =====================================================================
        # STEP 2: FEATURE EXTRACTION
        # =====================================================================
        self._print_step("FEATURES", "Extracting biomarkers...")
        feature_data = self.features.extract(input_object)
        
        biomarker = feature_data['vectors']['biomarker']
        reliability = feature_data['payload']['reliability_mask']
        modality_type = feature_data['payload'].get('modality_type', 'text')
        
        # Track biomarker in trajectory
        self.biomarker_tracker.add_turn(
            biomarker=biomarker,
            modality=modality_type,
            turn_number=current_turn
        )
        
        # Update persistent signals
        self._update_persistent_signals(biomarker, reliability, modality_type)
        
        # Get biomarker delta (changes from previous turn)
        biomarker_delta = self.biomarker_tracker.get_delta()
        
        # Create feature insights
        feature_insights = self._create_feature_insights(biomarker, modality_type)
        
        # Initialize debug info for features
        debug_info['feature_extraction'] = {
            "modality": modality_type,
            "biomarker_first_8": list(biomarker[:8]) if len(biomarker) >= 8 else list(biomarker),
            "has_delta": biomarker_delta.get('has_delta', False),
            "significant_changes": biomarker_delta.get('significant_changes', []),
            "feature_insights": feature_insights,
            "biomarker_available": len(biomarker) > 0 and any(b != 0 for b in biomarker[:8]),
            "visual_features": {},
            "notable_visual": []
        }
        notable_visual = []
        # Extract visual features if video modality
        visual_features = {
            'masking_score': 0,
            'brow_tension': 0,
            'gaze_aversion': 0,
            'facial_dynamism': 0,
            'stare_duration': 0,
            'blink_rate': 0,
        }
        
        if modality_type in ['video'] and len(biomarker) > 15:
            visual_features = {
                'masking_score': biomarker[8] if len(biomarker) > 8 else 0,
                'brow_tension': biomarker[9] if len(biomarker) > 9 else 0,
                'gaze_aversion': biomarker[10] if len(biomarker) > 10 else 0,
                'facial_dynamism': biomarker[11] if len(biomarker) > 11 else 0,
                'stare_duration': biomarker[12] if len(biomarker) > 12 else 0,
                'blink_rate': biomarker[13] if len(biomarker) > 13 else 0,
            }
            
            notable_visual = []
            if visual_features['brow_tension'] > 0.6:
                notable_visual.append(f"High brow tension ({visual_features['brow_tension']:.2f})")
            if visual_features['gaze_aversion'] > 0.5:
                notable_visual.append(f"Gaze aversion ({visual_features['gaze_aversion']:.2f})")
            if visual_features['masking_score'] > 0.6:
                notable_visual.append(f"Masking detected ({visual_features['masking_score']:.2f})")
            
            if notable_visual:
                self._print_debug("VISUAL", "; ".join(notable_visual))
                debug_info['feature_extraction']['visual_features'] = visual_features
                debug_info['feature_extraction']['notable_visual'] = notable_visual
        
        # Print biomarker changes
        if biomarker_delta.get('has_delta'):
            changes = biomarker_delta.get('significant_changes', [])
            if changes:
                self._print_debug("BIOMARKER", f"{len(changes)} changes detected")
                for c in changes[:2]:
                    self._print_debug("", f"  • {c['feature']}: {c['interpretation']}")
        
        # =====================================================================
        # STEP 3: ENTITY EXTRACTION & GRAPH-RAG
        # =====================================================================
        self._print_step("GRAPH-RAG", "Entity extraction and expansion...")
        entities = self.entity_extractor.extract(text)
        
        # Graph-RAG expansion
        graph_expansion = None
        related_entities = []
        emotional_associations = {}
        
        if entities:
            graph_expansion = self.graph_rag.get_query_expansion(entities)
            related_entities = [r['entity'] for r in graph_expansion.get('related_with_scores', [])[:5]]
            emotional_associations = graph_expansion.get('emotional_context', {})
            
            self._print_debug("GRAPH-RAG", f"Entities: {entities}")
            if related_entities:
                self._print_debug("GRAPH-RAG", f"Related (spreading activation): {related_entities}")
            if emotional_associations:
                top_emotions = sorted(emotional_associations.items(), key=lambda x: x[1], reverse=True)[:3]
                self._print_debug("GRAPH-RAG", f"Emotional associations: {dict(top_emotions)}")
        
        debug_info['graph_rag'] = {
            "original_entities": entities,
            "related_entities": related_entities,
            "emotional_associations": emotional_associations,
            "topics": graph_expansion.get('topic_context', []) if graph_expansion else [],
            "expansion_terms": graph_expansion.get('expansion_terms', []) if graph_expansion else []
        }
        
        # Check for trigger entities
        trigger_alerts = self.entity_graph.check_trigger_entities(entities) if entities else []
        if trigger_alerts:
            self.session_state["trigger_alerts_this_session"].extend(trigger_alerts)
            self._print_warning("TRIGGERS", f"⚠️ {[a['entity'] for a in trigger_alerts]}")
        
        debug_info['graph_rag']['trigger_alerts'] = trigger_alerts
        
        # =====================================================================
        # STEP 4: CREATE DETAILED BIOMARKER SUMMARY (NEW)
        # =====================================================================
        self._print_step("BIOMARKERS", "Creating detailed summary...")
        
        # Get laughter score from feature data
        laughter_score = feature_data['payload'].get('laughter_score', 0.0)
        
        # Create detailed biomarker summary using the new method
        biomarker_summary = self._create_detailed_biomarker_summary(
            bio_vector=biomarker,
            reliability_mask=reliability,
            laughter_score=laughter_score
        )
        
        # Add visual context if available
        if notable_visual:
            visual_context = "\n\nVISUAL SIGNALS:\n" + "\n".join([f"• {v}" for v in notable_visual])
            biomarker_summary += visual_context
        
        self._print_debug("BIOMARKERS", f"Summary created: {len(biomarker_summary)} chars")
        
        # =====================================================================
        # STEP 5: SAFETY CHECK
        # =====================================================================
        self._print_step("SAFETY", "Evaluating crisis risk...")
        prev_messages = [h.get('user', '') for h in self.conversation_history[-5:]]
        
        safety_details = self.safety_checker.evaluate_with_details(
            text=text,
            conversation_context=prev_messages,
            entities=entities
        )
        
        crisis_score = safety_details['score']
        is_crisis_from_safety = safety_details['is_crisis']
        
        # Fix: Ensure crisis_score is never 0 if patterns matched
        if safety_details.get('matched_patterns') and crisis_score == 0.0:
            crisis_score = max(0.5, max([p['weight'] for p in safety_details['matched_patterns']]))
            is_crisis_from_safety = crisis_score >= self.safety_checker.current_threshold
        
        debug_info['safety'] = {
            "crisis_score": crisis_score,
            "is_crisis": is_crisis_from_safety,
            "matched_patterns": [p['pattern'] for p in safety_details.get('matched_patterns', [])],
            "sparse_terms": [t['term'] for t in safety_details.get('sparse_terms', [])],
            "escalation_detected": safety_details.get('escalation_detected', False),
            "threshold": safety_details.get('threshold', 0.7)
        }
        
        if is_crisis_from_safety:
            self._print_warning("SAFETY", f"🚨 CRISIS DETECTED (score: {crisis_score:.2f})")
        
        # =====================================================================
        # STEP 6: BUILD RICH MEMORY CONTEXT (NEW METHOD)
        # =====================================================================
        self._print_step("MEMORY", "Building rich context...")
        
        # Get trajectory info for context
        trajectory_pattern = self.memory.trajectory_matcher.get_trajectory_pattern()
        
        # Build trigger context string
        trigger_context = ""
        if trigger_alerts:
            trigger_context = "⚠️ TRIGGER ENTITIES DETECTED:\n"
            for alert in trigger_alerts[:3]:
                trigger_context += f"• {alert['entity']}: trigger_score={alert['trigger_score']:.2f} "
                trigger_context += f"(crisis {alert['crisis_count']}/{alert['mention_count']} times)\n"
        
        # Build trajectory context string
        trajectory_info = ""
        if trajectory_pattern['pattern'] != 'INSUFFICIENT_DATA':
            if trajectory_pattern['alert_level'] in ['MODERATE', 'HIGH', 'CRITICAL']:
                trajectory_info = f"⚠️ TRAJECTORY: {trajectory_pattern['description']} "
                trajectory_info += f"(Confidence: {trajectory_pattern['confidence']:.2f})\n"
                trajectory_info += f"Recommendation: {trajectory_pattern['recommendation']}"
        
        # Use the NEW build_rich_context method
        memory_context = self.memory.build_rich_context(
            query_text=text,
            biomarker=biomarker,
            entities=entities,
            session_id=self.session_id,
            limit=8
        )
        
        # Get user facts
        user_facts = self.memory.get_user_facts()
        user_facts_for_llm = self.memory.get_user_facts_for_llm()
        
        # Get modality-specific context if needed
        modality_context = ""
        text_lower = text.lower()
        if any(w in text_lower for w in ['audio', 'voice', 'recording', 'sound']):
            modality_context = self.memory.get_modality_progression_for_llm('audio')
            self._print_debug("MODALITY", "Added audio history context")
        elif any(w in text_lower for w in ['video', 'face', 'camera']):
            modality_context = self.memory.get_modality_progression_for_llm('video')
            self._print_debug("MODALITY", "Added video history context")
        
        # Get intervention recommendation
        recommended_intervention = self.memory.get_effective_intervention_recommendation(
            current_emotion=self.previous_emotion or "unknown",
            current_biomarker=biomarker
        )
        
        if recommended_intervention:
            self._print_debug("INTERVENTION", f"Recommended: {recommended_intervention}")
        
        # Debug info
        self._print_debug("MEMORY", f"Context built: {len(memory_context)} chars")
        if user_facts.get('name'):
            self._print_debug("PROFILE", f"User: {user_facts['name']}")
        
        debug_info['memory'] = {
            "user_facts": user_facts if user_facts else {},
            "user_facts_for_llm": user_facts_for_llm if user_facts_for_llm else "",
            "memory_context_length": len(memory_context),
            "has_trigger_context": bool(trigger_context),
            "has_trajectory_info": bool(trajectory_info),
            "recommended_intervention": recommended_intervention,
            "modality_context_added": bool(modality_context)
        }
        
        debug_info['modality_tracking'] = {
            "current_modality": modality_type,
            "modality_context_added": bool(modality_context),
            "stats": self.modality_tracker.get_stats()
        }
        
        # =====================================================================
        # STEP 7: ROUTING DECISION
        # =====================================================================
        self._print_step("ROUTING", "Deciding response path...")
        
        routing_decision = self._llm_routing_decision(
            text=text,
            user_facts=user_facts,
            biomarker_summary=biomarker_summary,
            safety_details=safety_details,
            is_crisis=is_crisis_from_safety,
            crisis_score=crisis_score,
            trigger_alerts=trigger_alerts,
            emotional_associations=emotional_associations,
            memory_context=memory_context[:500]
        )
        
        debug_info['routing'] = routing_decision
        
        needs_deep_assessment = routing_decision.get('needs_deep_assessment', True)
        routing_reason = routing_decision.get('reason', 'default')
        message_type = routing_decision.get('message_type', 'UNKNOWN')
        
        self._print_debug("ROUTING", f"Message type: {message_type}")
        self._print_debug("ROUTING", f"Deep assessment: {needs_deep_assessment}")
        self._print_debug("ROUTING", f"Reason: {routing_reason}")
        
        # Force deep assessment for crisis
        if is_crisis_from_safety:
            needs_deep_assessment = True
            self._print_warning("ROUTING", "Forced deep assessment due to crisis")
        
        # Store notable_visual for later use
        if 'notable_visual' not in locals():
            notable_visual = []
        # =====================================================================
        # STEP 8: RESPONSE GENERATION
        # =====================================================================
        if needs_deep_assessment:
            # ===============================================================
            # PATH A: DEEP MENTAL ASSESSMENT
            # ===============================================================
            self._print_header("DEEP ASSESSMENT PATH")
            self.session_state["deep_assessments_done"] += 1
            
            # Step 8a: Understanding
            self._print_step("UNDERSTANDING", "Analyzing emotional state...")
            understanding = self._reason_understanding(
                text=text,
                biomarker=biomarker,
                biomarker_summary=biomarker_summary,
                biomarker_delta=biomarker_delta,
                modality=modality_type,
                user_facts=user_facts,
                memory_context=memory_context[:1000],  # Truncate for LLM
                emotional_associations=emotional_associations
            )
            debug_info['reasoning']['understanding'] = understanding
            
            self._print_debug("UNDERSTANDING", f"Emotion: {understanding.get('primary_emotion', 'unknown')}")
            self._print_debug("UNDERSTANDING", f"Intensity: {understanding.get('intensity', 0.5):.2f}")
            if understanding.get('what_theyre_not_saying'):
                self._print_debug("UNDERSTANDING", f"Subtext: {understanding.get('what_theyre_not_saying')[:80]}")
            
            # Step 8b: Signal Integration
            self._print_step("SIGNALS", "Integrating multimodal signals...")
            signal_analysis = self._reason_signal_integration(
                text=text,
                biomarker=biomarker,
                biomarker_summary=biomarker_summary,
                understanding=understanding,
                trigger_alerts=trigger_alerts,
                emotional_associations=emotional_associations
            )
            debug_info['reasoning']['signal_analysis'] = signal_analysis
            
            if signal_analysis.get('is_masking'):
                self._print_warning("SIGNALS", f"⚠️ MASKING: {signal_analysis.get('masking_type', 'unknown')}")
            if signal_analysis.get('hidden_distress'):
                self._print_warning("SIGNALS", "⚠️ Hidden distress detected")
            self._print_debug("SIGNALS", f"State: {signal_analysis.get('integrated_emotional_state', 'unknown')}")
            
            # Step 8c: Risk Assessment
            self._print_step("RISK", "Assessing risk level...")
            risk_assessment = self._reason_risk_assessment(
                text=text,
                understanding=understanding,
                signal_analysis=signal_analysis,
                safety_details=safety_details,
                trigger_alerts=trigger_alerts
            )
            debug_info['reasoning']['risk_assessment'] = risk_assessment
            
            is_crisis = risk_assessment.get('is_crisis', is_crisis_from_safety)
            risk_level = risk_assessment.get('risk_level', 'LOW')
            
            self._print_debug("RISK", f"Level: {risk_level}")
            self._print_debug("RISK", f"Crisis: {is_crisis}")
            if risk_assessment.get('reasoning'):
                self._print_debug("RISK", f"Reason: {risk_assessment.get('reasoning')[:100]}")
            
            # Step 8d: Response Planning
            self._print_step("PLANNING", "Planning response strategy...")
            response_plan = self._reason_response_plan(
                text=text,
                understanding=understanding,
                signal_analysis=signal_analysis,
                risk_assessment=risk_assessment,
                user_facts=user_facts,
                memory_context=memory_context[:800],
                recommended_intervention=recommended_intervention,
                message_type=message_type
            )
            debug_info['reasoning']['response_plan'] = response_plan
            
            self._print_debug("PLANNING", f"Strategy: {response_plan.get('response_type', 'SUPPORT')}")
            self._print_debug("PLANNING", f"Tone: {response_plan.get('tone', 'warm')}")
            if response_plan.get('key_points'):
                self._print_debug("PLANNING", f"Points: {len(response_plan.get('key_points', []))}")
            
            # Step 8e: Generate Response
            self._print_step("RESPONSE", "Generating response...")
            response = self._generate_deep_response(
                text=text,
                understanding=understanding,
                signal_analysis=signal_analysis,
                risk_assessment=risk_assessment,
                response_plan=response_plan,
                user_facts=user_facts,
                biomarker_summary=biomarker_summary,
                memory_context=memory_context[:1000],
                modality_context=modality_context
            )
            
            # Extract values for saving
            emotional_state = understanding.get('primary_emotion', 'uncertain')
            intensity = understanding.get('intensity', 0.5)
            intervention_type = response_plan.get('response_type', 'SUPPORT')
            
        else:
            # ===============================================================
            # PATH B: SIMPLE/DIRECT RESPONSE
            # ===============================================================
            self._print_header("SIMPLE RESPONSE PATH")
            self.session_state["simple_responses_done"] += 1
            
            simple_result = self._generate_simple_response(
                text=text,
                message_type=message_type,
                user_facts=user_facts,
                memory_context=memory_context[:600],
                modality_context=modality_context,
                biomarker_summary=biomarker_summary if routing_decision.get('use_biomarkers', False) else None
            )
            
            debug_info['reasoning']['simple_response'] = simple_result
            
            response = simple_result.get('response', "I'm here. What's on your mind?")
            emotional_state = simple_result.get('detected_emotion', 'neutral')
            intensity = simple_result.get('intensity', 0.3)
            risk_level = 'LOW'
            is_crisis = False
            intervention_type = simple_result.get('response_type', 'DIRECT_ANSWER')
            
            self._print_debug("SIMPLE", f"Type: {intervention_type}")
        
        # =====================================================================
        # STEP 9: POST-PROCESS RESPONSE
        # =====================================================================
        self._print_step("POST-PROCESS", "Finalizing response...")
        
        final_response = self._post_process_response(
            response=response,
            is_crisis=is_crisis,
            risk_level=risk_level
        )
        
        debug_info['response'] = {
            "path": "deep_assessment" if needs_deep_assessment else "simple",
            "emotional_state": emotional_state,
            "intensity": intensity,
            "risk_level": risk_level,
            "is_crisis": is_crisis,
            "intervention_type": intervention_type,
            "response_length": len(final_response)
        }
        
        # =====================================================================
        # STEP 10: SAVE TO MEMORY
        # =====================================================================
        self._print_step("SAVE", "Persisting to memory...")
        
        self._save_turn(
            text=text,
            response=final_response,
            feature_data=feature_data,
            entities=entities,
            is_crisis=is_crisis,
            crisis_score=crisis_score,
            emotional_state=emotional_state,
            intensity=intensity,
            risk_level=risk_level,
            intervention_type=intervention_type,
            modality=modality_type,
            feature_insights=feature_insights,
            trigger_alerts=trigger_alerts
        )
        
        # =====================================================================
        # STEP 11: UPDATE SESSION STATE & DEBUG INFO
        # =====================================================================
        # Ensure emotional_state is valid
        if emotional_state in ['uncertain', 'unknown', None, '']:
            if needs_deep_assessment and debug_info.get('reasoning', {}).get('understanding'):
                emotional_state = debug_info['reasoning']['understanding'].get('primary_emotion', 'neutral')
            else:
                emotional_state = 'neutral'
        
        # Set comprehensive debug info
        debug_info['final_analysis'] = {
            'emotion': emotional_state,
            'intensity': float(intensity),
            'risk_level': risk_level,
            'is_crisis': is_crisis,
            'intervention': intervention_type,
            'confidence': 0.7 if needs_deep_assessment else 0.5
        }
        
        # Top-level keys for backward compatibility
        debug_info['emotion'] = emotional_state
        debug_info['intensity'] = float(intensity)
        debug_info['risk_level'] = risk_level
        debug_info['is_crisis'] = is_crisis
        debug_info['intervention_type'] = intervention_type
        debug_info['session_state'] = self.session_state
        
        # Print summary
        self._print_summary(
            emotional_state=emotional_state,
            intensity=intensity,
            risk_level=risk_level,
            is_crisis=is_crisis,
            path="deep" if needs_deep_assessment else "simple"
        )
        
        print(f"\n  ✅ Turn {current_turn + 1} complete")
        print("=" * 60)
        
        return final_response, debug_info
    
    # =========================================================================
    # ROUTING DECISION
    # =========================================================================
    def _llm_routing_decision(
        self,
        text: str,
        user_facts: Dict[str, Any],
        biomarker_summary: str,
        safety_details: Dict,
        is_crisis: bool,
        crisis_score: float,
        trigger_alerts: List[Dict],
        emotional_associations:  Dict[str, float],
        memory_context: str
    ) -> Dict[str, Any]:
        """
        LLM decides whether this message needs deep mental assessment
        or can be answered simply/directly. 
        """
        
        # Build emotional history context
        emotional_history = ""
        if self.session_state["emotional_arc"]:
            recent_emotions = self.session_state["emotional_arc"][-5:]
            emotional_history = f"Recent emotions: {recent_emotions}"
            
            if self.session_state["crisis_turns"]:
                emotional_history += f"\nCrisis turns this session: {self.session_state['crisis_turns']}"
        
        prompt = f"""You are a routing system for a mental health support companion. 

Decide whether this message needs DEEP MENTAL ASSESSMENT or can be answered SIMPLY. 

=== USER'S MESSAGE ===
"{text}"

=== CONTEXT ===
User facts: {json.dumps(user_facts) if user_facts else "None"}
Crisis score: {crisis_score:.2f}
Is crisis: {is_crisis}
Trigger entities: {[a['entity'] for a in trigger_alerts]}
Emotional associations: {emotional_associations}
{emotional_history}
Session turn: {self.session_state['turn_count'] + 1}
Previous emotion: {self.previous_emotion or "unknown"}

=== BIOMARKER SIGNALS ===
{biomarker_summary[: 500] if biomarker_summary else "Text-only input"}

=== ROUTING RULES ===

NEEDS DEEP ASSESSMENT when:
- User is sharing emotional content (sadness, anxiety, fear, anger)
- Crisis indicators present
- Biomarkers show distress (high jitter, crying, etc.)
- User is discussing trauma, triggers, or difficult topics
- There's a mismatch between words and biomarkers (masking)
- User seems to be escalating emotionally
- User is discussing self-harm, suicide, hopelessness

CAN BE SIMPLE when:
- User is asking a FACTUAL question ("what's my name?", "what do you remember?")
- User is making small talk or greeting
- User is asking for information
- User is responding to your question with simple info
- Message is clearly non-emotional

Output JSON: 
{{
    "needs_deep_assessment": true/false,
    "message_type": "FACTUAL_QUESTION/EMOTIONAL_SHARING/CRISIS/GREETING/SMALL_TALK/INFO_REQUEST/GENERAL",
    "reason": "brief explanation",
    "detected_signals": ["list", "of", "signals", "you", "noticed"],
    "use_biomarkers": true/false,
    "urgency":  "LOW/MEDIUM/HIGH/CRITICAL"
}}"""

        try:
            response = self._call_llm(prompt, max_tokens=300, temperature=0.3)
            result = self._parse_json_response(response)
            
            if result: 
                return result
        except Exception as e:
            self._print_warning("ROUTING", f"LLM error: {e}")
        
        # Fallback:  be safe, do deep assessment
        return {
            "needs_deep_assessment": True,
            "message_type": "GENERAL",
            "reason": "Fallback - defaulting to deep assessment",
            "detected_signals": [],
            "use_biomarkers": True,
            "urgency": "MEDIUM"
        }
    
    # =========================================================================
    # DEEP ASSESSMENT REASONING STEPS
    # =========================================================================
    def _track_conversation(self, session_id: str, user_text: str, response: str, turn_number: int):
        """Track conversation for immediate context."""
        if session_id not in self.conversation_history_by_session:
            self.conversation_history_by_session[session_id] = []
        
        self.conversation_history_by_session[session_id].append({
            'turn': turn_number,
            'user': user_text[:300],
            'assistant': response[:300],
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 turns
        self.conversation_history_by_session[session_id] = self.conversation_history_by_session[session_id][-5:]

    def _get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation history."""
        if session_id not in self.conversation_history_by_session:
            return "No previous turns in this session."
        
        history = self.conversation_history_by_session[session_id]
        if not history:
            return "No previous turns in this session."
        
        lines = []
        for turn in history[-3:]:  # Last 3 turns
            lines.append(f"[Turn {turn['turn']}]")
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")
        
        return "\n".join(lines)
    def _reason_understanding(
        self,
        text: str,
        biomarker: List[float],
        biomarker_summary: str,
        biomarker_delta: Dict,
        modality: str,
        user_facts: Dict[str, Any],
        memory_context: str,
        emotional_associations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Step 1: Understand what the user is communicating."""
        
        conv_history = self._build_conversation_context()
        name = user_facts.get('name', 'the user')
        
        # Build delta info
        delta_text = ""
        if biomarker_delta and biomarker_delta.get('has_delta'):
            changes = biomarker_delta.get('significant_changes', [])
            if changes:
                delta_text = "\n\nCHANGES FROM LAST TURN:"
                for c in changes[:3]:
                    delta_text += f"\n  - {c['feature']}: {c['interpretation']}"
        
        # FIX: Add explicit emotion detection from text
        text_lower = text.lower()
        explicit_emotions = []
        
        emotion_keywords = {
            'sad': ['sad', 'depressed', 'down', 'miserable', 'hopeless', 'crying', 'cry'],
            'anxious': ['anxious', 'worried', 'nervous', 'panic', 'scared', 'afraid', 'fear'],
            'angry': ['angry', 'furious', 'mad', 'pissed', 'frustrated', 'hate', 'hates'],
            'hurt': ['hurt', 'pain', 'ache', 'suffering', 'wounded'],  # NEW
            'lonely': ['lonely', 'alone', 'isolated', 'nobody', 'everyone hates'],  # UPDATED
            'rejected': ['reject', 'rejected', 'unwanted', 'unloved', 'called me'],  # NEW
            'happy': ['happy', 'great', 'wonderful', 'excited', 'joy'],
            'tired': ['tired', 'exhausted', 'drained', 'burnt out'],
            'stressed': ['stressed', 'overwhelmed', 'too much', 'cant handle']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                explicit_emotions.append(emotion)
        
        emotion_hint = ""
        if explicit_emotions:
            emotion_hint = f"\n\n⚠️ CRITICAL - EXPLICIT EMOTIONS DETECTED IN TEXT: {', '.join(explicit_emotions)}"
            emotion_hint += f"\nYou MUST use one of these emotions as the primary_emotion."
        
        prompt = f"""Analyze what {name} is communicating.

    === MESSAGE ===
    "{text}"
    {emotion_hint}

    === KNOWN FACTS ===
    {json.dumps(user_facts) if user_facts else "None yet"}

    === VOICE/BODY SIGNALS ({modality.upper()}) ===
    {biomarker_summary if biomarker_summary else "Text-only input"}
    {delta_text}

    === EMOTIONAL ASSOCIATIONS (from past conversations) ===
    {emotional_associations if emotional_associations else "None"}

    === MEMORY CONTEXT ===
    {memory_context[:800] if memory_context else "No previous context"}

    === CONVERSATION HISTORY ===
    {conv_history}

    CRITICAL: You MUST identify a specific emotion, not "uncertain". Use the text content and biomarkers.

    Analyze deeply. What are they REALLY saying? What might they be hiding?

    Output JSON:
    {{
        "literal_message": "what they explicitly said",
        "primary_emotion": "MUST BE SPECIFIC: sad/anxious/angry/happy/lonely/stressed/tired/overwhelmed/fearful/etc",
        "secondary_emotions": ["other", "emotions"],
        "intensity": 0.0 to 1.0,
        "underlying_need": "what they actually need",
        "what_theyre_not_saying": "subtext or hidden meaning",
        "biomarker_insights": "what the voice/body reveals",
        "text_biomarker_match": true/false,
        "minimizing": true/false,
        "reasoning": "your analysis"
    }}"""

        try:
            response = self._call_llm(prompt, max_tokens=500, temperature=0.3)
            result = self._parse_json_response(response)
            if result:
                # FIX: Validate emotion is not "uncertain"
                # FIX: Validate emotion is not "uncertain"
                if result.get('primary_emotion') in ['uncertain', 'unknown', 'unclear', 'neutral']:
                    if explicit_emotions:
                        result['primary_emotion'] = explicit_emotions[0]
                    else:
                        # Infer from text content
                        text_lower = text.lower()
                        if any(w in text_lower for w in ['hate', 'hates', 'reject', 'everyone']):
                            result['primary_emotion'] = 'hurt'
                        elif any(w in text_lower for w in ['sad', 'cry', 'depressed']):
                            result['primary_emotion'] = 'sad'
                        elif len(biomarker) > 0 and biomarker[0] > 0.3:
                            result['primary_emotion'] = 'anxious'
                        elif len(biomarker) > 22 and biomarker[22] < -0.3:
                            result['primary_emotion'] = 'sad'
                        else:
                            result['primary_emotion'] = 'neutral'
                
                result['user_name'] = user_facts.get('name')
                return result
        except Exception as e:
            self._print_warning("UNDERSTANDING", f"Error: {e}")
        
        # Fallback with explicit emotion detection
        fallback_emotion = explicit_emotions[0] if explicit_emotions else 'neutral'
        
        return {
            "literal_message": text,
            "primary_emotion": fallback_emotion,
            "intensity": 0.5,
            "underlying_need": "support",
            "user_name": user_facts.get('name'),
            "reasoning": "Fallback analysis"
        }
    
    # REPLACE _reason_signal_integration (around line 600) with:

    def _reason_signal_integration(
        self,
        text: str,
        biomarker: List[float],
        biomarker_summary: str,
        understanding: Dict,
        trigger_alerts: List[Dict],
        emotional_associations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Step 2: Integrate all signals to find the true emotional state."""
        
        signals = self.session_state["cumulative_signals"]
        
        # FIX: Pre-detect masking from biomarkers
        potential_masking = False
        masking_type = "none"
        
        if len(biomarker) >= 23:
            # High jitter + positive sentiment = anxiety masking
            if biomarker[0] > 0.4 and biomarker[22] > 0.2:
                potential_masking = True
                masking_type = "anxiety_behind_positivity"
            # Flat F0 + neutral/positive text = depression masking
            elif biomarker[2] < -0.3 and biomarker[22] > -0.1:
                potential_masking = True
                masking_type = "depression_behind_normalcy"
        
        masking_hint = ""
        if potential_masking:
            masking_hint = f"\n\nPOTENTIAL MASKING DETECTED: {masking_type}"
        
        prompt = f"""Integrate ALL signals to understand the TRUE emotional state.

    === MESSAGE ===
    "{text}"

    === UNDERSTANDING ===
    Primary emotion: {understanding.get('primary_emotion', 'unknown')}
    Intensity: {understanding.get('intensity', 0.5)}
    What they're not saying: {understanding.get('what_theyre_not_saying', 'none')}
    Text-biomarker match: {understanding.get('text_biomarker_match', True)}

    === BIOMARKER SIGNALS ===
    {biomarker_summary if biomarker_summary else "Text-only"}
    {masking_hint}

    === CUMULATIVE SESSION SIGNALS ===
    Distress level: {signals.get('distress_level', 0):.2f}
    Anxiety indicators: {signals.get('anxiety_indicators', 0):.2f}
    Depression indicators: {signals.get('depression_indicators', 0):.2f}
    Crying confidence: {signals.get('crying_confidence', 0):.2f}
    Laughter confidence: {signals.get('laughter_confidence', 0):.2f}

    === TRIGGER ALERTS ===
    {[a['entity'] for a in trigger_alerts] if trigger_alerts else "None"}

    === EMOTIONAL ASSOCIATIONS ===
    {emotional_associations if emotional_associations else "None"}

    CRITICAL: The integrated_emotional_state MUST be a specific emotion, not "uncertain".

    Output JSON:
    {{
        "integrated_emotional_state": "SPECIFIC emotion (sad/anxious/angry/happy/etc)",
        "confidence": 0.0 to 1.0,
        "text_vs_biomarker_match": true/false,
        "is_masking": {str(potential_masking).lower()},
        "masking_type": "{masking_type}",
        "hidden_distress": true/false,
        "escalation_detected": true/false,
        "key_signals": ["most", "important", "signals"],
        "synthesis": "overall synthesis"
    }}"""

        try:
            response = self._call_llm(prompt, max_tokens=400, temperature=0.3)
            result = self._parse_json_response(response)
            if result:
                # FIX: Validate integrated state
                if result.get('integrated_emotional_state') in ['uncertain', 'unknown']:
                    result['integrated_emotional_state'] = understanding.get('primary_emotion', 'neutral')
                return result
        except Exception as e:
            self._print_warning("SIGNALS", f"Error: {e}")
        
        return {
            "integrated_emotional_state": understanding.get('primary_emotion', 'neutral'),
            "confidence": 0.5,
            "is_masking": potential_masking,
            "masking_type": masking_type,
            "hidden_distress": False,
            "synthesis": "Fallback integration"
        }
    
    def _reason_risk_assessment(
        self,
        text:  str,
        understanding: Dict,
        signal_analysis: Dict,
        safety_details: Dict,
        trigger_alerts: List[Dict]
    ) -> Dict[str, Any]:
        """Step 3: Assess risk level."""
        
        crisis_score = safety_details.get('score', 0)
        matched_patterns = [p['pattern'] for p in safety_details.get('matched_patterns', [])]
        
        prompt = f"""Assess the risk level for this interaction.

=== MESSAGE ===
"{text}"

=== SAFETY CHECK RESULTS ===
Crisis score: {crisis_score:.2f}
Matched patterns: {matched_patterns}
Escalation:  {safety_details.get('escalation_detected', False)}

=== UNDERSTANDING ===
Emotion: {understanding.get('primary_emotion', 'unknown')}
Intensity: {understanding.get('intensity', 0.5)}
What they're not saying: {understanding.get('what_theyre_not_saying', 'none')}

=== SIGNAL ANALYSIS ===
Integrated state: {signal_analysis.get('integrated_emotional_state', 'unknown')}
Hidden distress: {signal_analysis. get('hidden_distress', False)}
Is masking: {signal_analysis. get('is_masking', False)}
Escalation: {signal_analysis.get('escalation_detected', False)}

=== TRIGGERS ===
{[a['entity'] for a in trigger_alerts] if trigger_alerts else "None"}

=== SESSION HISTORY ===
Crisis turns: {self.session_state['crisis_turns']}
Last crisis turn: {self.session_state['last_crisis_turn']}

Assess risk.  Be appropriately cautious but don't over-pathologize.

Output JSON: 
{{
    "risk_level":  "LOW/MODERATE/HIGH/CRITICAL",
    "is_crisis": true/false,
    "crisis_score": 0.0 to 1.0,
    "danger_signs": ["list", "of", "signs"],
    "protective_factors": ["list", "of", "factors"],
    "reasoning": "explanation"
}}"""

        try: 
            response = self._call_llm(prompt, max_tokens=400, temperature=0.3)
            result = self._parse_json_response(response)
            if result:
                return result
        except Exception as e: 
            self._print_warning("RISK", f"Error: {e}")
        
        # Fallback based on safety score
        if crisis_score >= 0.7:
            return {
                "risk_level": "HIGH",
                "is_crisis": True,
                "crisis_score": crisis_score,
                "reasoning": "High safety score"
            }
        elif crisis_score >= 0.5:
            return {
                "risk_level": "MODERATE",
                "is_crisis": False,
                "crisis_score":  crisis_score,
                "reasoning": "Moderate safety score"
            }
        return {
            "risk_level":  "LOW",
            "is_crisis": False,
            "crisis_score": crisis_score,
            "reasoning": "Low safety score"
        }
    
    def _reason_response_plan(
        self,
        text: str,
        understanding:  Dict,
        signal_analysis:  Dict,
        risk_assessment: Dict,
        user_facts: Dict[str, Any],
        memory_context: str,
        recommended_intervention: Optional[str],
        message_type: str
    ) -> Dict[str, Any]:
        """Step 4: Plan the response strategy."""
        
        is_crisis = risk_assessment.get('is_crisis', False)
        questions_asked = self.session_state["questions_asked_last_3_turns"]
        resources_given_recently = (
            self.session_state["resources_given_turn"] >= 0 and
            self.session_state["turn_count"] - self.session_state["resources_given_turn"] <= 3
        )
        
        prompt = f"""Plan how to respond to this user. 

=== MESSAGE ===
"{text}"

=== MESSAGE TYPE ===
{message_type}

=== UNDERSTANDING ===
Emotion: {understanding.get('primary_emotion', 'unknown')}
Intensity: {understanding.get('intensity', 0.5)}
What they need: {understanding.get('underlying_need', 'support')}

=== SIGNAL ANALYSIS ===
Is masking: {signal_analysis.get('is_masking', False)}
Hidden distress: {signal_analysis. get('hidden_distress', False)}

=== RISK ===
Level: {risk_assessment.get('risk_level', 'LOW')}
Is crisis: {is_crisis}

=== USER INFO ===
Name: {user_facts.get('name', 'unknown')}
Style: {self.session_state['user_language_style']}
Uses profanity: {self.session_state['user_uses_profanity']}

=== SESSION STATE ===
Questions asked recently: {questions_asked}
Resources given recently: {resources_given_recently}
Turn:  {self.session_state['turn_count'] + 1}

=== RECOMMENDED INTERVENTION ===
{recommended_intervention if recommended_intervention else "None"}

=== RULES ===
1. If FACTUAL_QUESTION: Answer directly, don't psychoanalyze
2. If masking:  Gently acknowledge what you sense
3. If crisis: Care first, then resources (988/741741)
4. Don't ask too many questions if you've asked many recently
5. Match their communication style
6. Use their name naturally if known

Output JSON:
{{
    "response_type": "DIRECT_ANSWER/VALIDATION/SUPPORT/GENTLE_EXPLORATION/CRISIS_SUPPORT/GROUNDING/COGNITIVE_REFRAME",
    "tone": "specific tone",
    "should_use_name": true/false,
    "should_reference_biomarkers": true/false,
    "biomarker_reference": "what to say about what you sense (if applicable)",
    "should_ask_question": true/false,
    "question_if_yes":  "the question",
    "should_give_resources": true/false,
    "key_points": ["what", "to", "include"],
    "must_avoid":  ["what", "to", "avoid"],
    "opening_approach": "how to start",
    "reasoning": "why this approach"
}}"""

        try:
            response = self._call_llm(prompt, max_tokens=500, temperature=0.4)
            result = self._parse_json_response(response)
            if result:
                result['user_name'] = user_facts.get('name')
                return result
        except Exception as e:
            self._print_warning("PLANNING", f"Error: {e}")
        
        return {
            "response_type": "CRISIS_SUPPORT" if is_crisis else "SUPPORT",
            "tone": "caring and direct" if is_crisis else "warm",
            "should_use_name": user_facts.get('name') is not None,
            "user_name": user_facts.get('name'),
            "should_give_resources": is_crisis,
            "should_ask_question": not is_crisis and questions_asked < 2
        }
    
    def _generate_deep_response(
        self,
        text: str,
        understanding: Dict,
        signal_analysis: Dict,
        risk_assessment: Dict,
        response_plan: Dict,
        user_facts: Dict[str, Any],
        biomarker_summary: str,
        memory_context: str,
        modality_context: str
    ) -> str:
        """Generate response using deep assessment results."""
        
        is_crisis = risk_assessment.get('is_crisis', False)
        user_name = user_facts.get('name')
        
        intensity = understanding.get('intensity', 0.5)
        if not isinstance(intensity, (int, float)):
            intensity = 0.5
        
        prompt = f"""You are Kairos, a deeply empathetic friend with insight into emotional states.

=== USER SAID ===
"{text}"

=== WHAT YOU UNDERSTAND ===
Emotion: {understanding.get('primary_emotion', 'unknown')} (intensity: {int(intensity * 100)}%)
What they need: {understanding.get('underlying_need', 'support')}
What they're not saying: {understanding.get('what_theyre_not_saying', 'none')}
Biomarker insights: {understanding.get('biomarker_insights', 'N/A')}

=== SIGNAL ANALYSIS ===
Is masking: {signal_analysis.get('is_masking', False)}
Masking type: {signal_analysis. get('masking_type', 'none')}
Hidden distress: {signal_analysis.get('hidden_distress', False)}

=== RISK ===
Level: {risk_assessment.get('risk_level', 'LOW')}
Is crisis: {is_crisis}

=== YOUR PLAN ===
Response type: {response_plan.get('response_type', 'SUPPORT')}
Tone: {response_plan.get('tone', 'warm')}
Opening:  {response_plan.get('opening_approach', 'acknowledge feelings')}
Key points: {response_plan.get('key_points', [])}
Must avoid: {response_plan.get('must_avoid', [])}
{"Reference biomarkers: " + response_plan.get('biomarker_reference', '') if response_plan.get('should_reference_biomarkers') else ""}
{"Ask:  " + response_plan.get('question_if_yes', '') if response_plan.get('should_ask_question') else "Don't ask questions"}
{"Include 988/741741 crisis resources" if response_plan.get('should_give_resources') else "No crisis resources needed"}

=== USER INFO ===
{"Name: " + user_name + " (use it naturally)" if user_name else "Name unknown"}
Style: {self.session_state['user_language_style']}

=== CRITICAL RULES ===
1. Sound like a REAL FRIEND, not a therapist
2. 2-4 sentences max (unless crisis needs more)
3. Don't be robotic or use clichés like "I hear you" or "That must be hard"
4. If masking detected, gently acknowledge:  "I sense there might be more beneath the surface..."
5. {"🚨 CRISIS:  Lead with immediate care, then resources" if is_crisis else ""}

Write your response (just the response, no explanation):"""

        try:
            response = self._call_llm(prompt, max_tokens=300, temperature=0.7)
            return self._clean_response(response)
        except Exception as e:
            self._print_warning("RESPONSE", f"Error: {e}")
            return self._fallback_response(is_crisis, user_name)
    
    # =========================================================================
    # SIMPLE RESPONSE PATH
    # =========================================================================
    def _generate_simple_response(
        self,
        text: str,
        message_type: str,
        user_facts: Dict[str, Any],
        memory_context: str,
        modality_context:  str,
        biomarker_summary:  Optional[str]
    ) -> Dict[str, Any]:
        """Generate a simple/direct response without deep assessment."""
        
        user_name = user_facts.get('name')
        
        prompt = f"""You are Kairos, a helpful and friendly companion.

=== USER SAID ===
"{text}"

=== MESSAGE TYPE ===
{message_type}

=== USER INFO ===
{json.dumps(user_facts) if user_facts else "No facts stored yet"}

=== MEMORY CONTEXT ===
{memory_context[: 600] if memory_context else "No previous context"}

{f"=== MODALITY CONTEXT ===" + chr(10) + modality_context[: 400] if modality_context else ""}

=== RULES FOR {message_type} ===
{"FACTUAL_QUESTION:  Answer DIRECTLY.  If they ask 'what's my name' - tell them or say you don't know.  If they ask 'what do you remember' - summarize what you know." if message_type == "FACTUAL_QUESTION" else ""}
{"GREETING: Warm welcome, invite them to share what's on their mind." if message_type == "GREETING" else ""}
{"SMALL_TALK: Be friendly, engage naturally." if message_type == "SMALL_TALK" else ""}
{"INFO_REQUEST:  Provide the information if you have it, or say you don't know." if message_type == "INFO_REQUEST" else ""}

=== CRITICAL ===
1. Be direct and helpful
2. {"Use their name (" + user_name + ") naturally" if user_name else "Be warm"}
3. Don't over-psychoanalyze simple questions
4. 1-3 sentences

Output JSON:
{{
    "response":  "your response",
    "response_type": "DIRECT_ANSWER/GREETING/SMALL_TALK/INFO",
    "detected_emotion": "emotion if any",
    "intensity": 0.0 to 0.5
}}"""

        try:
            response = self._call_llm(prompt, max_tokens=300, temperature=0.7)
            result = self._parse_json_response(response)
            
            if result and result.get('response'):
                return result
        except Exception as e:
            self._print_warning("SIMPLE", f"Error: {e}")
        
        # Fallback
        if message_type == "GREETING":
            greeting = f"Hey{' ' + user_name if user_name else ''}! What's on your mind?"
            return {"response": greeting, "response_type": "GREETING", "detected_emotion": "neutral", "intensity": 0.2}
        
        if message_type == "FACTUAL_QUESTION":
            if user_facts: 
                facts_summary = ", ".join([f"{k}: {v}" for k, v in user_facts. items() if v])
                return {
                    "response": f"Here's what I know: {facts_summary}.  What else would you like to talk about?",
                    "response_type": "DIRECT_ANSWER",
                    "detected_emotion":  "curious",
                    "intensity": 0.2
                }
            return {
                "response": "We're still getting to know each other - I don't have much stored yet.  Tell me about yourself!",
                "response_type": "DIRECT_ANSWER",
                "detected_emotion":  "curious",
                "intensity":  0.2
            }
        
        return {
            "response": f"{'Hey ' + user_name + ', ' if user_name else ''}I'm here.  What's going on? ",
            "response_type":  "SUPPORT",
            "detected_emotion":  "neutral",
            "intensity": 0.3
        }
    
    # =========================================================================
    # POST-PROCESSING
    # =========================================================================
    def _post_process_response(self, response: str, is_crisis: bool, risk_level: str) -> str:
        """Post-process and verify response."""
        response = self._clean_response(response)
        
        # Ensure crisis resources
        if is_crisis and "988" not in response and "741741" not in response: 
            response += "\n\nIf you need support right now:  988 (call or text) or text HOME to 741741."
        
        # Track questions
        question_count = response.count('?')
        self.session_state["questions_asked_last_3_turns"] += question_count
        self.session_state["last_response_had_question"] = question_count > 0
        
        # Track resources
        if "988" in response or "741741" in response: 
            self.session_state["resources_given_turn"] = self.session_state["turn_count"]
        
        return response. strip()
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _save_turn(
        self,
        text: str,
        response: str,
        feature_data: Dict,
        entities: List[str],
        is_crisis: bool,
        crisis_score: float,
        emotional_state: str,
        intensity: float,
        risk_level: str,
        intervention_type: str,
        modality: str,
        feature_insights: str,
        trigger_alerts: List[Dict]
    ):
        """Save turn to memory and update state."""
        current_turn = self.session_state["turn_count"]
        biomarker = feature_data['vectors']['biomarker']
        
        # Record intervention outcome
        if self.previous_biomarker is not None and self.previous_intervention_type is not None:
            self.memory. save_intervention_outcome(
                pre_biomarker=self.previous_biomarker,
                post_biomarker=biomarker,
                intervention_type=self.previous_intervention_type,
                user_emotion=emotional_state,
                session_id=self.session_id
            )
        
        # Store for next turn
        self.previous_biomarker = biomarker. copy() if isinstance(biomarker, list) else list(biomarker)
        self.previous_intervention_type = intervention_type
        self.previous_emotion = emotional_state
        
        # Update entity graph
        for entity in entities:
            self.entity_graph.update_entity(
                entity=entity,
                emotion=emotional_state,
                is_crisis=is_crisis,
                is_positive=emotional_state in ['joy', 'contentment', 'happiness'],
                context=text[: 200],
                co_occurring_entities=[e for e in entities if e != entity]
            )
        # CRITICAL: Force name extraction check
        if hasattr(self, 'user_profile'):
            if not self.user_profile.get_name():
                name_check = self.user_profile.extract_facts_from_text(text, current_turn, response)
                if name_check:
                    print(f"  [PROFILE] ✅ Extracted facts: {[f['type'] for f in name_check]}")
                
        self.memory.save_interaction(
            user_text=text,
            system_response=response,
            vectors=feature_data['vectors'],
            payload={
                **feature_data['payload'],
                "turn_number": current_turn,
                "crisis_score": crisis_score,
                "is_crisis": is_crisis,
                "detected_emotion": emotional_state,
                "emotional_intensity": intensity,
                "risk_level": risk_level,
                "intervention_type": intervention_type,
                "entities": entities,
                "trigger_entities": [a['entity'] for a in (trigger_alerts or [])],
                "modality_type": modality,
                "feature_insights": feature_insights
            },
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Update conversation history
        self.conversation_history.append({
            "user":  text,
            "assistant": response,
            "turn":  current_turn,
            "is_crisis": is_crisis,
            "emotion": emotional_state,
            "intervention_type": intervention_type,
            "modality": modality
        })
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Update session state
        self.session_state["turn_count"] += 1
        self.session_state["emotional_arc"]. append(emotional_state)
        self.session_state["intensity_arc"].append(intensity)
        self.session_state["risk_arc"].append(risk_level)
        self.session_state["interventions_used"].append(intervention_type)
        
        if is_crisis:
            self. session_state["crisis_turns"].append(current_turn)
            self.session_state["last_crisis_turn"] = current_turn
        
        # Decay question counter
        self.session_state["questions_asked_last_3_turns"] = max(
            0,
            self.session_state["questions_asked_last_3_turns"] - 0.5
        )
        
        # Track topics
        if len(text) > 20:
            self.session_state["topics_user_shared"].append(text[:100])
            self.session_state["topics_user_shared"] = self.session_state["topics_user_shared"][-20:]
    
    # In Orchestrator class - ADD this method

    def _create_detailed_biomarker_summary(
        self,
        bio_vector: np.ndarray,
        reliability_mask: np.ndarray,
        laughter_score: float
    ) -> str:
        """Create detailed, interpretable biomarker summary."""
        
        if len(bio_vector) < 8:
            return "Insufficient biomarker data."
        
        parts = []
        
        # Acoustic features (0-7)
        if reliability_mask[0] > 0.5:
            jitter = bio_vector[0]
            shimmer = bio_vector[1]
            pitch_var = bio_vector[2]
            energy = bio_vector[3]
            
            if abs(jitter) > 0.3:
                parts.append(f"🔴 JITTER: {jitter:+.2f} ({'HIGH anxiety/stress' if jitter > 0 else 'LOW'})")
            
            if abs(shimmer) > 0.3:
                parts.append(f"🔴 SHIMMER: {shimmer:+.2f} ({'Voice instability' if shimmer > 0 else 'Stable'})")
            
            if pitch_var < -0.2:
                parts.append(f"🔴 FLAT AFFECT: pitch_var={pitch_var:.2f} (Depression indicator)")
            elif pitch_var > 0.3:
                parts.append(f"🟢 ANIMATED: pitch_var={pitch_var:.2f} (Engaged/aroused)")
            
            if abs(energy) > 0.3:
                parts.append(f"ENERGY: {energy:+.2f} ({'High' if energy > 0 else 'Low/exhausted'})")
        
        # Linguistic features (16-23)
        if len(bio_vector) > 22 and reliability_mask[22] > 0.5:
            sentiment = bio_vector[22]
            if abs(sentiment) > 0.3:
                parts.append(f"SENTIMENT: {sentiment:+.2f} ({'Positive' if sentiment > 0 else 'Negative'})")
        
        # Special markers (24-31)
        if len(bio_vector) > 25:
            crying = bio_vector[25]
            if crying > 0.4:
                parts.append(f"🔴 CRYING DETECTED: {crying:.2f}")
        
        if laughter_score > 0.5:
            parts.append(f"🟢 LAUGHTER: {laughter_score:.2f} (Genuine positive emotion)")
        
        if not parts:
            return "No significant biomarker deviations detected (normal baseline)."
        
        return "\n".join(parts)
    def _build_conversation_context(self) -> str:
        """Build conversation history."""
        if not self.conversation_history:
            return "This is the start of the conversation."
        
        lines = []
        for ex in self.conversation_history[-5:]:
            turn = ex. get('turn', 0)
            modality = ex.get('modality', 'text')
            modality_marker = f"[{modality. upper()}] " if modality != 'text' else ""
            
            lines.append(f"[Turn {turn + 1}] {modality_marker}")
            lines.append(f"  User: {ex. get('user', '')[: 150]}{'...' if len(ex.get('user', '')) > 150 else ''}")
            lines.append(f"  Kairos: {ex.get('assistant', '')[:150]}{'...' if len(ex.get('assistant', '')) > 150 else ''}")
            if ex.get('is_crisis'):
                lines.append("  [CRISIS]")
            lines.append("")
        
        return "\n".join(lines)
    
    def _analyze_user_style(self, text: str):
        """Analyze user's language style."""
        text_lower = text.lower()
        
        profanity = ['fuck', 'shit', 'damn', 'hell', 'ass', 'crap']
        if any(p in text_lower for p in profanity):
            self.session_state["user_uses_profanity"] = True
        
        casual = ['bro', 'dude', 'man', 'like', 'yeah', 'nah', 'gonna', 'wanna']
        if sum(1 for c in casual if c in text_lower) >= 2:
            self.session_state["user_language_style"] = "casual"
        elif len(text. split()) > 20:
            self.session_state["user_language_style"] = "formal"
        else: 
            self.session_state["user_language_style"] = "mixed"
    
    def _update_persistent_signals(self, biomarker:  List[float], reliability: List[float], modality: str):
        """Update persistent signals."""
        signals = self.session_state["cumulative_signals"]
        
        # Decay
        for key in signals:
            signals[key] *= 0.85
        
        if modality == "text":
            return
        
        if len(biomarker) >= 8:
            if biomarker[0] > 0.3:
                signals["anxiety_indicators"] += 0.2
            if biomarker[2] < -0.3:
                signals["depression_indicators"] += 0.25
            if biomarker[1] > 0.5:
                signals["distress_level"] += 0.2
        
        if len(biomarker) > 25:
            alpha = 0.4
            signals["laughter_confidence"] = (1 - alpha) * signals["laughter_confidence"] + alpha * biomarker[24]
            signals["crying_confidence"] = (1 - alpha) * signals["crying_confidence"] + alpha * biomarker[25]
            
            if biomarker[25] > 0.5:
                signals["distress_level"] += 0.3
            if biomarker[24] > 0.6:
                signals["positive_level"] += 0.25
        
        for key in signals:
            signals[key] = max(0.0, min(1.0, signals[key]))
    
    def _create_feature_insights(self, biomarker: List[float], modality: str) -> str:
        """Create feature insights string."""
        if modality == 'text':
            return "Text-only input."
        
        insights = []
        if len(biomarker) >= 8:
            if biomarker[0] > 0.3:
                insights. append(f"voice tremor ({biomarker[0]:.2f})")
            if biomarker[1] > 0.3:
                insights.append(f"voice instability ({biomarker[1]:.2f})")
            if biomarker[2] < -0.3:
                insights.append(f"flat pitch ({biomarker[2]:.2f})")
            if biomarker[6] > 0.3:
                insights.append(f"fast speech ({biomarker[6]:.2f})")
            if biomarker[7] > 0.3:
                insights. append(f"frequent pauses ({biomarker[7]:.2f})")
        
        if len(biomarker) > 25:
            if biomarker[24] > 0.3:
                insights.append(f"laughter ({biomarker[24]:.2f})")
            if biomarker[25] > 0.3:
                insights. append(f"crying ({biomarker[25]:.2f})")
        
        return "; ".join(insights) if insights else "No notable features."
    
    def _fallback_response(self, is_crisis: bool, user_name:  Optional[str]) -> str:
        """Fallback response."""
        prefix = f"{user_name}, " if user_name else ""
        if is_crisis:
            return f"{prefix}I'm here with you. Please reach out:  988 or text HOME to 741741."
        return f"{prefix}I'm here.  What's on your mind?"
    
    def _call_llm(self, prompt:  str, max_tokens: int = 500, temperature: float = 0.5) -> str:
        """Call LLM with fallback."""
        try:
            response = self. llm.chat.completions.create(
                model=self.model,
                messages=[{"role":  "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0]. message.content. strip()
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                try:
                    response = self. llm.chat.completions.create(
                        model=self.fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response. choices[0].message.content. strip()
                except: 
                    raise e
            raise e
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from response."""
        try:
            return json.loads(response)
        except: 
            pass
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response: 
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
        except:
            pass
        
        try:
            start = response.find('{')
            end = response. rfind('}') + 1
            if start >= 0 and end > start: 
                return json.loads(response[start:end])
        except:
            pass
        
        return None
    
    def _clean_response(self, response: str) -> str:
        """Clean response."""
        prefixes = ["Kairos:", "Response:", "Here's my response:", "As Kairos,", "*", "**"]
        for prefix in prefixes:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix):].strip()
        
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        response = response.replace("**", "").replace("*", "")
        return " ".join(response.split()).strip()
    
    # =========================================================================
    # DEBUG PRINTING HELPERS
    # =========================================================================
    def _print_header(self, title: str):
        """Print section header."""
        if self.debug_mode:
            print("\n" + "=" * 60)
            print(f"  {title}")
            print("=" * 60)
    
    def _print_step(self, step: str, message: str):
        """Print step info."""
        if self.debug_mode:
            print(f"\n  [{step}] {message}")
    
    def _print_debug(self, category: str, message: str):
        """Print debug info."""
        if self.debug_mode:
            print(f"    [{category}] {message}")
    
    def _print_warning(self, category: str, message: str):
        """Print warning."""
        if self.debug_mode:
            print(f"    ⚠️ [{category}] {message}")
    
    def _print_summary(self, emotional_state: str, intensity: float, risk_level: str, is_crisis: bool, path: str):
        """Print turn summary."""
        if self.debug_mode:
            print("\n" + "-" * 60)
            print(f"  SUMMARY:")
            print(f"    Path: {path.upper()}")
            
            # FIX: Better emotion display
            emotion_display = emotional_state.upper() if emotional_state else "UNKNOWN"
            if emotion_display == "UNCERTAIN":
                emotion_display = "NEUTRAL (uncertain)"
            
            print(f"    Emotion: {emotion_display} | Intensity: {int(intensity * 100)}%")
            print(f"    Risk: {risk_level} | Crisis: {is_crisis}")
            print(f"    Deep assessments this session: {self.session_state['deep_assessments_done']}")
            print(f"    Simple responses this session: {self.session_state['simple_responses_done']}")
            print("-" * 60)
        
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "turns":  self.session_state["turn_count"],
            "emotional_arc": self.session_state["emotional_arc"],
            "intensity_arc": self.session_state["intensity_arc"],
            "risk_arc": self.session_state["risk_arc"],
            "crisis_count": len(self.session_state["crisis_turns"]),
            "crisis_turns": self.session_state["crisis_turns"],
            "interventions_used": self.session_state["interventions_used"],
            "trigger_alerts": self.session_state["trigger_alerts_this_session"],
            "trajectory_alerts": self.session_state["trajectory_alerts"],
            "modalities_used": list(self.session_state["modalities_used"]),
            "deep_assessments":  self.session_state["deep_assessments_done"],
            "simple_responses": self.session_state["simple_responses_done"],
            "user_facts": self.memory.get_user_facts(),
            "cumulative_signals": self.session_state["cumulative_signals"],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_session(self):
        """Save session to memory."""
        summary = self.get_session_summary()
        self.memory.save_session_summary(self.session_id, summary)
        self.entity_graph.force_save()
        self.graph_rag.force_save()
        self.modality_tracker.force_save()
        print(f"  ✅ Session saved to memory")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug info for all features."""
        return {
            "session_state": {
                "turn_count": self.session_state["turn_count"],
                "emotional_arc":  self.session_state["emotional_arc"],
                "risk_arc": self.session_state["risk_arc"],
                "crisis_turns": self.session_state["crisis_turns"],
                "deep_assessments":  self.session_state["deep_assessments_done"],
                "simple_responses": self.session_state["simple_responses_done"],
                "cumulative_signals": self.session_state["cumulative_signals"],
                "questions_asked_recently": self.session_state["questions_asked_last_3_turns"],
                "user_style": self.session_state["user_language_style"],
            },
            "modalities_used": list(self. session_state. get("modalities_used", set())),
            "user_profile": self.user_profile. get_stats(),
            "graph_rag": self.graph_rag.get_stats(),
            "entity_graph":  self.entity_graph.get_stats(),
            "modality_tracker": self.modality_tracker.get_stats(),
            "salience_tracker": self.memory.salience_tracker.get_session_stats(),
            "session_anchors": self.memory.session_anchors.get_stats(),
            "intervention_tracker": self.memory.intervention_tracker.get_stats(),
            "safety_checker": self.safety_checker.get_stats(),
            "memory_stats": {
                "total_memories": self.memory.total_memories,
                "session_memories": self.memory.session_memories,
            },
            "conversation_history_length": len(self.conversation_history)
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable/disable debug mode."""
        self. debug_mode = enabled
        self.memory.set_debug_mode(enabled)
    
    def print_feature_summary(self):
        """Print comprehensive summary of all active features."""
        print("\n" + "=" * 70)
        print("  KAIROS FEATURE SUMMARY")
        print("=" * 70)
        
        # User Profile
        print("\n  📋 USER PROFILE (World State)")
        profile_stats = self.user_profile. get_stats()
        print(f"    ├─ Has name: {profile_stats.get('has_name', False)}")
        if profile_stats.get('name'):
            print(f"    ├─ Name: {profile_stats['name']}")
        print(f"    ├─ Total facts: {profile_stats.get('total_facts', 0)}")
        print(f"    └─ Facts:  {list(profile_stats.get('facts', {}).keys())[:5]}")
        
        # Graph-RAG
        print("\n  🔗 GRAPH-RAG (Associative Memory)")
        graph_stats = self.graph_rag. get_stats()
        print(f"    ├─ Nodes: {graph_stats.get('total_nodes', 0)}")
        print(f"    ├─ Edges: {graph_stats.get('total_edges', 0)}")
        print(f"    ├─ Topics: {graph_stats.get('total_topics', 0)}")
        print(f"    ├─ Triggers tracked: {graph_stats.get('trigger_count', 0)}")
        if graph_stats.get('top_entities'):
            print(f"    └─ Top entities: {graph_stats['top_entities'][: 5]}")
        else:
            print(f"    └─ Top entities: None yet")
        
        # Entity Graph
        print("\n  🕸️ ENTITY GRAPH (Triggers & Relationships)")
        entity_stats = self.entity_graph.get_stats()
        print(f"    ├─ Total entities: {entity_stats.get('total_entities', 0)}")
        print(f"    ├─ Trigger entities: {entity_stats.get('trigger_count', 0)}")
        print(f"    ├─ Positive entities: {entity_stats.get('positive_count', 0)}")
        if entity_stats.get('top_triggers'):
            print(f"    └─ Top triggers: {entity_stats['top_triggers'][: 3]}")
        else:
            print(f"    └─ Top triggers: None yet")
        
        # Modality Tracking
        print("\n  🎙️ MODALITY TRACKING")
        modality_stats = self.modality_tracker.get_stats()
        for mod in ['audio', 'video', 'text']: 
            stats = modality_stats.get(mod, {})
            if stats.get('count', 0) > 0:
                print(f"    ├─ {mod. upper()}: {stats['count']} interactions")
                print(f"    │    ├─ Avg intensity: {stats. get('avg_intensity', 0):.2f}")
                print(f"    │    └─ Crisis moments: {stats.get('crisis_count', 0)}")
        print(f"    └─ Total:  {modality_stats.get('total_interactions', 0)} interactions")
        
        # Session Anchors
        print("\n  ⚓ SESSION ANCHORS")
        anchor_stats = self.memory.session_anchors.get_stats()
        print(f"    ├─ Anchors set: {anchor_stats.get('anchor_count', 0)}/{anchor_stats.get('max_anchors', 3)}")
        print(f"    ├─ Locked:  {anchor_stats.get('anchors_locked', False)}")
        if anchor_stats.get('anchors'):
            for i, anchor in enumerate(anchor_stats['anchors'][:2]):
                print(f"    │    └─ Anchor {i+1}: \"{anchor. get('user_text', '')[:40]}... \"")
        print(f"    └─ IDs: {anchor_stats.get('anchor_ids', [])}")
        
        # Salience Tracking
        print("\n  ⭐ SALIENCE TRACKING")
        salience_stats = self.memory.salience_tracker. get_session_stats()
        print(f"    ├─ Memories this session: {salience_stats. get('count', 0)}")
        print(f"    ├─ High salience (core): {salience_stats.get('high_salience_count', 0)}")
        print(f"    ├─ Avg salience: {salience_stats.get('avg_salience', 0):.2f}")
        print(f"    └─ Always retrieve: {salience_stats.get('always_retrieve_count', 0)}")
        
        # Intervention Learning
        print("\n  💊 INTERVENTION LEARNING")
        intervention_stats = self.memory. intervention_tracker.get_stats()
        print(f"    ├─ Interventions recorded: {intervention_stats.get('total_interventions', 0)}")
        print(f"    ├─ Effective: {intervention_stats.get('effective_count', 0)}")
        if intervention_stats.get('most_effective_intervention'):
            print(f"    ├─ Most effective: {intervention_stats['most_effective_intervention']}")
        if intervention_stats.get('intervention_success_rates'):
            print(f"    └─ Success rates: {intervention_stats['intervention_success_rates']}")
        else:
            print(f"    └─ Success rates: Not enough data")
        
        # Safety Checker
        print("\n  🛡️ SAFETY CHECKER")
        safety_stats = self. safety_checker.get_stats()
        print(f"    ├─ Total evaluations: {safety_stats.get('total_evaluations', 0)}")
        print(f"    ├─ Crisis detections: {safety_stats.get('crisis_count', 0)}")
        print(f"    ├─ Current threshold: {safety_stats.get('current_threshold', 0.7):.2f}")
        print(f"    └─ Recent scores: {[f'{s:.2f}' for s in safety_stats.get('recent_scores', [])]}")
        
        # Memory Stats
        print("\n  🧠 MEMORY")
        print(f"    ├─ Total memories: {self.memory. total_memories}")
        print(f"    ├─ Session memories: {self.memory.session_memories}")
        print(f"    └─ Conversation history: {len(self.conversation_history)} turns")
        
        # Routing Stats
        print("\n  🔀 ROUTING (this session)")
        print(f"    ├─ Deep assessments: {self.session_state['deep_assessments_done']}")
        print(f"    ├─ Simple responses:  {self.session_state['simple_responses_done']}")
        total = self.session_state['deep_assessments_done'] + self. session_state['simple_responses_done']
        if total > 0:
            deep_pct = (self.session_state['deep_assessments_done'] / total) * 100
            print(f"    └─ Deep assessment rate: {deep_pct:. 1f}%")
        
        print("\n" + "=" * 70 + "\n")
    
    def print_turn_debug(self, debug_info: Dict[str, Any]):
        """Print detailed debug info for a single turn."""
        if not self.debug_mode:
            return
        
        print("\n" + "=" * 70)
        print("  DETAILED TURN DEBUG")
        print("=" * 70)
        
        # Routing
        if debug_info.get('routing'):
            r = debug_info['routing']
            print("\n  📍 ROUTING DECISION")
            print(f"    ├─ Message type: {r.get('message_type', 'unknown')}")
            print(f"    ├─ Needs deep assessment: {r.get('needs_deep_assessment', 'unknown')}")
            print(f"    ├─ Reason: {r.get('reason', 'unknown')}")
            print(f"    ├─ Urgency: {r.get('urgency', 'unknown')}")
            print(f"    └─ Signals: {r.get('detected_signals', [])}")
        
        # Features
        if debug_info.get('feature_extraction'):
            f = debug_info['feature_extraction']
            print("\n  🎤 FEATURE EXTRACTION")
            print(f"    ├─ Modality: {f. get('modality', 'unknown')}")
            print(f"    ├─ Has delta: {f.get('has_delta', False)}")
            print(f"    ├─ Significant changes: {f.get('significant_changes', 0)}")
            print(f"    └─ Insights: {f. get('feature_insights', 'none')[:60]}")
        
        # Graph-RAG
        if debug_info.get('graph_rag'):
            g = debug_info['graph_rag']
            print("\n  🔗 GRAPH-RAG")
            print(f"    ├─ Original entities: {g.get('original_entities', [])}")
            print(f"    ├─ Related (spreading): {g.get('related_entities', [])[:5]}")
            print(f"    ├─ Emotional associations: {g.get('emotional_associations', {})}")
            print(f"    ├�� Topics: {g.get('topics', [])}")
            print(f"    └─ Triggers: {[a['entity'] for a in g.get('trigger_alerts', [])]}")
        
        # Safety
        if debug_info.get('safety'):
            s = debug_info['safety']
            print("\n  🛡️ SAFETY")
            print(f"    ├─ Crisis score: {s.get('crisis_score', 0):.2f}")
            print(f"    ├─ Is crisis: {s.get('is_crisis', False)}")
            print(f"    ├─ Matched patterns: {s.get('matched_patterns', [])[:3]}")
            print(f"    ├─ Sparse terms: {s.get('sparse_terms', [])[:3]}")
            print(f"    └─ Escalation: {s.get('escalation_detected', False)}")
        
        # Memory
        if debug_info.get('memory'):
            m = debug_info['memory']
            print("\n  🧠 MEMORY")
            print(f"    ├─ User facts: {list(m.get('user_facts', {}).keys())}")
            print(f"    ├─ Anchors retrieved: {len(m.get('anchor_memories', []))}")
            print(f"    ├─ High salience: {len(m.get('high_salience_memories', []))}")
            print(f"    └─ Total results: {m.get('total_results', 0)}")
        
        # Modality
        if debug_info. get('modality_tracking'):
            mt = debug_info['modality_tracking']
            print("\n  🎙️ MODALITY")
            print(f"    ├─ Current:  {mt.get('current_modality', 'text')}")
            print(f"    ├─ Context added: {mt.get('modality_context_added', False)}")
            print(f"    └─ Stats: {mt.get('stats', {})}")
        
        # Reasoning (if deep assessment)
        if debug_info.get('reasoning'):
            r = debug_info['reasoning']
            
            if r.get('understanding'):
                u = r['understanding']
                print("\n  🧐 UNDERSTANDING")
                print(f"    ├─ Primary emotion: {u.get('primary_emotion', 'unknown')}")
                print(f"    ├─ Intensity: {u.get('intensity', 0):.2f}")
                print(f"    ├─ Underlying need: {u.get('underlying_need', 'unknown')}")
                print(f"    ├─ What they're not saying: {u.get('what_theyre_not_saying', 'none')[:50]}")
                print(f"    └─ Biomarker insights: {u.get('biomarker_insights', 'N/A')[:50]}")
            
            if r.get('signal_analysis'):
                s = r['signal_analysis']
                print("\n  📊 SIGNAL INTEGRATION")
                print(f"    ├─ Integrated state: {s.get('integrated_emotional_state', 'unknown')}")
                print(f"    ├─ Is masking: {s.get('is_masking', False)}")
                print(f"    ├─ Masking type: {s.get('masking_type', 'none')}")
                print(f"    ├─ Hidden distress: {s.get('hidden_distress', False)}")
                print(f"    └─ Key signals: {s.get('key_signals', [])[:5]}")
            
            if r.get('risk_assessment'):
                ra = r['risk_assessment']
                print("\n  ⚠️ RISK ASSESSMENT")
                print(f"    ├─ Risk level: {ra.get('risk_level', 'unknown')}")
                print(f"    ├─ Is crisis: {ra.get('is_crisis', False)}")
                print(f"    ├─ Crisis score: {ra.get('crisis_score', 0):.2f}")
                print(f"    ├─ Danger signs: {ra.get('danger_signs', [])[:3]}")
                print(f"    └─ Protective factors: {ra.get('protective_factors', [])[:3]}")
            
            if r.get('response_plan'):
                rp = r['response_plan']
                print("\n  📝 RESPONSE PLAN")
                print(f"    ├─ Response type: {rp.get('response_type', 'unknown')}")
                print(f"    ├─ Tone: {rp.get('tone', 'unknown')}")
                print(f"    ├─ Use name: {rp.get('should_use_name', False)}")
                print(f"    ├─ Reference biomarkers: {rp.get('should_reference_biomarkers', False)}")
                print(f"    ├─ Ask question: {rp.get('should_ask_question', False)}")
                print(f"    ├─ Give resources: {rp.get('should_give_resources', False)}")
                print(f"    └─ Key points: {rp.get('key_points', [])[:3]}")
            
            if r.get('simple_response'):
                sr = r['simple_response']
                print("\n  💬 SIMPLE RESPONSE")
                print(f"    ├─ Response type: {sr.get('response_type', 'unknown')}")
                print(f"    ├─ Detected emotion: {sr.get('detected_emotion', 'unknown')}")
                print(f"    └─ Intensity: {sr. get('intensity', 0):.2f}")
        
        # Response
        if debug_info.get('response'):
            resp = debug_info['response']
            print("\n  💬 FINAL RESPONSE")
            print(f"    ├─ Path: {resp.get('path', 'unknown')}")
            print(f"    ├─ Emotional state: {resp.get('emotional_state', 'unknown')}")
            print(f"    ├─ Intensity: {resp.get('intensity', 0):.2f}")
            print(f"    ├─ Risk level: {resp.get('risk_level', 'unknown')}")
            print(f"    ├─ Is crisis: {resp.get('is_crisis', False)}")
            print(f"    ├─ Intervention type: {resp.get('intervention_type', 'unknown')}")
            print(f"    └─ Response length: {resp.get('response_length', 0)} chars")
        
        print("\n" + "=" * 70 + "\n")