"""
Strategic Planner (Prompt B) - ENHANCED VERSION
Integrates intervention effectiveness data and trajectory concerns.
"""
import json
from typing import Dict, Any, Optional, List

from groq import Groq


PLANNING_PROMPT = """SYSTEM:  You are the Therapeutic Strategist for Kairos.  

Based on the clinical diagnosis and all available context, select the best therapeutic approach. 

DIAGNOSIS:
- Emotion: {emotion} (intensity: {intensity})
- Risk: {risk_level}
- Masking: {is_masking}

BIOMARKER EVIDENCE:
{biomarker_summary}

{trajectory_alert}

{trigger_alert}

PAST EFFECTIVE STRATEGIES:
{effective_interventions}
ADDITIONAL CONTEXT:
- Sentiment Score: {sentiment_score}
- Laughter Detected: {laughter_detected}
- Input Modality: {modality}

=== PAST EFFECTIVE INTERVENTIONS ===
{effective_interventions}

=== TRAJECTORY ALERT ===
{trajectory_alert}

=== TRIGGER ENTITIES ===
{trigger_entities}

AVAILABLE STRATEGIES: 

1. VALIDATE_AND_AMPLIFY
   - When:  Genuine positive emotions, joy, achievements
   - Action: Celebrate with them, reinforce positive feelings

2. GENTLE_EXPLORATION
   - When: Masking detected, mixed signals, contradictions
   - Action: Gently probe deeper, create safe space for honesty

3. ACTIVE_SUPPORT
   - When: Clear distress, sadness, anxiety (not crisis)
   - Action: Provide empathy, validation, coping strategies

4. COGNITIVE_REFRAME
   - When: Negative thought patterns, rumination, catastrophizing
   - Action:  Gently challenge distortions, offer alternative perspectives

5. GROUNDING_PRESENCE
   - When: High anxiety, dissociation signs, overwhelm
   - Action: Bring to present moment, breathing exercises, sensory grounding

6. CRISIS_INTERVENTION
   - When: Risk level is CRITICAL or HIGH
   - Action: Safety assessment, de-escalation, professional referral

7. CURIOUS_ENGAGEMENT
   - When: Neutral state, general conversation, building rapport
   - Action: Show genuine interest, ask meaningful questions

SELECTION GUIDELINES:
- If past interventions were effective for similar states, consider using them
- If trajectory shows escalation, be more proactive with support
- If trigger entities are involved, approach with extra care
- If masking detected, use GENTLE_EXPLORATION to uncover true feelings

OUTPUT FORMAT (JSON only):
{{
    "primary_strategy": "strategy name",
    "secondary_strategy": "backup strategy or null",
    "tone": "specific tone description",
    "focus_areas": ["what to address"],
    "avoid":  ["what NOT to do"],
    "opening_approach": "how to start the response",
    "therapeutic_tools": ["specific techniques to use"],
    "biomarker_response": "how to address what biomarkers reveal",
    "trigger_handling": "how to handle trigger entities if present",
    "follow_up_intention": "what to explore next"
}}"""


class StrategicPlanner:
    """
    Enhanced Strategic Planner with: 
    - Intervention effectiveness integration
    - Trajectory-aware planning
    - Trigger entity handling
    """
    
    def __init__(self, llm_client:  Groq, model: str):
        self.llm = llm_client
        self.model = model
    
    def plan(
        self,
        diagnosis:  Dict[str, Any],
        sentiment_score: float,
        laughter_detected: bool,
        modality: str = "text",
        effective_interventions: List[Dict] = None,
        trajectory_alert: Dict = None,
        trigger_entities: List[str] = None
    ) -> Dict[str, Any]:
        """Create therapeutic strategy plan with all available context."""
        
        # Check for strategy override (crisis)
        if diagnosis.get("strategy_override") == "CRISIS_INTERVENTION":
            return self._crisis_plan()
        
        if diagnosis.get("risk_level") in ["HIGH", "CRITICAL"]:
            return self._crisis_plan()
        
        # Format effective interventions
        interventions_str = "No past intervention data available."
        if effective_interventions:
            interventions_str = "\n".join([
                f"- {i['intervention_type']}: success={i['success_score']:.2f}, emotion_match={i['emotion_match']}"
                for i in effective_interventions[: 3]
            ])
        
        # Format trajectory alert
        trajectory_str = "No trajectory concerns."
        if trajectory_alert: 
            trajectory_str = f"⚠️ {trajectory_alert['pattern']}:  {trajectory_alert['description']} (confidence: {trajectory_alert['confidence']:.2f})"
        
        # Format trigger entities
        triggers_str = "No trigger entities detected."
        if trigger_entities:
            triggers_str = f"⚠️ Triggers present: {', '.join(trigger_entities)}"
        
        sentiment_score_str = f"{sentiment_score:.2f}"
        
        prompt = PLANNING_PROMPT.format(
            diagnosis=json.dumps(diagnosis, indent=2),
            sentiment_score=sentiment_score_str,
            laughter_detected=laughter_detected,
            modality=modality,
            effective_interventions=interventions_str,
            trajectory_alert=trajectory_str,
            trigger_entities=triggers_str
        )
        
        try: 
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":  "You are a therapeutic planning system. Respond only with valid JSON. "},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            plan = json.loads(result_text)
            
            plan.setdefault("primary_strategy", "ACTIVE_SUPPORT")
            plan.setdefault("tone", "compassionate and warm")
            plan.setdefault("biomarker_response", "")
            plan.setdefault("trigger_handling", "")
            
            return plan
            
        except Exception as e:
            print(f"  Warning: Strategic planning error:  {e}")
            return self._default_plan(diagnosis, laughter_detected, trigger_entities)
    
    def _crisis_plan(self) -> Dict[str, Any]:
        """Return crisis intervention plan."""
        return {
            "primary_strategy":  "CRISIS_INTERVENTION",
            "secondary_strategy": None,
            "tone": "calm, caring, and direct",
            "focus_areas":  ["immediate safety", "de-escalation", "professional support"],
            "avoid": ["minimizing", "being preachy", "leaving them alone"],
            "opening_approach": "Express immediate care and concern",
            "therapeutic_tools": ["safety assessment", "grounding", "crisis resources"],
            "biomarker_response": "Acknowledge distress signals sensed",
            "trigger_handling":  "Validate sensitivity around triggers",
            "follow_up_intention": "ensure safety and connection to help"
        }
    
    def _default_plan(
        self,
        diagnosis: Dict[str, Any],
        laughter_detected: bool,
        trigger_entities: List[str] = None
    ) -> Dict[str, Any]:
        """Fallback plan based on basic heuristics."""
        emotion = diagnosis.get("suspected_emotion", "neutral")
        risk = diagnosis.get("risk_level", "LOW")
        is_masking = diagnosis.get("is_masking", False)
        
        if risk in ["HIGH", "CRITICAL"]: 
            return self._crisis_plan()
        
        if laughter_detected and emotion in ["joy", "contentment", "neutral"]:
            return {
                "primary_strategy": "VALIDATE_AND_AMPLIFY",
                "tone": "warm and enthusiastic",
                "focus_areas": ["celebrate the positive moment"],
                "avoid": ["being dismissive", "changing subject abruptly"],
                "opening_approach": "Match their positive energy",
                "therapeutic_tools": ["positive reinforcement", "savoring"],
                "biomarker_response": "",
                "trigger_handling": "",
                "follow_up_intention": "explore what's bringing them joy"
            }
        
        if is_masking: 
            return {
                "primary_strategy": "GENTLE_EXPLORATION",
                "tone": "soft and non-judgmental",
                "focus_areas": ["create safety for honesty"],
                "avoid": ["confrontation", "assuming"],
                "opening_approach": "Acknowledge what they said, gently note observation",
                "therapeutic_tools":  ["open questions", "validation"],
                "biomarker_response": "Gently mention sensing something beneath the surface",
                "trigger_handling":  "Be extra careful around sensitive topics",
                "follow_up_intention": "understand what's beneath the surface"
            }
        
        if trigger_entities:
            return {
                "primary_strategy": "ACTIVE_SUPPORT",
                "tone": "gentle and validating",
                "focus_areas": ["acknowledge sensitivity", "provide safety"],
                "avoid": ["pushing too hard", "dismissing"],
                "opening_approach": "Validate the difficulty of the topic",
                "therapeutic_tools":  ["validation", "grounding if needed"],
                "biomarker_response": "",
                "trigger_handling": f"Handle {', '.join(trigger_entities)} with extra care",
                "follow_up_intention": "support without retraumatizing"
            }
        
        if emotion in ["sadness", "anxiety", "fear"]:
            return {
                "primary_strategy": "ACTIVE_SUPPORT",
                "tone": "compassionate and steady",
                "focus_areas":  ["validation", "presence"],
                "avoid": ["toxic positivity", "rushing to fix"],
                "opening_approach":  "Acknowledge their feelings first",
                "therapeutic_tools":  ["empathy", "normalization", "coping strategies"],
                "biomarker_response": "Reference what you sense if appropriate",
                "trigger_handling": "",
                "follow_up_intention": "understand more about their experience"
            }
        
        return {
            "primary_strategy": "CURIOUS_ENGAGEMENT",
            "tone": "friendly and interested",
            "focus_areas":  ["building connection"],
            "avoid": ["being robotic", "interrogating"],
            "opening_approach": "Show genuine interest",
            "therapeutic_tools": ["active listening", "reflection"],
            "biomarker_response": "",
            "trigger_handling": "",
            "follow_up_intention": "deepen the conversation"
        }