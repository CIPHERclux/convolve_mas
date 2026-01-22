"""
Responder (Prompt C) - ENHANCED VERSION
Generates responses that USE biomarker data and context.
"""
from typing import Dict, Any, List, Optional

from groq import Groq


RESPONDER_PROMPT = """You are Kairos, an empathetic mental health support companion who can sense emotional states through voice and behavior.

CURRENT STATE:
- Emotion: {emotion} (intensity: {intensity})
- Crisis Level: {crisis_level}

WHAT YOU SENSED:
{biomarker_insight}

{trigger_alert}

{trajectory_alert}

CONVERSATION HISTORY (last 3 turns):
{conversation_history}

PAST RELEVANT MOMENTS:
{memory_context}

USER JUST SAID:
"{user_message}"

{crisis_instructions}

RESPONSE STRATEGY: {strategy}
TONE: {tone}
FOCUS: {focus_areas}

CRITICAL RULES:
1. Keep response 2-3 sentences (4 max in crisis)
2. NEVER start with "I'm sorry to hear" or "That must be hard"
3. If biomarkers show distress but words don't, MENTION IT: "Your voice tells me there's more going on than your words suggest..."
4. Maximum ONE question per response
5. Sound like a perceptive human friend, NOT a therapist
6. USE the biomarker insights - don't ignore them

{must_include_section}

Respond naturally as Kairos:"""

class Responder:
    """
    Enhanced Responder that:
    - Uses biomarker insights in responses
    - Addresses detected triggers appropriately
    - References trajectory concerns
    - Handles contradictions thoughtfully
    """
    
    def __init__(self, llm_client: Groq, model: str):
        self.llm = llm_client
        self.model = model
    
    def generate(
        self,
        user_message: str,
        plan: Dict[str, Any],
        diagnosis: Dict[str, Any],
        memory_context: str,
        conversation_history: str,
        laughter_detected: bool = False,
        is_crisis: bool = False,
        crisis_elements: Dict[str, Any] = None,
        biomarker_summary: str = "",
        trigger_context: str = "",
        trajectory_concern: str = "",
        contradictions: List[Dict] = None
    ) -> str:
        """Generate the final response using all available context."""
        
        # Build crisis instructions
        if is_crisis:
            crisis_instructions = CRISIS_INSTRUCTIONS
            crisis_level = f"🚨 {crisis_elements.get('severity', 'HIGH')} - CRISIS ACTIVE"
        else:
            crisis_instructions = NO_CRISIS_INSTRUCTIONS
            crisis_level = "Normal"
        
        # Build must-include section
        must_include_section = ""
        if is_crisis and crisis_elements:
            must_include = crisis_elements.get('must_include', [])
            if must_include:
                must_include_section = "\nYOU MUST INCLUDE:\n" + "\n".join(f"- {item}" for item in must_include)
        
        # Build biomarker insight for response
        biomarker_insight = self._create_response_insight(biomarker_summary, laughter_detected, diagnosis)
        
        # Format contradictions
        contradictions_str = "None detected."
        if contradictions:
            contradictions_str = "User shows conflicting feelings:\n"
            for c in contradictions[: 2]: 
                contradictions_str += f"- \"{c['clause1']}\" vs \"{c['clause2']}\"\n"
        
        # Build memory instruction
        if memory_context and "No previous context" not in memory_context: 
            memory_instruction = f"\nUSE THIS TO SHOW YOU REMEMBER THEM:\n{memory_context[: 400]}"
        else:
            memory_instruction = "\nThis is early in your relationship - focus on connection."
        
        # Get intensity
        intensity = diagnosis.get('emotional_intensity', 0.5)
        intensity_desc = "very high" if intensity > 0.8 else "high" if intensity > 0.6 else "moderate" if intensity > 0.4 else "mild"
        
        prompt = RESPONDER_PROMPT. format(
            user_message=user_message[: 600],
            crisis_level=crisis_level,
            crisis_instructions=crisis_instructions,
            strategy=plan.get("primary_strategy", "ACTIVE_SUPPORT"),
            tone=plan.get("tone", "warm and present"),
            focus_areas=", ".join(plan.get("focus_areas", ["connection"])),
            biomarker_response=plan.get("biomarker_response", ""),
            trigger_handling=plan.get("trigger_handling", ""),
            must_include_section=must_include_section,
            emotion=diagnosis.get("suspected_emotion", "uncertain"),
            intensity=intensity_desc,
            biomarker_insight=biomarker_insight,
            trigger_context=trigger_context or "No trigger entities detected.",
            trajectory_concern=trajectory_concern or "No trajectory concerns.",
            contradictions=contradictions_str,
            memory_context=memory_context[: 400] if memory_context else "First interaction",
            conversation_history=conversation_history if conversation_history else "Start of conversation",
            memory_instruction=memory_instruction
        )
        
        try: 
            response = self.llm. chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=350
            )
            
            result = response.choices[0].message. content.strip()
            result = self._clean_response(result)
            
            if is_crisis and "988" not in result and "741741" not in result: 
                result += "\n\nIf you need to talk to someone right now:  988 (call or text) or text HOME to 741741."
            
            return result
            
        except Exception as e: 
            print(f"  Warning:  Response generation error: {e}")
            return self._fallback_response(diagnosis, is_crisis)
    
    def _create_response_insight(
        self,
        biomarker_summary: str,
        laughter_detected: bool,
        diagnosis: Dict[str, Any]
    ) -> str:
        """Create insight about what biomarkers revealed for response generation."""
        insights = []
        
        if not biomarker_summary:
            biomarker_summary = ""
        
        summary_lower = biomarker_summary.lower()
        
        # Check for specific biomarker indicators
        if "jitter" in summary_lower and "elevated" in summary_lower:
            insights.append("You can HEAR stress/anxiety in their voice (voice tremor detected)")
        
        if "shimmer" in summary_lower and "elevated" in summary_lower: 
            insights.append("Their voice sounds shaky/unstable (emotional turbulence)")
        
        if "pitch variation" in summary_lower and ("low" in summary_lower or "reduced" in summary_lower):
            insights.append("Their voice sounds flat/monotone (possible depression sign)")
        
        if "pause" in summary_lower and ("high" in summary_lower or "elevated" in summary_lower):
            insights.append("They're hesitating a lot, struggling to express themselves")
        
        if "speech rate" in summary_lower and "fast" in summary_lower:
            insights.append("They're speaking quickly, possibly anxious or agitated")
        
        if "speech rate" in summary_lower and "slow" in summary_lower:
            insights.append("They're speaking slowly, possibly tired or low energy")
        
        if laughter_detected: 
            insights.append("Genuine laughter/joy detected - this is real positive emotion")
        
        if "crying" in summary_lower and "detected" in summary_lower:
            insights.append("Signs of crying/distress in their voice")
        
        # Check diagnosis for masking
        if diagnosis.get("is_masking"):
            masking_type = diagnosis.get("masking_type", "unknown")
            if masking_type == "anxiety_behind_positivity":
                insights.append("⚠️ MASKING:  Voice shows anxiety but words are positive - they may be hiding distress")
            elif masking_type == "depression_behind_normalcy":
                insights.append("⚠️ MASKING: Voice sounds flat but words seem normal - possible hidden depression")
        
        if insights:
            return "WHAT YOU SENSE (use this in your response):\n" + "\n".join(f"- {i}" for i in insights)
        else:
            return "No strong vocal/physical indicators beyond their words (text-only input or normal readings)."
    
    def _clean_response(self, response: str) -> str:
        """Clean up response artifacts."""
        prefixes_to_remove = [
            "Kairos:", "Response:", "Here's my response:",
            "As Kairos,", "KAIROS:", "*", "**"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        response = response.replace("**", "")
        response = response.replace("*", "")
        
        return response. strip()
    
    def _fallback_response(self, diagnosis: Dict[str, Any], is_crisis: bool) -> str:
        """Generate fallback response if LLM fails."""
        if is_crisis:
            return """Hey.  I need you to hear this:  you matter. What you're feeling is real, and you reaching out right now took courage. 

You don't have to go through this alone. Please reach out: 
• 988 Suicide & Crisis Lifeline (call or text 988)
• Crisis Text Line (text HOME to 741741)

I'm here.  Are you somewhere safe right now?"""
        
        emotion = diagnosis.get("suspected_emotion", "neutral")
        
        responses = {
            "joy": "There's something lighter in your words right now. What's bringing that energy?",
            "contentment": "Something feels settled in what you're sharing. Tell me more.",
            "sadness": "The heaviness in what you shared...  I felt it.  I'm here.",
            "grief": "There are no words for some kinds of pain. But I'm sitting with you in it.",
            "anxiety": "That racing feeling you're describing - like your mind won't slow down. What's weighing on you most right now?",
            "fear":  "Something's got you scared. You're safe here - what's going on?",
            "anger":  "I can feel the heat in your words. What happened? ",
            "frustration": "Something's really getting to you. Let it out.",
            "loneliness": "That ache of feeling alone, even in a crowd...  it's one of the hardest feelings.  I see you.",
            "stress": "There's tension in what you're sharing. Your body's telling you something.  What feels most overwhelming?",
            "overwhelm": "It sounds like everything's hitting at once. Let's slow down.  What's the one thing weighing on you most?",
            "hopelessness": "When everything feels pointless, it's hard to see a way through. But you reached out, and that matters.  What's making things feel so heavy?",
            "neutral": "I'm here. What's on your mind today?"
        }
        
        return responses.get(emotion, responses["neutral"])