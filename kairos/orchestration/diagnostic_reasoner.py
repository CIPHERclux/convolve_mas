"""
Diagnostic Reasoner (Prompt A) - ENHANCED VERSION
Integrates biomarker data, entity triggers, and trajectory patterns. 
"""
import json
from typing import Dict, Any, List, Optional

from groq import Groq


DIAGNOSTIC_PROMPT = """SYSTEM:  You are the Clinical Reasoner for Kairos, a mental health support system. 

Your task is to analyze the user's message along with their physiological signals to understand their true emotional state. 

INPUT DATA:
- User Text: "{text}"
- Modality: {modality}
- Laughter Score: {laughter_score}
- Reliability:  {reliability_summary}

=== BIOMARKER ANALYSIS ===
{biomarker_summary}

=== ENTITY CONTEXT ===
{entity_context}

=== TRAJECTORY PATTERNS ===
{trajectory_info}

=== MEMORY CONTEXT ===
{memory_context}

ANALYSIS TASK:
1. FIRST CHECK LAUGHTER:  If Laughter Score > 0.5, the emotional state likely includes genuine joy/humor. 

2. CHECK VOICE BIOMARKERS (USE THIS DATA):
   - High jitter (voice tremor) = anxiety/stress
   - High shimmer = voice instability/emotional turbulence
   - Low pitch variation = possible depression/flatness
   - High pause frequency = hesitation/cognitive load

3. CHECK TRIGGER ENTITIES:  If any entity is marked as a TRIGGER, factor this into risk assessment.

4. CHECK TRAJECTORY:  If escalating pattern detected, this is a warning sign.

5. MASKING DETECTION: Does the biological state match the text sentiment? 
   - High jitter/shimmer with positive text = possible anxiety masking
   - Low F0 variance with enthusiastic text = possible depression masking

6. EMOTIONAL INFERENCE: What is the most likely underlying emotion? 

7. RISK ASSESSMENT:  Evaluate risk level (LOW, MODERATE, HIGH, CRITICAL)

OUTPUT FORMAT (JSON only, no other text):
{{
    "is_masking": true or false,
    "masking_type": "anxiety_behind_positivity" or "depression_behind_normalcy" or "anger_suppression" or "none",
    "suspected_emotion": "primary emotion",
    "secondary_emotions": ["list", "of", "other", "emotions"],
    "emotional_intensity": 0.0 to 1.0,
    "risk_level": "LOW" or "MODERATE" or "HIGH" or "CRITICAL",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation INCLUDING what biomarkers and triggers revealed",
    "biomarker_insights": "specific insights from the biomarker data",
    "trigger_impact": "how trigger entities affect assessment",
    "trajectory_concern": "trajectory pattern concern if any",
    "recommended_focus": "what aspect to address"
}}"""


class DiagnosticReasoner:
    """
    Enhanced Diagnostic Reasoner with: 
    - Full biomarker integration
    - Entity trigger awareness
    - Trajectory pattern analysis
    """
    
    def __init__(self, llm_client: Groq, model: str):
        self.llm = llm_client
        self. model = model
    
    def analyze(
        self,
        text: str,
        bio_vector: List[float],
        laughter_score: float,
        reliability_mask: List[float],
        memory_context: str,
        modality: str = "text",
        biomarker_summary: str = "",
        entity_context: str = "",
        trajectory_info: str = ""
    ) -> Dict[str, Any]:
        """Perform clinical analysis of user state with all available context."""
        
        reliability_summary = self._summarize_reliability(reliability_mask, modality)
        laughter_score_str = f"{laughter_score:. 2f}"
        
        if not biomarker_summary:
            biomarker_summary = self._create_basic_summary(bio_vector, reliability_mask)
        
        prompt = DIAGNOSTIC_PROMPT.format(
            text=text[: 500],
            modality=modality,
            laughter_score=laughter_score_str,
            reliability_summary=reliability_summary,
            biomarker_summary=biomarker_summary,
            entity_context=entity_context or "No entity context available.",
            trajectory_info=trajectory_info or "No trajectory patterns detected.",
            memory_context=memory_context[: 1000] if memory_context else "No previous context."
        )
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical analysis system.  Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            diagnosis = json.loads(result_text)
            
            diagnosis. setdefault("is_masking", False)
            diagnosis.setdefault("suspected_emotion", "neutral")
            diagnosis.setdefault("risk_level", "LOW")
            diagnosis.setdefault("confidence", 0.5)
            diagnosis.setdefault("biomarker_insights", "")
            diagnosis.setdefault("trigger_impact", "none")
            diagnosis.setdefault("trajectory_concern", "none")
            
            return diagnosis
            
        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse diagnosis JSON: {e}")
            return self._default_diagnosis(text, laughter_score, bio_vector)
        except Exception as e:
            print(f"  Warning:  Diagnostic reasoning error: {e}")
            return self._default_diagnosis(text, laughter_score, bio_vector)
    
    def _create_basic_summary(self, bio_vector: List[float], reliability_mask: List[float]) -> str:
        """Create basic biomarker summary if not provided."""
        parts = []
        
        labels = [
            "jitter (anxiety)", "shimmer (instability)", "pitch variation",
            "loudness", "vocal energy", "clarity", "speech rate", "pauses"
        ]
        
        for i, label in enumerate(labels):
            if i < len(bio_vector) and i < len(reliability_mask):
                if reliability_mask[i] > 0.5 and abs(bio_vector[i]) > 0.2:
                    direction = "elevated" if bio_vector[i] > 0 else "reduced"
                    parts.append(f"- {label}: {direction} ({bio_vector[i]: +.2f})")
        
        return "\n".join(parts) if parts else "No significant biomarker signals detected."
    
    def _summarize_reliability(self, reliability_mask: List[float], modality: str) -> str:
        """Summarize which signals are trustworthy."""
        acoustic_trust = sum(reliability_mask[0:8]) / 8 if len(reliability_mask) >= 8 else 0
        visual_trust = sum(reliability_mask[8:16]) / 8 if len(reliability_mask) >= 16 else 0
        
        parts = []
        parts.append(f"Acoustic:  {'TRUSTED' if acoustic_trust > 0.5 else 'unavailable'}")
        parts.append(f"Visual: {'TRUSTED' if visual_trust > 0.5 else 'unavailable'}")
        parts.append(f"Linguistic:  TRUSTED")
        parts.append(f"Modality: {modality}")
        
        return " | ".join(parts)
    
    def _default_diagnosis(self, text: str, laughter_score: float, bio_vector:  List[float]) -> Dict[str, Any]:
        """Return default diagnosis when LLM fails."""
        text_lower = text.lower()
        
        jitter = bio_vector[0] if len(bio_vector) > 0 else 0
        sentiment = bio_vector[22] if len(bio_vector) > 22 else 0
        
        if any(word in text_lower for word in ["happy", "great", "wonderful", "excited", "joy"]):
            emotion = "joy" if laughter_score > 0.3 else "contentment"
            risk = "LOW"
        elif any(word in text_lower for word in ["sad", "depressed", "hopeless", "worthless"]):
            emotion = "sadness"
            risk = "MODERATE"
        elif any(word in text_lower for word in ["anxious", "worried", "scared", "panic"]) or jitter > 0.4:
            emotion = "anxiety"
            risk = "MODERATE"
        elif any(word in text_lower for word in ["angry", "furious", "hate", "frustrated"]):
            emotion = "anger"
            risk = "MODERATE"
        elif sentiment < -0.3:
            emotion = "sadness"
            risk = "MODERATE"
        else:
            emotion = "neutral"
            risk = "LOW"
        
        is_masking = False
        masking_type = "none"
        if jitter > 0.4 and sentiment > 0.2:
            is_masking = True
            masking_type = "anxiety_behind_positivity"
        
        return {
            "is_masking": is_masking,
            "masking_type":  masking_type,
            "suspected_emotion": emotion,
            "secondary_emotions": [],
            "emotional_intensity": min(1.0, 0.5 + abs(sentiment) * 0.3 + jitter * 0.2),
            "risk_level": risk,
            "confidence": 0.4,
            "reasoning": f"Fallback analysis.  Jitter={jitter:.2f}, Sentiment={sentiment:.2f}",
            "biomarker_insights": f"Jitter:  {jitter:.2f}, Sentiment: {sentiment:.2f}",
            "trigger_impact":  "unknown",
            "trajectory_concern":  "unknown",
            "recommended_focus": "general support"
        }