"""
User Profile (World State)
Stores deterministic facts about the user that should be retrieved exactly,
not through vector similarity search. 

This is CRITICAL for fixing the "evasive therapist" problem - when users ask
"what's my name?" or "what do you remember? ", we need exact factual retrieval,
not semantic similarity. 
"""
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class UserProfile:
    """
    Stores and retrieves user facts deterministically.
    
    Facts include:
    - Name, pronouns, age
    - Location, occupation
    - Relationships (mom's name, partner's name, etc.)
    - Diagnoses, medications
    - Important dates
    - Preferences and communication style
    - Goals and aspirations
    - Fears and triggers (linked to EntityGraph)
    
    All facts are stored as exact key-value pairs for O(1) retrieval.
    """
    
    def __init__(self, user_id: str, storage_path: str = "./profiles"):
        self.user_id = user_id
        self. storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.profile_file = self.storage_path / f"{user_id}_profile.json"
        
        # Core facts structure
        self.facts: Dict[str, Any] = {
            # Identity
            'name': None,
            'nickname': None,
            'pronouns': None,
            'age':  None,
            'birthday': None,
            
            # Location & Work
            'location': None,
            'hometown': None,
            'occupation': None,
            'workplace': None,
            'school': None,
            
            # Relationships (dict of relationship_type -> name)
            'relationships': {},
            
            # Health
            'diagnoses': [],
            'medications':  [],
            'therapist_name': None,
            'therapy_type': None,
            'therapy_frequency': None,
            
            # Important dates
            'important_dates': {},
            
            # Preferences
            'preferences': {
                'communication_style': None,  # casual, formal, etc.
                'prefers_questions': None,  # True/False
                'prefers_advice': None,
                'prefers_validation': None,
            },
            
            # Goals and concerns
            'goals': [],
            'main_concerns': [],
            'fears': [],
            
            # Hobbies and interests
            'hobbies': [],
            'interests': [],
            
            # Custom/other facts
            'custom_facts': {},
            
            # Metadata
            'first_interaction': None,
            'last_updated': None,
            'total_interactions': 0,
            'facts_extracted_count': 0,
        }
        
        # Extraction patterns - carefully crafted to avoid false positives
        self._init_extraction_patterns()
        
        # Facts extracted this session (for tracking)
        self.session_extractions: List[Dict[str, Any]] = []
        
        # Load existing profile
        self._load_profile()
    
    def _init_extraction_patterns(self):
        """Initialize comprehensive fact extraction patterns."""
        
        # Name patterns - be careful to avoid false positives
        self. name_patterns = [
            # Direct statements
            (r"(? : my name is|i'm|i am|call me|they call me|everyone calls me)\s+([A-Z][a-z]+)", 1.0),
            (r"(?:name's|names)\s+([A-Z][a-z]+)", 0.9),
            (r"^([A-Z][a-z]+)\s+here\b", 0.7),
            (r"(?: i go by)\s+([A-Z][a-z]+)", 0.95),
            # Response to "what's your name"
            (r"^([A-Z][a-z]+)\.? $", 0.5),  # Single capitalized word
        ]
        
        # Age patterns
        self.age_patterns = [
            (r"(? :i'm|i am|im)\s+(\d{1,2})\s*(?:years?\s*old|yo|y/o)", 1.0),
            (r"(\d{1,2})\s*(?:years?\s*old|yo|y/o)", 0.9),
            (r"(?:age|aged)\s*(? :is|: )?\s*(\d{1,2})", 0.95),
            (r"(?: turned|turning)\s+(\d{1,2})", 0.8),
            (r"(?: i'm|i am)\s+(\d{1,2})\b(? !\s*(? :minutes|hours|days|times))", 0.7),
        ]
        
        # Pronoun patterns
        self.pronoun_patterns = [
            (r"(?:my pronouns are|i use|i go by)\s+(she/her|he/him|they/them|she/they|he/they|xe/xem|ze/zir)", 1.0),
            (r"(she/her|he/him|they/them|she/they|he/they)\s+(? :pronouns|please)", 0.95),
            (r"(?: use|prefer)\s+(she/her|he/him|they/them)", 0.9),
        ]
        
        # Location patterns
        self.location_patterns = [
            (r"(?: i live in|i'm from|i am from|living in|based in|moved to)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|! |\?|$)", 1.0),
            (r"(?:from|in)\s+([A-Z][a-z]+(? : ,\s*[A-Z]{2})?)\b", 0.7),
            (r"(?: hometown is|grew up in)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$)", 0.95),
        ]
        
        # Occupation patterns
        self.occupation_patterns = [
            (r"(? :i work as|i'm a|i am a|my job is|i work in|working as)\s+(? :a\s+)?([a-zA-Z\s]+?)(?:\.|,|!|\? |$)", 1.0),
            (r"(?:work|job|career)\s+(?:as|is|in)\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\.|,|$)", 0.9),
            (r"(?:i'm|i am)\s+(?:a\s+)?([a-zA-Z]+(? :\s+[a-zA-Z]+)?)\s+(?:at|for|in)", 0.7),
        ]
        
        # School patterns
        self.school_patterns = [
            (r"(? :i go to|i attend|studying at|student at)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", 1.0),
            (r"(?:in|at)\s+([A-Z][a-zA-Z\s]+? )\s+(?:university|college|school|high school)", 0.9),
        ]
        
        # Relationship patterns - (pattern, relationship_type, name_group)
        self.relationship_patterns = [
            # Parents
            (r"(?:my\s+)?(mom|mother|mum|mommy|mama)(? :'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'mother', 2),
            (r"(?: my\s+)?(dad|father|daddy|papa)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'father', 2),
            
            # Siblings
            (r"(?:my\s+)?(brother|bro)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'brother', 2),
            (r"(?:my\s+)?(sister|sis)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'sister', 2),
            
            # Partners
            (r"(?:my\s+)?(partner|spouse|husband|wife)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'partner', 2),
            (r"(?:my\s+)?(boyfriend|bf)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'boyfriend', 2),
            (r"(?:my\s+)?(girlfriend|gf)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'girlfriend', 2),
            (r"(?:my\s+)?(fiance|fiancee|fiancé|fiancée)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'fiance', 2),
            
            # Friends
            (r"(?:my\s+)?(best friend|bestie|bff)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'best_friend', 2),
            (r"(?:my\s+)?(friend)(?:'s name is|'s name's| is named| is called| named| called)\s+([A-Z][a-z]+)", 'friend', 2),
            
            # Professional
            (r"(?:my\s+)?(therapist|counselor|psychiatrist|psychologist)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)(?:Dr\. ?\s+)?([A-Z][a-z]+)", 'therapist', 2),
            (r"(?:my\s+)?(boss|manager|supervisor)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'boss', 2),
            
            # Children
            (r"(?:my\s+)?(son|daughter|child|kid)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'child', 2),
            
            # Pets
            (r"(?:my\s+)?(dog|cat|pet)(?:'s name is|'s name's| is named| is called| named| called|,?\s+)([A-Z][a-z]+)", 'pet', 2),
            
            # Generic "X's name is Y" pattern
            (r"my\s+([a-z]+(? :'s)?)\s+(? :name is|called)\s+([A-Z][a-z]+)", None, None),  # Dynamic
        ]
        
        # Diagnosis patterns
        self.diagnosis_patterns = [
            (r"(? :i have|diagnosed with|i've been diagnosed with|i was diagnosed with|suffering from|i deal with|i struggle with|living with)\s+(depression|anxiety|bipolar|bipolar disorder|bpd|borderline|ptsd|ocd|adhd|add|schizophrenia|eating disorder|anorexia|bulimia|autism|asd|social anxiety|gad|generalized anxiety|panic disorder|insomnia)", 1.0),
            (r"(?:my|the)\s+(depression|anxiety|bipolar|ptsd|ocd|adhd|bpd)", 0.8),
        ]
        
        # Medication patterns
        self.medication_patterns = [
            (r"(? :i take|i'm on|taking|prescribed)\s+([A-Z][a-z]+(? : in|ol|am|ex|ir)?)\s*(?:\d+\s*mg)?", 0.8),
            (r"(?:medication|meds? )\s+(?:is|are|called)?\s*([A-Z][a-z]+)", 0.7),
        ]
        
        # Hobby/interest patterns
        self.hobby_patterns = [
            (r"(? :i love|i enjoy|i like|into|passionate about|hobby is|hobbies are|hobbies include)\s+([a-z]+(? : ing)?(? :\s+and\s+[a-z]+(? :ing)?)*)", 0.9),
        ]
        
        # Goal patterns
        self.goal_patterns = [
            (r"(? :i want to|i'm trying to|my goal is to|hoping to|working on)\s+(. +?)(?:\.|,|!|$)", 0.8),
        ]
        
        # Common words to exclude from name extraction
        self.name_exclusions = {
            'i', 'me', 'my', 'the', 'a', 'an', 'and', 'or', 'but', 'so', 'just',
            'really', 'actually', 'basically', 'literally', 'honestly', 'well',
            'yeah', 'yes', 'no', 'not', 'very', 'much', 'here', 'there', 'now',
            'then', 'today', 'yesterday', 'tomorrow', 'always', 'never', 'sometimes',
            'maybe', 'probably', 'definitely', 'certainly', 'please', 'thanks',
            'thank', 'sorry', 'okay', 'ok', 'sure', 'right', 'wrong', 'good', 'bad',
            'great', 'fine', 'nice', 'cool', 'awesome', 'terrible', 'horrible',
            'happy', 'sad', 'angry', 'scared', 'worried', 'anxious', 'depressed',
            'tired', 'exhausted', 'confused', 'lost', 'alone', 'lonely',
            'hey', 'hi', 'hello', 'bye', 'goodbye', 'morning', 'evening', 'night',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
        }
    
    def _load_profile(self):
        """Load profile from disk."""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                
                # Deep merge with defaults to handle new fields
                self._deep_merge(self.facts, data)
                
                if self.facts. get('name'):
                    print(f"  [UserProfile] Loaded profile for {self.facts['name']}")
                else:
                    print(f"  [UserProfile] Loaded profile (no name yet)")
                    
            except Exception as e: 
                print(f"  [UserProfile] Warning: Could not load profile: {e}")
        else:
            # New profile
            self.facts['first_interaction'] = datetime.now().isoformat()
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge update into base."""
        for key, value in update.items():
            if key in base: 
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._deep_merge(base[key], value)
                else: 
                    base[key] = value
            else:
                base[key] = value
    
    def _save_profile(self):
        """Save profile to disk."""
        try:
            self.facts['last_updated'] = datetime. now().isoformat()
            with open(self.profile_file, 'w') as f:
                json.dump(self. facts, f, indent=2, default=str)
        except Exception as e:
            print(f"  [UserProfile] Warning: Could not save profile: {e}")
    
    def extract_facts_from_text(self, text:  str, turn_number: int = 0, 
                                 system_response: str = None) -> List[Dict[str, Any]]:
        """
        Extract facts from user text and store them. 
        
        Returns list of facts that were extracted with metadata.
        """
        extracted = []
        
        if not text or len(text. strip()) < 2:
            return extracted
        
        # Update interaction count
        self.facts['total_interactions'] += 1
        
        # Extract name
        name_result = self._extract_name(text)
        if name_result:
            extracted.append(name_result)
        
        # Extract age
        age_result = self._extract_age(text)
        if age_result: 
            extracted.append(age_result)
        
        # Extract pronouns
        pronoun_result = self._extract_pronouns(text)
        if pronoun_result:
            extracted.append(pronoun_result)
        
        # Extract location
        location_result = self._extract_location(text)
        if location_result:
            extracted. append(location_result)
        
        # Extract occupation
        occupation_result = self._extract_occupation(text)
        if occupation_result:
            extracted.append(occupation_result)
        
        # Extract school
        school_result = self._extract_school(text)
        if school_result: 
            extracted.append(school_result)
        
        # Extract relationships
        relationship_results = self._extract_relationships(text)
        extracted.extend(relationship_results)
        
        # Extract diagnoses
        diagnosis_results = self._extract_diagnoses(text)
        extracted.extend(diagnosis_results)
        
        # Extract medications
        medication_results = self._extract_medications(text)
        extracted.extend(medication_results)
        
        # Save if we extracted anything
        if extracted:
            self. facts['facts_extracted_count'] += len(extracted)
            self._save_profile()
            
            # Track session extractions
            for fact in extracted:
                self.session_extractions.append({
                    **fact,
                    'turn_number': turn_number,
                    'timestamp': datetime.now().isoformat()
                })
        
        return extracted
    
    def _extract_name(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract name from text with aggressive detection."""
        if self.facts['name']:
            return None  # Already have name
        
        text_stripped = text.strip()
        
        # PRIORITY 1: Direct single-word name (most common in greetings)
        words = text_stripped.split()
        if len(words) <= 3:  # "hi i am NAME" or just "NAME"
            for word in words:
                if word[0].isupper() and len(word) >= 2 and self._is_valid_name(word):
                    self.facts['name'] = word
                    print(f"  [UserProfile] ✅ EXTRACTED NAME (direct): {word}")
                    return {
                        'type': 'name',
                        'value': word,
                        'confidence': 0.95,
                        'pattern': 'direct_capitalized_word'
                    }
        
        # PRIORITY 2: Explicit patterns
        explicit_patterns = [
            (r"(?:i am|i'm|im|my name is|call me|this is)\s+([A-Z][a-z]+)", 1.0),
            (r"^([A-Z][a-z]+)(?:\s+here)?[!.]?\s*$", 0.9),
            (r"([A-Z][a-z]+)\s+(?:here|speaking)", 0.85)
        ]
        
        for pattern, confidence in explicit_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if self._is_valid_name(name):
                        self.facts['name'] = name
                        print(f"  [UserProfile] ✅ EXTRACTED NAME (pattern): {name}")
                        return {
                            'type': 'name',
                            'value': name,
                            'confidence': confidence,
                            'pattern': pattern
                        }
            except Exception:
                continue
        
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate extracted name."""
        if not name or len(name) < 2:
            return False
        if name.lower() in self.name_exclusions:
            return False
        if not name[0].isupper():
            return False
        if len(name) > 20:
            return False
        if any(c.isdigit() for c in name):
            return False
        return True
    
    def _extract_age(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract age from text."""
        if self.facts['age']:
            return None
        
        for pattern, confidence in self.age_patterns:
            try:
                match = re. search(pattern, text, re. IGNORECASE)
                if match:
                    age = int(match.group(1))
                    if 10 <= age <= 100:  # Reasonable age range
                        self.facts['age'] = age
                        return {
                            'type': 'age',
                            'value': age,
                            'confidence': confidence,
                            'pattern': pattern
                        }
            except Exception:
                continue
        
        return None
    
    def _extract_pronouns(self, text: str) -> Optional[Dict[str, Any]]: 
        """Extract pronouns from text."""
        if self.facts['pronouns']:
            return None
        
        for pattern, confidence in self.pronoun_patterns:
            try:
                match = re. search(pattern, text, re. IGNORECASE)
                if match:
                    pronouns = match.group(1).lower()
                    self.facts['pronouns'] = pronouns
                    return {
                        'type': 'pronouns',
                        'value': pronouns,
                        'confidence': confidence,
                        'pattern': pattern
                    }
            except Exception: 
                continue
        
        return None
    
    def _extract_location(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract location from text."""
        if self.facts['location']:
            return None
        
        for pattern, confidence in self.location_patterns:
            try:
                match = re. search(pattern, text)
                if match:
                    location = match.group(1).strip()
                    if len(location) > 2 and len(location) < 50:
                        self.facts['location'] = location
                        return {
                            'type': 'location',
                            'value': location,
                            'confidence': confidence,
                            'pattern': pattern
                        }
            except Exception:
                continue
        
        return None
    
    def _extract_occupation(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract occupation from text."""
        if self.facts['occupation']: 
            return None
        
        for pattern, confidence in self.occupation_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    occupation = match.group(1).strip().lower()
                    # Validate
                    if len(occupation) > 2 and len(occupation) < 40:
                        # Exclude false positives
                        exclude_words = {'good', 'bad', 'fine', 'okay', 'well', 'better', 'worse'}
                        if occupation.split()[0] not in exclude_words:
                            self.facts['occupation'] = occupation
                            return {
                                'type': 'occupation',
                                'value': occupation,
                                'confidence': confidence,
                                'pattern': pattern
                            }
            except Exception: 
                continue
        
        return None
    
    def _extract_school(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract school from text."""
        if self.facts['school']:
            return None
        
        for pattern, confidence in self.school_patterns:
            try:
                match = re. search(pattern, text)
                if match:
                    school = match.group(1).strip()
                    if len(school) > 2:
                        self.facts['school'] = school
                        return {
                            'type': 'school',
                            'value':  school,
                            'confidence':  confidence,
                            'pattern':  pattern
                        }
            except Exception:
                continue
        
        return None
    
    def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationship names from text."""
        extracted = []
        
        for pattern, rel_type, name_group in self.relationship_patterns:
            if rel_type and rel_type in self.facts. get('relationships', {}):
                continue  # Already have this relationship
            
            try:
                match = re.search(pattern, text, re. IGNORECASE)
                if match:
                    if name_group:
                        name = match.group(name_group).strip()
                    else:
                        # Dynamic pattern handling
                        continue
                    
                    if self._is_valid_name(name):
                        if rel_type: 
                            self.facts['relationships'][rel_type] = name
                            extracted.append({
                                'type': 'relationship',
                                'relationship_type': rel_type,
                                'value': name,
                                'confidence': 0.9,
                                'pattern': pattern
                            })
            except Exception:
                continue
        
        return extracted
    
    def _extract_diagnoses(self, text: str) -> List[Dict[str, Any]]:
        """Extract diagnoses from text."""
        extracted = []
        
        for pattern, confidence in self.diagnosis_patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    diagnosis = match.lower().strip()
                    if diagnosis and diagnosis not in self.facts['diagnoses']:
                        self. facts['diagnoses'].append(diagnosis)
                        extracted.append({
                            'type': 'diagnosis',
                            'value': diagnosis,
                            'confidence': confidence,
                            'pattern': pattern
                        })
            except Exception:
                continue
        
        return extracted
    
    def _extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """Extract medications from text."""
        extracted = []
        
        for pattern, confidence in self.medication_patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        med = match[0]
                    else:
                        med = match
                    med = med.strip()
                    if med and len(med) > 2 and med. lower() not in [m.lower() for m in self.facts['medications']]:
                        self. facts['medications'].append(med)
                        extracted.append({
                            'type': 'medication',
                            'value': med,
                            'confidence': confidence,
                            'pattern': pattern
                        })
            except Exception:
                continue
        
        return extracted
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def set_fact(self, key: str, value: Any, sub_key: str = None):
        """Manually set a fact."""
        if sub_key:
            if key in self.facts and isinstance(self.facts[key], dict):
                self.facts[key][sub_key] = value
            else:
                self. facts['custom_facts'][f"{key}. {sub_key}"] = value
        elif key in self.facts:
            self.facts[key] = value
        else:
            self. facts['custom_facts'][key] = value
        
        self._save_profile()
    
    def get_fact(self, key: str, sub_key:  str = None) -> Any:
        """Get a specific fact."""
        if sub_key:
            if key in self.facts and isinstance(self. facts[key], dict):
                return self.facts[key].get(sub_key)
            return self.facts.get('custom_facts', {}).get(f"{key}.{sub_key}")
        
        if key in self.facts:
            return self.facts[key]
        return self.facts.get('custom_facts', {}).get(key)
    
    def get_name(self) -> Optional[str]:
        """Get user's name."""
        return self. facts.get('name')
    
    def get_all_facts(self) -> Dict[str, Any]:
        """Get all non-null facts in a flat structure."""
        result = {}
        
        for key, value in self.facts. items():
            if key in ['custom_facts', 'preferences', 'important_dates']: 
                continue  # Handle separately
            if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                result[key] = value
        
        # Add relationships flattened
        if self.facts.get('relationships'):
            result['relationships'] = self.facts['relationships']
        
        return result
    
    def get_facts_for_llm(self) -> str:
        """Get formatted facts string for LLM context."""
        facts = self.get_all_facts()
        
        if not facts or all(not v for v in facts.values() if not isinstance(v, dict)):
            return ""
        
        lines = ["=== USER PROFILE (Known Facts) ==="]
        
        # Core identity
        if facts. get('name'):
            lines. append(f"• Name: {facts['name']}")
        if facts.get('nickname'):
            lines.append(f"• Nickname: {facts['nickname']}")
        if facts.get('pronouns'):
            lines.append(f"• Pronouns: {facts['pronouns']}")
        if facts.get('age'):
            lines.append(f"• Age: {facts['age']}")
        
        # Location & work
        if facts.get('location'):
            lines.append(f"• Location: {facts['location']}")
        if facts.get('occupation'):
            lines.append(f"• Occupation: {facts['occupation']}")
        if facts.get('school'):
            lines. append(f"• School: {facts['school']}")
        
        # Relationships
        if facts.get('relationships'):
            lines.append("• Relationships:")
            for rel_type, name in facts['relationships'].items():
                lines.append(f"    - {rel_type}: {name}")
        
        # Health
        if facts.get('diagnoses'):
            lines.append(f"• Diagnoses: {', '.join(facts['diagnoses'])}")
        if facts.get('medications'):
            lines.append(f"• Medications: {', '.join(facts['medications'])}")
        if facts.get('therapist_name'):
            lines.append(f"• Therapist: {facts['therapist_name']}")
        
        # Goals
        if facts.get('goals'):
            lines.append(f"• Goals: {', '.join(facts['goals'][:3])}")
        if facts.get('main_concerns'):
            lines.append(f"• Main concerns: {', '.join(facts['main_concerns'][:3])}")
        
        # Metadata
        if facts.get('total_interactions'):
            lines.append(f"• Total interactions: {facts['total_interactions']}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profile statistics."""
        facts = self.get_all_facts()
        
        return {
            'has_name': self.facts.get('name') is not None,
            'name':  self.facts.get('name'),
            'has_age': self.facts.get('age') is not None,
            'has_pronouns': self.facts. get('pronouns') is not None,
            'has_location':  self.facts.get('location') is not None,
            'has_occupation': self.facts.get('occupation') is not None,
            'relationship_count': len(self.facts.get('relationships', {})),
            'diagnosis_count': len(self.facts.get('diagnoses', [])),
            'medication_count': len(self.facts.get('medications', [])),
            'total_interactions': self.facts.get('total_interactions', 0),
            'facts_extracted_count': self.facts.get('facts_extracted_count', 0),
            'session_extractions': len(self.session_extractions),
            'total_facts': len([v for v in facts.values() if v and (not isinstance(v, (list, dict)) or len(v) > 0)]),
            'facts':  facts
        }
    
    def force_save(self):
        """Force save to disk."""
        self._save_profile()
    
    def clear_profile(self):
        """Clear all profile data (use with caution)."""
        self.facts = {
            'name': None,
            'nickname':  None,
            'pronouns':  None,
            'age': None,
            'birthday': None,
            'location':  None,
            'hometown': None,
            'occupation':  None,
            'workplace': None,
            'school':  None,
            'relationships': {},
            'diagnoses': [],
            'medications': [],
            'therapist_name': None,
            'therapy_type': None,
            'therapy_frequency': None,
            'important_dates': {},
            'preferences': {},
            'goals': [],
            'main_concerns': [],
            'fears': [],
            'hobbies': [],
            'interests': [],
            'custom_facts': {},
            'first_interaction': datetime.now().isoformat(),
            'last_updated': None,
            'total_interactions':  0,
            'facts_extracted_count': 0,
        }
        self._save_profile()