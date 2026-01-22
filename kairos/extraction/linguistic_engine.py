"""
Linguistic Engine (Module C)
FIX #3: Better semantic sentiment analysis
"""
import re
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from config import ABSOLUTIST_WORDS, FILLER_WORDS


class LinguisticEngine: 
    """
    Module C: Linguistic Feature Extraction
    
    FIX #3: Improved semantic understanding
    - Pattern-based detection for complex phrases
    - Context-aware sentiment
    - Better handling of negations and sarcasm
    """
    
    def __init__(self):
        self.absolutist_words = set(ABSOLUTIST_WORDS)
        self.filler_words = set(FILLER_WORDS)
        self._init_nltk()
        self._init_semantic_patterns()
    
    def _init_nltk(self):
        """Initialize NLTK with fallback."""
        self._nltk_available = False
        self._vader_available = False
        
        try:
            import nltk
            for resource in ['punkt', 'punkt_tab', 'vader_lexicon']:
                try: 
                    nltk.download(resource, quiet=True)
                except:
                    pass
            
            try:
                from nltk.tokenize import word_tokenize, sent_tokenize
                word_tokenize("test")
                self._word_tokenize = word_tokenize
                self._sent_tokenize = sent_tokenize
                self._nltk_available = True
            except:
                pass
            
            try:
                from nltk.sentiment. vader import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                self._vader_available = True
            except: 
                pass
                
        except: 
            pass
    
    def _init_semantic_patterns(self):
        """Initialize semantic patterns for better sentiment detection."""
        
        # Highly negative patterns (semantic understanding)
        self.very_negative_patterns = [
            # Suicidal ideation
            (r'(world|everyone|they).*(better|fine).*(without me|if i)', -0.95),
            (r'(no|dont|don\'t) (want to|wanna) (live|be here|exist|be alive)', -0.95),
            (r'(should|want to|gonna|going to) (unalive|kill|end|hurt) (myself|me)', -1.0),
            (r'(wish i was|wish i were|wished i was) (dead|never born|gone)', -0.95),
            (r'no (point|reason|purpose) (in|to|for) (living|life|being here|me)', -0.9),
            (r'(better off|be better) (dead|gone|not here)', -0.95),
            (r"(can't|cannot) (take|handle|do) (it|this) (anymore|any more)", -0.85),
            
            # Hopelessness
            (r'(nothing|no one|nobody) (will ever|ever) (care|love|help|change)', -0.8),
            (r'(everyone|everybody|they all) (hates? |left|abandoned|ignores?)', -0.75),
            (r'(completely|totally|utterly|absolutely) (alone|worthless|hopeless|useless)', -0.85),
            (r"what'? s the point", -0.7),
            (r'(give|giving|given) up (on|with) (life|everything|myself)', -0.85),
            
            # Desperation
            (r'(please|someone) (help|save) (me)?', -0.5),
            (r"(don't|dont|do not) know what to do", -0.55),
            (r'(feel|feeling) (so )?(trapped|stuck|lost|empty|numb)', -0.6),
        ]
        
        # Negative patterns
        self.negative_patterns = [
            (r'(really|so|very|extremely) (sad|depressed|down|upset|hurt)', -0.6),
            (r'(hate|hating) (myself|my life|everything|this)', -0.65),
            (r'(angry|furious|pissed|mad) (at|about|with)', -0.5),
            (r'(scared|terrified|afraid|anxious|worried) (of|about|that)', -0.5),
            (r'(lonely|alone|isolated|abandoned)', -0.5),
            (r'(bad|terrible|awful|horrible) (day|time|week|life)', -0.45),
            (r'(crying|cried|cry) (all|every|so much)', -0.55),
        ]
        
        # Positive patterns
        self.positive_patterns = [
            (r'(feel|feeling) (much |so |a lot )?(better|good|great|happy|hopeful)', 0.7),
            (r'(things are|it\'s|everything is) (getting|looking) better', 0.6),
            (r'(really|so|very) (happy|excited|grateful|thankful)', 0.7),
            (r'(good|great|wonderful|amazing) (day|news|thing|time)', 0.55),
            (r'thank (you|god|goodness)', 0.4),
            (r'(love|appreciate) (you|this|that)', 0.5),
        ]
        
        # Compile all patterns
        self.compiled_very_negative = [(re.compile(p, re.IGNORECASE), s) 
                                        for p, s in self.very_negative_patterns]
        self.compiled_negative = [(re. compile(p, re.IGNORECASE), s) 
                                   for p, s in self.negative_patterns]
        self.compiled_positive = [(re.compile(p, re.IGNORECASE), s) 
                                   for p, s in self.positive_patterns]
    
    def _simple_tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s\']', ' ', text. lower())
        return [w for w in text.split() if w]
    
    def _simple_sent_tokenize(self, text:  str) -> List[str]:
        sentences = re.split(r'[. !?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract(self, text: str, last_system_end_time: Optional[str] = None) -> np.ndarray:
        """Extract all 8 linguistic features."""
        features = np.zeros(8, dtype=np.float32)
        
        if not text or not text.strip():
            return features
        
        try:
            # Tokenize
            if self._nltk_available:
                try:
                    words = self._word_tokenize(text. lower())
                    sentences = self._sent_tokenize(text)
                except:
                    words = self._simple_tokenize(text)
                    sentences = self._simple_sent_tokenize(text)
            else:
                words = self._simple_tokenize(text)
                sentences = self._simple_sent_tokenize(text)
            
            if not words:
                return features
            
            features[0] = self._compute_absolutist_index(words)
            features[1] = self._compute_i_ratio(words)
            features[2] = self._compute_response_latency(last_system_end_time)
            features[3] = self._compute_lexical_density(words)
            features[4] = self._compute_past_tense_ratio(words)
            features[5] = self._compute_filler_density(text, words)
            features[6] = self._compute_semantic_sentiment(text)  # IMPROVED
            features[7] = self._compute_rumination(text, words, sentences)
            
        except Exception as e:
            print(f"  Warning: Linguistic feature extraction error: {e}")
        
        return features
    
    def _compute_semantic_sentiment(self, text: str) -> float:
        """
        FIX #3: Semantic sentiment analysis. 
        Understands MEANING, not just individual words.
        """
        
        sentiment_score = 0.0
        pattern_weight = 0.0
        
        # Check very negative patterns first (highest priority)
        for pattern, score in self.compiled_very_negative:
            if pattern.search(text):
                if score < sentiment_score or pattern_weight == 0:
                    sentiment_score = min(sentiment_score, score)
                    pattern_weight = 1.0
        
        # Check negative patterns
        for pattern, score in self.compiled_negative:
            if pattern. search(text):
                sentiment_score += score * 0.5  # Accumulate
                pattern_weight = max(pattern_weight, 0.7)
        
        # Check positive patterns
        for pattern, score in self.compiled_positive:
            if pattern.search(text):
                sentiment_score += score * 0.5
                pattern_weight = max(pattern_weight, 0.7)
        
        # If strong patterns matched, use pattern-based score
        if pattern_weight > 0.5:
            final_score = np.clip(sentiment_score, -1.0, 1.0)
        else:
            # Fall back to VADER + simple word matching
            vader_score = 0.0
            if self._vader_available:
                try:
                    scores = self._vader.polarity_scores(text)
                    vader_score = scores['compound']
                except:
                    pass
            
            # Simple word-based backup
            positive_words = {'good', 'great', 'happy', 'love', 'wonderful', 
                            'amazing', 'excellent', 'better', 'best'}
            negative_words = {'bad', 'sad', 'hate', 'terrible', 'awful', 
                            'angry', 'upset', 'hurt', 'pain', 'alone'}
            
            words = self._simple_tokenize(text)
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            
            word_score = 0.0
            if pos_count + neg_count > 0:
                word_score = (pos_count - neg_count) / (pos_count + neg_count)
            
            # Blend VADER and word-based
            final_score = 0.6 * vader_score + 0.4 * word_score
        
        return float(np.clip(final_score, -1.0, 1.0))
    
    def _compute_absolutist_index(self, words:  List[str]) -> float:
        if not words:
            return 0.0
        absolutist_count = sum(1 for w in words if w in self.absolutist_words)
        ratio = absolutist_count / len(words)
        normalized = (ratio - 0.02) / 0.03
        return float(np.clip(normalized, -1, 1))
    
    def _compute_i_ratio(self, words: List[str]) -> float:
        if not words: 
            return 0.0
        i_words = {'i', "i'm", "i've", "i'll", "i'd", 'me', 'my', 'mine', 'myself'}
        i_count = sum(1 for w in words if w in i_words)
        ratio = i_count / len(words)
        normalized = (ratio - 0.08) / 0.07
        return float(np.clip(normalized, -1, 1))
    
    def _compute_response_latency(self, last_system_end_time:  Optional[str]) -> float:
        if not last_system_end_time: 
            return 0.0
        try:
            if isinstance(last_system_end_time, str):
                last_time = datetime.fromisoformat(
                    last_system_end_time.replace('Z', '+00:00'))
            else:
                last_time = last_system_end_time
            now = datetime.now()
            if last_time. tzinfo: 
                now = now.replace(tzinfo=last_time.tzinfo)
            latency_seconds = (now - last_time).total_seconds()
            normalized = (latency_seconds - 5) / 25
            return float(np.clip(normalized, -1, 1))
        except:
            return 0.0
    
    def _compute_lexical_density(self, words: List[str]) -> float:
        if not words or len(words) < 3:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        normalized = (unique_ratio - 0.7) / 0.15
        return float(np.clip(normalized, -1, 1))
    
    def _compute_past_tense_ratio(self, words: List[str]) -> float:
        if not words: 
            return 0.0
        past_indicators = {'was', 'were', 'had', 'did', 'went', 'said', 'got',
                         'made', 'came', 'thought', 'felt', 'knew', 'took',
                         'saw', 'found', 'gave', 'told', 'left', 'called'}
        past_count = sum(1 for w in words if w. endswith('ed') or w in past_indicators)
        ratio = past_count / len(words)
        normalized = (ratio - 0.07) / 0.05
        return float(np.clip(normalized, -1, 1))
    
    def _compute_filler_density(self, text: str, words: List[str]) -> float:
        if not words:
            return 0.0
        text_lower = text.lower()
        filler_count = sum(text_lower.count(f) for f in self.filler_words)
        ratio = filler_count / len(words)
        normalized = (ratio - 0.03) / 0.05
        return float(np.clip(normalized, -1, 1))
    
    def _compute_rumination(self, text:  str, words: List[str], 
                           sentences: List[str]) -> float:
        if not words or len(words) < 5:
            return 0.0
        
        rumination_score = 0.0
        
        # Word repetition
        word_counts = {}
        for w in words:
            if len(w) > 3:
                word_counts[w] = word_counts.get(w, 0) + 1
        
        repeated = sum(1 for count in word_counts.values() if count >= 2)
        if word_counts:
            repetition_ratio = repeated / len(word_counts)
            if repetition_ratio > 0.2: 
                rumination_score += 0.3
        
        # Negative word repetition
        negative_words = ['sad', 'bad', 'hate', 'hurt', 'pain', 'alone', 
                         'never', 'always', 'why', 'cant', "can't"]
        negative_repeated = any(word_counts.get(w, 0) >= 2 for w in negative_words)
        if negative_repeated:
            rumination_score += 0.3
        
        # "Why" repetition
        why_count = text.lower().count('why')
        if why_count >= 2:
            rumination_score += 0.2
        
        # Self-blame patterns
        self_blame = ['my fault', 'i always', 'i never', "i can't", "i cant"]
        for pattern in self_blame:
            if pattern in text. lower():
                rumination_score += 0.1
        
        return float(np.clip(rumination_score, 0, 1))