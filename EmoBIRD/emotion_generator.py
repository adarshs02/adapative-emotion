"""
EmotionGenerator: Extracts 3-5 crucial emotions from user situations.

This module identifies the most relevant emotions for a given situation,
providing the foundation for emotion-specific analysis.
"""

import json
import re
from typing import Dict, List, Any
try:
    from EmoBIRD.config import EmobirdConfig
except ImportError:  # Allow running from within package dir
    from config import EmobirdConfig


class EmotionGenerator:
    """
    Generates 3-5 crucial emotions from user input situations.
    """
    
    def __init__(self, config: EmobirdConfig):
        """Initialize the emotion generator."""
        self.config = config
        self.vllm_wrapper = None
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper for inference."""
        self.vllm_wrapper = vllm_wrapper
        
    
    def extract_crucial_emotions(self, situation: str) -> List[str]:
        """
        Extract 3-5 crucial emotions from the full user situation.
        """
        if not self.vllm_wrapper:
            raise ValueError("vLLM wrapper not set. Call set_vllm() first.")
        
        prompt = self._build_emotion_extraction_prompt(situation)
        
        # Generate with a slightly higher temperature for diversity while staying stable
        response = self.vllm_wrapper.generate(
            prompt,
            component="emotion_generator",
            interaction_type="emotion_extraction_full",
            temperature_override=0.6,
            max_tokens_override=80,
            stop=[],  # disable global stops (e.g., "###") to avoid empty outputs
        )
        print(f"🔍 Raw emotion response (full): {response}")
        
        emotions = self._parse_emotion_response(response)
        
        if not emotions or len(emotions) < 3:
            print("⚠️ Insufficient emotions (full). Retrying deterministically...")
            # One-time deterministic retry to avoid stochastic empties
            response2 = self.vllm_wrapper.generate(
                prompt,
                component="emotion_generator",
                interaction_type="emotion_extraction_full_retry",
                temperature_override=0.0,
                max_tokens_override=128,
                stop=[],
            )
            print(f"🔁 Raw emotion response (full, retry): {response2}")
            emotions = self._parse_emotion_response(response2)
            if not emotions or len(emotions) < 3:
                print("⚠️ Still insufficient after retry (full). Using fallback defaults")
                return ["anxiety", "sadness", "frustration"]
        if len(emotions) > 5:
            emotions = emotions[:5]
        
        # Normalize
        emotions = [str(e).strip().lower() for e in emotions if e and str(e).strip()]
        
        if len(emotions) < 3:
            print("⚠️ Final validation failed (full), using fallback")
            fb = []
            try:
                defaults = getattr(self.config, "default_emotions", None)
                if defaults:
                    fb = [str(x).strip().lower() for x in defaults if str(x).strip()][:3]
            except Exception:
                fb = []
            if len(fb) < 3:
                fb = ["anxiety", "sadness", "frustration"]
            return fb
        return emotions[:5]
    
    def _build_emotion_extraction_prompt(self, situation: str) -> str:
        """Build the prompt for extracting crucial emotions from full situations."""
        prompt = f"""TASK: From the following situation, extract 3-5 crucial emotions.

SITUATION: {situation}

STRICT OUTPUT RULES (read carefully):
- Output EXACTLY 3 to 5 lines.
- Each line must be ONE standalone lowercase emotion word (e.g., anxiety, sadness, hope).
- Do NOT include any headers, prefaces, or explanations (e.g., do NOT write "Here are the emotions:").
- Do NOT include numbering, bullets, colons, parentheses, or extra words.
- Do NOT include punctuation other than an internal hyphen/apostrophe if part of a single word (e.g., self-doubt).

Allowed format (example only — do not include the word 'Example' in your output):
anxiety
sadness
hope
relief

Now output only the list of emotion words, one per line, and nothing else:"""
        return prompt
    
    def extract_crucial_emotions_from_abstract(self, abstract: str) -> List[str]:
        """
        Extract 3-5 crucial emotions from an abstract/summary instead of full user situation.
        This is for comparison testing to see how abstracts vs full prompts affect emotion extraction.
        
        Args:
            abstract: Abstract/summary of the user's situation
            
        Returns:
            List of 3-5 emotion strings that are crucial for this situation
        """
        if not self.vllm_wrapper:
            raise ValueError("vLLM wrapper not set. Call set_vllm() first.")
            
        prompt = self._build_abstract_emotion_extraction_prompt(abstract)
        
        # Generate structured text response with higher temperature for diversity
        response = self.vllm_wrapper.generate(
            prompt,
            component="emotion_generator",
            interaction_type="emotion_extraction",
            temperature_override=0.6,
            max_tokens_override=80,
            stop=[],  # disable global stops (e.g., "###") to avoid empty outputs
        )
        print(f"🔍 Raw emotion response from abstract: {response}")
        
        # Parse structured text response
        emotions = self._parse_emotion_response(response)
        
        # Validate emotions count
        if not emotions or len(emotions) < 3:
            print("⚠️ Insufficient emotions (abstract). Retrying deterministically...")
            response2 = self.vllm_wrapper.generate(
                prompt,
                component="emotion_generator",
                interaction_type="emotion_extraction_retry",
                temperature_override=0.0,
                max_tokens_override=128,
                stop=[],
            )
            print(f"🔁 Raw emotion response from abstract (retry): {response2}")
            emotions = self._parse_emotion_response(response2)
            if not emotions or len(emotions) < 3:
                print("⚠️ Still insufficient after retry (abstract). Using fallback defaults")
                fb = []
                try:
                    defaults = getattr(self.config, "default_emotions", None)
                    if defaults:
                        fb = [str(x).strip().lower() for x in defaults if str(x).strip()][:3]
                except Exception:
                    fb = []
                if len(fb) < 3:
                    fb = ["anxiety", "sadness", "frustration"]
                return fb
        elif len(emotions) > 5:
            # Take top 5 if too many
            emotions = emotions[:5]
        
        # Ensure emotions are valid strings
        emotions = [str(emotion).strip().lower() for emotion in emotions if emotion and str(emotion).strip()]
        
        # Final validation - ensure we have 3-5 emotions
        if len(emotions) < 3:
            print(" Final validation failed, using fallback")
            fb = []
            try:
                defaults = getattr(self.config, "default_emotions", None)
                if defaults:
                    fb = [str(x).strip().lower() for x in defaults if str(x).strip()][:3]
            except Exception:
                fb = []
            if len(fb) < 3:
                fb = ["anxiety", "sadness", "frustration"]
            return fb  # Basic fallback emotions
            
        return emotions[:5]  # Ensure max 5 emotions
    
    def _build_abstract_emotion_extraction_prompt(self, abstract: str) -> str:
        """Build the prompt for extracting crucial emotions from abstracts."""
        
        prompt = f"""TASK: Extract 3-5 crucial emotions from this abstract/summary.

ABSTRACT: {abstract}

Guidelines:
- Choose 3-5 distinct, important emotions, prioritizing the final emotional state if the narrative evolves.

STRICT OUTPUT RULES:
- Output EXACTLY 3 to 5 lines.
- Each line must be ONE standalone lowercase emotion word.
- No headers, prefaces, or explanations (e.g., do NOT write 'Here are...').
- No numbering, bullets, colons, parentheses, or extra words.

Allowed format (example only):
anxiety
sadness
hope
relief

Now output only the list of emotion words, one per line, and nothing else:"""
        
        return prompt
    
    def _parse_emotion_response(self, response: str) -> List[str]:
        """Parse emotion list from LLM structured text response."""
        if not response:
            return []
        
        emotions: List[str] = []
        lines = [ln.strip() for ln in response.strip().splitlines()]
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            # If the line has a label with a colon, strip the label and parse the right-hand side FIRST
            if ":" in line:
                left, right = line.split(":", 1)
                if any(kw in left.lower() for kw in ("emotion", "abstract", "here", "list", "output", "example", "following")):
                    line = right.strip()
                    low = line.lower()
                    if not line:
                        continue
            # Skip obvious headers/prefaces (when there is no RHS content to parse)
            if low.startswith((
                "your emotions",
                "example",
                "output",
                "list:",
                "here are",
                "here's",
                "below are",
                "the following",
            )):
                continue
            # Normalize spaced hyphens within words (e.g., "self - doubt" -> "self-doubt")
            line = re.sub(r"(?i)(\b[a-z]+)\s*-\s*([a-z]+\b)", r"\1-\2", line)
            # Remove labels like "emotion1:" at the beginning
            line = re.sub(r"^\s*emotions?\s*\d*\s*:\s*", "", line, flags=re.IGNORECASE)
            # Remove bullets/numbering like "- ", "1.", "1)", "(1)", "1:"
            line = re.sub(r"^\s*(?:[-*•]+|\(?\d+\)?[.):]?)\s*", "", line)
            # Split into multiple candidates on common delimiters
            parts = re.split(r"\s*[,;|/]\s*|\s+and\s+|\)\s*", line)
            for part in parts:
                token = part.strip().strip("'\"")
                if not token:
                    continue
                # Strip trailing punctuation like .,;:!?) and quotes/brackets
                token = token.rstrip(",.;:!?)\]}>”’'\"")
                t = token.lower()
                # Skip tokens that contain whitespace (not standalone words)
                if re.search(r"\s", t):
                    continue
                # Allow only alphabetic words with optional internal hyphen/apostrophe
                if not re.fullmatch(r"[a-z][a-z\-']{1,24}", t):
                    continue
                # Filter obvious non-emotion filler words
                if t in {"emotion", "emotions", "crucial", "extracted", "abstract", "here", "are", "the", "list"}:
                    continue
                emotions.append(t)
        
        # Deduplicate while preserving order
        seen = set()
        cleaned: List[str] = []
        for e in emotions:
            if e and e not in seen:
                seen.add(e)
                cleaned.append(e)
        # If nothing parsed, run a lenient fallback over the entire response
        if not cleaned:
            fallback: List[str] = []
            fillers = {"emotion", "emotions", "crucial", "extracted", "abstract", "here", "are", "the", "list", "is", "are", "and", "or", "of", "to", "from", "with"}
            for m in re.findall(r"[A-Za-z][A-Za-z\-']{1,24}", response):
                t = m.lower()
                if t in fillers:
                    continue
                if not re.fullmatch(r"[a-z][a-z\-']{1,24}", t):
                    continue
                if t not in fallback:
                    fallback.append(t)
                if len(fallback) >= 5:
                    break
            cleaned = fallback
        # Enforce max 5
        if len(cleaned) > 5:
            cleaned = cleaned[:5]
        return cleaned
       
    def validate_emotions(self, emotions: List[str]) -> bool:
        """
        Validate that the extracted emotions are reasonable.
        
        Args:
            emotions: List of emotion strings
            
        Returns:
            True if emotions are valid, False otherwise
        """
        if not emotions or len(emotions) < 3 or len(emotions) > 5:
            return False
            
        # Check for duplicates
        if len(emotions) != len(set(emotions)):
            return False
            
        # Check that emotions are non-empty strings
        for emotion in emotions:
            if not isinstance(emotion, str) or not emotion.strip():
                return False
                
        return True
