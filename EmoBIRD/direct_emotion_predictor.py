"""
DirectEmotionPredictor: Simplified emotion prediction without BIRD pooling.

This module directly asks the LLM to assess emotion probabilities given
the identified factor values, removing the need for complex pooling formulas.
"""

from typing import Dict, List, Any
from EmoBIRD.utils import validate_rating, RATING_SCALE


class DirectEmotionPredictor:
    """
    Direct emotion prediction using LLM assessment of factor combinations.
    """
    
    def __init__(self, vllm_wrapper, factors: List[Dict[str, Any]], emotions: List[str]):
        """
        Initialize direct emotion predictor.
        
        Args:
            vllm_wrapper: vLLM wrapper for LLM inference
            factors: List of factor definitions
            emotions: List of emotions to predict
        """
        self.vllm_wrapper = vllm_wrapper
        self.factors = factors
        self.emotions = emotions
        
    def predict_emotions(self, user_situation: str, factor_values: Dict[str, str], draft_essay: str = "") -> Dict[str, float]:
        """
        Directly predict emotion probabilities given factor values.
        
        Args:
            user_situation: User's situation description
            factor_values: Dictionary mapping factor names to their values
            
        Returns:
            Dictionary mapping emotion to probability
        """
        print("🎯 Direct emotion prediction from factor values...")
        
        # 1) Try strict JSON path first for reliable parsing
        try:
            json_prompt = self._build_emotion_assessment_json_prompt(user_situation, factor_values, draft_essay)
            schema = {
                "required": ["ratings"],
                "properties": {
                    "ratings": {"type": "object"},
                    "justification": {"type": "string"}
                }
            }
            data = self.vllm_wrapper.json_call(
                prompt=json_prompt,
                schema=schema,
                component="direct_emotion_predictor",
                interaction_type="emotion_assessment_json",
                max_retries=1,
                temperature_override=None,  # use strict/deterministic JSON settings
                max_tokens_override=getattr(self.vllm_wrapper.config, 'json_rating_max_tokens', 200),
            )
            print(f"🔍 JSON ratings response: {data}")
            # If JSON path returned fallback minimal (no 'ratings'), trigger text fallback instead of neutral defaults
            if not (isinstance(data, dict) and data.get("ratings")):
                print("⚠️ JSON ratings missing or empty; falling back to text parsing")
                raise ValueError("missing_ratings")
            emotion_probs = self._parse_emotion_json(data)
            return emotion_probs
        except Exception as e:
            print(f"⚠️ JSON ratings path failed, falling back to text parsing: {e}")
        
        # 2) Fallback to text prompt + parsing
        prompt = self._build_emotion_assessment_prompt(user_situation, factor_values, draft_essay)
        response = self.vllm_wrapper.generate(
            prompt,
            component="direct_emotion_predictor",
            interaction_type="emotion_assessment",
            temperature_override=0.6,
        )
        print(f"🔍 Raw emotion assessment (text): {response}")
        return self._parse_emotion_response(response)
    
    def _build_emotion_assessment_prompt(self, user_situation: str, factor_values: Dict[str, str], draft_essay: str = "") -> str:
        """
        Build prompt for direct emotion assessment.
        """
        # Format factor values for clarity
        factor_desc = "\n".join([f"- {factor}: {value}" for factor, value in factor_values.items()])
        
        # Format emotions list
        emotions_list = ", ".join(self.emotions)
        
        extra = f"\n\nDRAFT ESSAY (for context):\n{draft_essay}" if draft_essay else ""
        prompt = f"""Given the following situation and psychological factors, assess the likelihood of each emotion.

SITUATION:
{user_situation}

IDENTIFIED PSYCHOLOGICAL FACTORS:
{factor_desc}
{extra}

TASK:
For each emotion below, provide a likelihood rating based on the situation and factors above.
Use ONLY these exact ratings: very-unlikely, unlikely, neutral, likely, very-likely

EMOTIONS TO ASSESS:
{emotions_list}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS (one line per emotion):
emotion_name: rating

Example format:
joy: likely
anger: very-unlikely
sadness: neutral

Now provide your assessment WITHOUT ANY EXTRAS ONLY EXAMPLE FORMAT AS SHOWN ABOVE:"""
        
        return prompt

    def _build_emotion_assessment_json_prompt(self, user_situation: str, factor_values: Dict[str, str], draft_essay: str = "") -> str:
        """
        Build prompt that requests STRICT JSON with ratings only.
        """
        factor_desc = "\n".join([f"- {factor}: {value}" for factor, value in factor_values.items()])
        emotions_list = ", ".join(self.emotions)
        allowed = ", ".join(RATING_SCALE.keys())
        extra = f"\n\nDRAFT ESSAY (for context):\n{draft_essay}" if draft_essay else ""
        prompt = f"""You are evaluating likely emotions from a situation and its psychological factors.

SITUATION:
{user_situation}

FACTORS:
{factor_desc}
{extra}

TASK:
Return STRICT JSON with a single top-level object containing:
- ratings: an object mapping each emotion to one of these EXACT strings: {allowed}
- justification: a brief 1-2 sentence rationale (string)

EMOTIONS:
{emotions_list}

FORMAT:
{{
  "ratings": {{
    "joy": "likely",
    "sadness": "very-unlikely"
  }},
  "justification": "..."
}}

Write only the JSON object."""
        return prompt
    
    def _parse_emotion_response(self, response: str) -> Dict[str, float]:
        """
        Parse LLM response to extract emotion probabilities.
        """
        emotion_probs = {}
        
        # Split response into lines and process each
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
                
            try:
                # Parse "emotion: rating" format
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                    
                emotion = parts[0].strip().lower()
                rating = parts[1].strip().lower()
                
                # Validate the rating and convert to probability
                validated_rating = validate_rating(rating)
                probability = RATING_SCALE[validated_rating]
                
                # Store if emotion is in our list
                if any(emotion in e.lower() for e in self.emotions):
                    # Find the exact emotion name from our list
                    for e in self.emotions:
                        if emotion in e.lower():
                            emotion_probs[e] = probability
                            print(f"  ✅ {e}: {validated_rating} → {probability}")
                            break
                            
            except Exception as e:
                print(f"  ⚠️ Failed to parse line '{line}': {e}")
                continue
        
        # Fill in missing emotions with neutral probability
        for emotion in self.emotions:
            if emotion not in emotion_probs:
                print(f"  ⚠️ Missing assessment for {emotion}, using neutral (0.5)")
                emotion_probs[emotion] = 0.5
                
        return emotion_probs
    
    def _parse_emotion_json(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Parse ratings from strict JSON into probability map, defaulting missing to neutral (0.5).
        """
        emotion_probs: Dict[str, float] = {}
        ratings = {}
        try:
            if isinstance(data, dict):
                ratings = data.get("ratings", {}) or {}
        except Exception:
            ratings = {}
        
        # Normalize keys to lowercase for matching
        lower_map = {str(k).strip().lower(): v for k, v in ratings.items()}
        
        for emotion in self.emotions:
            key = emotion.lower()
            raw_rating = lower_map.get(key, "neutral")
            try:
                validated = validate_rating(str(raw_rating).strip().lower())
                emotion_probs[emotion] = RATING_SCALE[validated]
                print(f"  ✅ {emotion}: {validated} → {emotion_probs[emotion]}")
            except Exception as e:
                print(f"  ⚠️ Invalid JSON rating for '{emotion}' ('{raw_rating}'): {e}. Using neutral (0.5)")
                emotion_probs[emotion] = 0.5
        
        return emotion_probs
    
    def predict_with_explanation(self, user_situation: str, factor_values: Dict[str, str], draft_essay: str = "") -> Dict[str, Any]:
        """
        Predict emotions with explanation of reasoning.
        """
        # Get basic predictions
        emotion_probs = self.predict_emotions(user_situation, factor_values, draft_essay)
        
        # Get top emotions
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        top_emotions = dict(sorted_emotions[:3])
        
        # Generate explanation
        explanation = self._generate_explanation(user_situation, factor_values, top_emotions, draft_essay)
        
        return {
            'emotions': emotion_probs,
            'top_emotions': top_emotions,
            'factor_values': factor_values,
            'explanation': explanation
        }
    
    def _generate_explanation(self, user_situation: str, factor_values: Dict[str, str], 
                             top_emotions: Dict[str, float], draft_essay: str = "") -> str:
        """
        Generate explanation for the emotion predictions.
        """
        extra = f"\n\nDRAFT ESSAY (for context):\n{draft_essay}" if draft_essay else ""
        prompt = f"""Based on the psychological factors identified, explain why these emotions are most likely.

SITUATION: {user_situation}

FACTORS: {factor_values}

TOP EMOTIONS: {top_emotions}

Provide a brief 2-3 sentence explanation of how the factors lead to these emotions:{extra}"""
        
        response = self.vllm_wrapper.generate(
            prompt,
            component="direct_emotion_predictor",
            interaction_type="explanation",
            temperature_override=0.6,
        )
        
        return response
