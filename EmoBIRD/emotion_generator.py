"""
EmotionGenerator: Extracts 2-4 crucial emotions from user situations.

This module identifies the most relevant emotions for a given situation,
providing the foundation for emotion-specific analysis.
"""

import json
from typing import Dict, List, Any
from config import EmobirdConfig


class EmotionGenerator:
    """
    Generates 2-4 crucial emotions from user input situations.
    """
    
    def __init__(self, config: EmobirdConfig):
        """Initialize the emotion generator."""
        self.config = config
        self.vllm_wrapper = None
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper for inference."""
        self.vllm_wrapper = vllm_wrapper
        
    
    def extract_crucial_emotions_from_abstract(self, abstract: str) -> List[str]:
        """
        Extract 2-4 crucial emotions from an abstract/summary instead of full user situation.
        This is for comparison testing to see how abstracts vs full prompts affect emotion extraction.
        
        Args:
            abstract: Abstract/summary of the user's situation
            
        Returns:
            List of 2-4 emotion strings that are crucial for this situation
        """
        if not self.vllm_wrapper:
            raise ValueError("vLLM wrapper not set. Call set_vllm() first.")
            
        prompt = self._build_abstract_emotion_extraction_prompt(abstract)
        
        # Generate JSON response
        emotion_data = self.vllm_wrapper.generate_json(prompt)
        print(f"🔍 Raw emotion data from abstract: {emotion_data}")
        
        # Parse and validate emotions
        emotions = emotion_data.get('crucial_emotions', [])
        
        # If JSON parsing failed or returned invalid data, just print fallback message
        if not emotions or len(emotions) < 2:
            print("⚠️ JSON parsing failed or insufficient emotions, fell back")
            return []  # Return empty list if fallback needed
        elif len(emotions) > 4:
            # Take top 4 if too many
            emotions = emotions[:4]
        
        # Ensure emotions are valid strings
        emotions = [str(emotion).strip().lower() for emotion in emotions if emotion and str(emotion).strip()]
        
        # Final validation - ensure we have 2-4 emotions
        if len(emotions) < 2:
            print("⚠️ Final validation failed, fell back")
            return []  # Return empty list if fallback needed
            
        return emotions[:4]  # Ensure max 4 emotions
    
    def _build_abstract_emotion_extraction_prompt(self, abstract: str) -> str:
        """Build the prompt for extracting crucial emotions from abstracts."""
        
        prompt = f"""TASK: Extract 2-4 crucial emotions from this abstract/summary. Respond ONLY with valid JSON.

        ABSTRACT: {abstract}

        Analyze the emotions someone would feel based on this summary. Choose 2-4 of the most important and distinct emotions.

        Respond with ONLY this JSON format:
        {{
        "crucial_emotions": ["emotion1", "emotion2"]
        }}

        Do not include any other text. Just the JSON object."""
        
        return prompt
       
    def validate_emotions(self, emotions: List[str]) -> bool:
        """
        Validate that the extracted emotions are reasonable.
        
        Args:
            emotions: List of emotion strings
            
        Returns:
            True if emotions are valid, False otherwise
        """
        if not emotions or len(emotions) < 2 or len(emotions) > 4:
            return False
            
        # Check for duplicates
        if len(emotions) != len(set(emotions)):
            return False
            
        # Check that emotions are non-empty strings
        for emotion in emotions:
            if not isinstance(emotion, str) or not emotion.strip():
                return False
                
        return True
