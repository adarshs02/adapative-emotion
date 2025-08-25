"""
EmotionGenerator: Extracts 3-5 crucial emotions from user situations.

This module identifies the most relevant emotions for a given situation,
providing the foundation for emotion-specific analysis.
"""

import json
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
        )
        print(f"🔍 Raw emotion response from abstract: {response}")
        
        # Parse structured text response
        emotions = self._parse_emotion_response(response)
        
        # Validate emotions count
        if not emotions or len(emotions) < 3:
            print("⚠️ Response parsing failed or insufficient emotions, using fallback")
            return []  # Basic fallback emotions
        elif len(emotions) > 5:
            # Take top 5 if too many
            emotions = emotions[:5]
        
        # Ensure emotions are valid strings
        emotions = [str(emotion).strip().lower() for emotion in emotions if emotion and str(emotion).strip()]
        
        # Final validation - ensure we have 3-5 emotions
        if len(emotions) < 3:
            print("⚠️ Final validation failed, using fallback")
            return []  # Basic fallback emotions
            
        return emotions[:5]  # Ensure max 5 emotions
    
    def _build_abstract_emotion_extraction_prompt(self, abstract: str) -> str:
        """Build the prompt for extracting crucial emotions from abstracts."""
        
        prompt = f"""TASK: Extract 3-5 crucial emotions from this abstract/summary.

ABSTRACT: {abstract}

Analyze the emotions someone would feel based on this summary. Choose 3-5 of the most important and distinct emotions.
It is extremely important to note that the emotion evolves over time, track these emotions and pay attention to the final emotion as it is the one that is most relevant to the situation.

Format your response as a simple list, one emotion per line:
emotion1
emotion2
emotion3

Example:
anxiety
sadness
hope
relief

Your emotions:"""
        
        return prompt
    
    def _parse_emotion_response(self, response: str) -> List[str]:
        """Parse emotion list from LLM structured text response."""
        if not response:
            return []
            
        # Split by lines and clean up
        lines = response.strip().split('\n')
        emotions = []
        
        for line in lines:
            line = line.strip().lower()
            # Skip empty lines and common prefixes
            if line and not line.startswith(('your emotions:', 'emotions:', '-', '*', '•')):
                # Remove any numbering or bullet points
                emotion = line.split('.', 1)[-1].strip()
                emotion = emotion.split(')', 1)[-1].strip()
                if emotion and len(emotion) > 1:  # Basic validation
                    emotions.append(emotion)
        
        return emotions
       
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
