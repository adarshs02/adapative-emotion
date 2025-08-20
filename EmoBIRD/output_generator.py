"""
OutputGenerator: Conversational AI response generation for EmoBIRD.

This module generates natural, empathetic conversational responses that incorporate
emotion analysis insights in a human-like, supportive manner.
"""

from typing import Dict, List, Any
try:
    from EmoBIRD.config import EmobirdConfig
except ImportError:  # Allow running from within package dir
    from config import EmobirdConfig


class OutputGenerator:
    """
    Generates natural conversational responses using emotion analysis insights.
    """
    
    def __init__(self, config: EmobirdConfig):
        """Initialize the output generator."""
        self.config = config
        self.vllm_wrapper = None
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper for inference."""
        self.vllm_wrapper = vllm_wrapper
    
    def generate_response(self, 
                         user_input: str, 
                         top_emotions: Dict[str, float],
                         context: Dict[str, Any] = None) -> str:
        """
        Generate a natural conversational response incorporating emotion insights.
        
        Args:
            user_input: Original user situation description
            top_emotions: Dictionary of top emotions with probabilities
            context: Optional additional context (factors, abstract, etc.)
            
        Returns:
            Natural conversational response string
        """
        if not self.vllm_wrapper:
            raise ValueError("vLLM wrapper not set. Call set_vllm() first.")
        
        # Robust input validation to prevent string/type errors
        try:
            # Validate user_input
            if not isinstance(user_input, str):
                user_input = str(user_input) if user_input else "[No input provided]"
                
            # Validate top_emotions - handle case where it might be a string or malformed
            if isinstance(top_emotions, str):
                print(f"⚠️ Warning: top_emotions received as string: {top_emotions[:100]}...")
                top_emotions = {}  # Fallback to empty dict
            elif not isinstance(top_emotions, dict):
                print(f"⚠️ Warning: top_emotions is not a dict, got {type(top_emotions)}")
                top_emotions = {}  # Fallback to empty dict
            elif not top_emotions:  # Empty dict
                top_emotions = {}  # Ensure it's an empty dict, not None
                
            # Validate context
            if context is not None and not isinstance(context, dict):
                print(f"⚠️ Warning: context is not a dict, got {type(context)}")
                context = {}  # Fallback to empty dict
                
        except Exception as e:
            print(f"⚠️ Error in OutputGenerator input validation: {e}")
            # Set safe defaults
            user_input = str(user_input) if user_input else "[Input validation failed]"
            top_emotions = {}
            context = {}
        
        print("💬 Generating conversational response...")
        
        # Build conversational prompt
        prompt = self._build_conversational_prompt(user_input, top_emotions, context)
        
        # Generate response using standard text generation
        response = self.vllm_wrapper.generate(
            prompt, 
            component="output_generator", 
            interaction_type="conversational_response"
        )
        
        # Clean up the response
        cleaned_response = self._clean_response(response)
        
        print(f"✅ Generated response ({len(cleaned_response)} characters)")
        return cleaned_response
    
    def _build_conversational_prompt(self, 
                                   user_input: str, 
                                   top_emotions: Dict[str, float],
                                   context: Dict[str, Any] = None) -> str:
        """
        Build a conversational prompt that naturally incorporates emotion insights.
        
        Args:
            user_input: User's situation description
            top_emotions: Top emotions with probabilities
            context: Additional context
            
        Returns:
            Formatted conversational prompt
        """
        # Convert emotions to natural language
        emotion_insights = self._format_emotions_naturally(top_emotions)
        
        # Build context information if available
        context_info = ""
        if context:
            factors = context.get('factors', {})
            if factors:
                factor_list = [f"{k}: {v}" for k, v in factors.items()]
                context_info = f"\nAdditional context: The person is dealing with {', '.join(factor_list)}."
        
        prompt = f"""You are an empathetic, wise, and supportive AI assistant. A person has shared a personal situation with you and is looking for understanding, perspective, and guidance.

SITUATION SHARED:
{user_input}

EMOTIONAL INSIGHTS:
Based on psychological analysis, this person is likely experiencing: {emotion_insights}
{context_info}

INSTRUCTIONS:
Respond as a caring, insightful conversational partner would. Your response should:
- Acknowledge their feelings with genuine empathy and validation
- Naturally weave in emotional insights (don't just list them clinically)
- Provide thoughtful perspective and gentle guidance
- Be conversational and human-like, not robotic or overly structured
- Offer practical support or suggestions where appropriate
- Be encouraging while remaining realistic

Write a natural, flowing response as if you're a trusted friend or counselor who really understands what they're going through.

Response:"""

        return prompt
    
    def _format_emotions_naturally(self, emotions: Dict[str, float]) -> str:
        """
        Convert emotion probabilities into natural language description.
        
        Args:
            emotions: Dictionary of emotions with probabilities
            
        Returns:
            Natural language description of emotions
        """
        if not emotions:
            return "mixed emotions"
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 3 emotions
        top_emotions = sorted_emotions[:3]
        
        # Convert probabilities to intensity words
        emotion_descriptions = []
        for emotion, prob in top_emotions:
            if prob >= 0.8:
                intensity = "strong"
            elif prob >= 0.7:
                intensity = "significant"
            elif prob >= 0.6:
                intensity = "noticeable"
            else:
                intensity = "some"
            
            emotion_descriptions.append(f"{intensity} {emotion}")
        
        # Format naturally
        if len(emotion_descriptions) == 0:
            return "mixed emotions"
        if len(emotion_descriptions) == 1:
            return emotion_descriptions[0]
        elif len(emotion_descriptions) == 2:
            return f"{emotion_descriptions[0]} and {emotion_descriptions[1]}"
        else:
            return f"{', '.join(emotion_descriptions[:-1])}, and {emotion_descriptions[-1]}"
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format the generated response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response string
        """
        # Remove common artifacts
        cleaned = response.strip()
        
        # Remove "Response:" prefix if present
        if cleaned.lower().startswith("response:"):
            cleaned = cleaned[9:].strip()
        
        # Remove any markdown formatting that might have appeared
        cleaned = cleaned.replace("**", "").replace("*", "")
        
        # Remove trailing completion artifacts
        stop_phrases = [
            "\n\nHuman:",
            "\n\nUser:",
            "\n\nAssistant:",
            "\n---",
            "Is there anything else"
        ]
        
        for phrase in stop_phrases:
            if phrase in cleaned:
                cleaned = cleaned[:cleaned.find(phrase)].strip()
        
        # Ensure proper paragraph spacing
        lines = cleaned.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        # Join with double newlines for paragraph spacing
        result = '\n\n'.join(formatted_lines)
        
        return result
    
    def generate_follow_up_response(self, 
                                  original_input: str,
                                  original_response: str,
                                  follow_up_input: str,
                                  emotions: Dict[str, float]) -> str:
        """
        Generate a follow-up response that maintains conversation context.
        
        Args:
            original_input: User's original situation
            original_response: AI's previous response
            follow_up_input: User's follow-up message
            emotions: Current emotion analysis
            
        Returns:
            Contextualized follow-up response
        """
        prompt = f"""You are continuing a supportive conversation. Here's the context:

ORIGINAL SITUATION:
{original_input}

YOUR PREVIOUS RESPONSE:
{original_response}

USER'S FOLLOW-UP:
{follow_up_input}

CURRENT EMOTIONAL STATE:
{self._format_emotions_naturally(emotions)}

Continue the conversation naturally, building on what you've already discussed. Be supportive and maintain the caring tone from your previous response.

Response:"""

        response = self.vllm_wrapper.generate(
            prompt,
            component="output_generator",
            interaction_type="follow_up_response"
        )
        
        return self._clean_response(response)
    
    def get_response_info(self) -> Dict[str, Any]:
        """
        Get information about the output generation setup.
        
        Returns:
            Dictionary with generation configuration info
        """
        return {
            'response_style': 'conversational_empathetic',
            'emotion_integration': 'natural_language',
            'max_emotions': 3,
            'supports_follow_up': True,
            'tone': 'supportive_and_caring'
        }
