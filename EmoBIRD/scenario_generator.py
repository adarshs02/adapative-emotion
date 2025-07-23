"""
Scenario Generator for Emobird

Dynamically generates scenarios from user input rather than matching pre-existing scenarios.
"""

import json
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class ScenarioGenerator:
    """
    Generates scenarios dynamically from user input.
    """
    
    def __init__(self, config):
        """Initialize the scenario generator."""
        self.config = config
        self.tokenizer = None
        self.model = None
        
        # We'll reuse the main LLM from the parent class
        # This is set by the parent Emobird class
        
    def set_llm(self, tokenizer, model):
        """Set the LLM tokenizer and model from parent class."""
        self.tokenizer = tokenizer
        self.model = model
    
    def generate_scenario(self, user_situation: str) -> Dict[str, Any]:
        """
        Generate a scenario description and metadata from user input.
        Uses a two-step process: first generate abstract, then scenario.
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing:
            - id: Generated scenario ID
            - description: Scenario description
            - context: Additional context about the scenario
            - tags: Relevant tags for the scenario
            - abstract: Generated abstract of the user input
        """
        
        # Step 1: Generate abstract/summary of user input
        print("📋 Generating abstract of user input...")
        abstract = self._generate_abstract(user_situation)
        
        # Step 2: Generate scenario from abstract
        print("🎭 Generating scenario from abstract...")
        prompt = self._build_scenario_prompt(user_situation, abstract)
        
        # Generate scenario using LLM
        if self.tokenizer and self.model:
            response = self._generate_with_llm(prompt)
        else:
            # Fallback: create a basic scenario
            response = self._create_fallback_scenario(user_situation)
        
        try:
            scenario_data = json.loads(response)
            
            # Ensure required fields exist
            scenario = {
                'id': scenario_data.get('id', self._generate_scenario_id(user_situation)),
                'description': scenario_data.get('description', user_situation),
                'context': scenario_data.get('context', ''),
                'tags': scenario_data.get('tags', []),
                'abstract': abstract,
                'generated_from': user_situation[:100] + '...' if len(user_situation) > 100 else user_situation
            }
            
            return scenario
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse scenario JSON: {e}")
            return self._create_fallback_scenario(user_situation)
    
    def _generate_abstract(self, user_situation: str) -> str:
        """Generate an abstract/summary of the user input."""
        
        prompt = f"""You are an expert at creating concise abstracts. Given a user's emotional situation, create a brief, focused abstract that captures the key emotional elements and context.

USER SITUATION:
"{user_situation}"

Create a concise abstract (2-3 sentences) that:
1. Identifies the core emotional situation
2. Highlights key contextual factors
3. Captures the emotional tone

Respond with ONLY the abstract text, no additional formatting or explanation."""
        
        if self.tokenizer and self.model:
            response = self._generate_with_llm(prompt)
            # Clean up response - remove any markdown or extra formatting
            abstract = response.strip()
            if abstract.startswith('"') and abstract.endswith('"'):
                abstract = abstract[1:-1]
            return abstract
        else:
            # Fallback: create basic abstract
            return self._create_fallback_abstract(user_situation)
    
    def _build_scenario_prompt(self, user_situation: str, abstract: str) -> str:
        """Build the prompt for scenario generation using the abstract."""
        
        prompt = f"""You are a scenario analysis expert. Given a user's situation and its abstract summary, generate a comprehensive scenario description that captures the emotional context and relevant factors.

USER SITUATION:
"{user_situation}"

ABSTRACT SUMMARY:
"{abstract}"

Based on the abstract summary, generate a JSON response with the following structure:

{{
  "id": "scenario_<descriptive_name>",
  "description": "Clear, concise description of the emotional scenario",
  "context": "Additional context about the situation and relevant factors",
  "tags": ["tag1", "tag2", "tag3"]
}}

Requirements:
- The scenario should be emotionally relevant and capture key aspects
- Include 3-5 relevant tags that describe the situation type
- Keep description under 200 characters
- Context should provide deeper insights into emotional factors

Respond with ONLY the JSON object, no additional text."""

        return prompt
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using the LLM."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up response
        response = response.strip()
        
        # Extract JSON if wrapped in markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return response
    
    def _create_fallback_scenario(self, user_situation: str) -> Dict[str, Any]:
        """Create a basic fallback scenario when LLM generation fails."""
        
        scenario_id = self._generate_scenario_id(user_situation)
        
        return {
            'id': scenario_id,
            'description': f"Emotional situation involving: {user_situation[:100]}",
            'context': "Generated fallback scenario for dynamic analysis",
            'tags': self._extract_basic_tags(user_situation),
            'abstract': self._create_fallback_abstract(user_situation),
            'generated_from': user_situation
        }
    
    def _generate_scenario_id(self, user_situation: str) -> str:
        """Generate a scenario ID from user situation."""
        # Simple approach: take first few words and create ID
        words = user_situation.lower().replace(',', '').replace('.', '').split()[:3]
        scenario_id = "scenario_" + "_".join(words)
        return scenario_id
    
    def _create_fallback_abstract(self, user_situation: str) -> str:
        """Create a basic fallback abstract when LLM generation fails."""
        # Simple approach: truncate and identify key emotional indicators
        truncated = user_situation[:200]
        
        # Look for emotional keywords
        emotional_keywords = ['happy', 'sad', 'angry', 'excited', 'worried', 'stressed', 
                            'anxious', 'frustrated', 'disappointed', 'proud', 'scared', 'nervous']
        
        found_emotions = []
        situation_lower = user_situation.lower()
        for keyword in emotional_keywords:
            if keyword in situation_lower:
                found_emotions.append(keyword)
        
        if found_emotions:
            emotion_str = ", ".join(found_emotions[:2])  # Limit to 2 emotions
            return f"An emotional situation involving {emotion_str} feelings. {truncated}"
        else:
            return f"An emotional situation requiring analysis. {truncated}"
    
    def _extract_basic_tags(self, user_situation: str) -> list:
        """Extract basic tags from user situation using simple keyword matching."""
        
        situation_lower = user_situation.lower()
        
        # Common emotional keywords and their tags
        emotion_keywords = {
            'happy': 'positive',
            'sad': 'negative', 
            'angry': 'negative',
            'excited': 'positive',
            'worried': 'anxiety',
            'stressed': 'stress',
            'anxious': 'anxiety',
            'frustrated': 'negative',
            'disappointed': 'negative',
            'proud': 'positive',
            'scared': 'fear',
            'nervous': 'anxiety',
            'work': 'professional',
            'job': 'professional',
            'family': 'personal',
            'relationship': 'social',
            'friend': 'social',
            'school': 'academic',
            'test': 'academic',
            'exam': 'academic'
        }
        
        tags = []
        for keyword, tag in emotion_keywords.items():
            if keyword in situation_lower:
                if tag not in tags:
                    tags.append(tag)
        
        # Ensure we have at least one tag
        if not tags:
            tags.append('general')
        
        return tags[:5]  # Limit to 5 tags
    
    def batch_generate_scenarios(self, situations: list) -> list:
        """Generate scenarios for multiple situations."""
        scenarios = []
        for situation in situations:
            scenario = self.generate_scenario(situation)
            scenarios.append(scenario)
        return scenarios
