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
    
    def __init__(self, config, vllm_wrapper=None):
        """Initialize the scenario generator."""
        self.config = config
        self.vllm_wrapper = vllm_wrapper
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper from parent class."""
        self.vllm_wrapper = vllm_wrapper
    
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
        
        # Generate scenario using vLLM
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate(
                prompt, 
                component="scenario_generator", 
                interaction_type="scenario_generation"
            )
        else:
            # Fallback: create a basic scenario
            return self._create_fallback_scenario(user_situation)
        
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
        
        prompt = f"""You are an expert at generating summaries from user inputs. Summarize this user by looking out for the context, emotional impact, and the main idea in the shortest way possible:

"{user_situation}"

Write only the summary:"""
        
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_abstract(
                prompt, 
                component="scenario_generator", 
                interaction_type="abstract_generation"
            )
            print(f"🔍 Raw vLLM response: '{response}'")  # Debug print
            print(f"🔍 Response length: {len(response)} chars")  # Debug print
            # Extract clean abstract from potentially messy model output
            abstract = self._extract_clean_abstract(response)
            print(f"🔍 After extract_clean_abstract: '{abstract}'")  # Debug print
            return abstract
        else:
            # Fallback: create basic abstract
            return self._create_fallback_abstract(user_situation)
    
    def _extract_clean_abstract(self, response: str) -> str:
        """Extract clean abstract from potentially messy model output."""
        if not response or not response.strip():
            return ""
        
        # Start with the raw response
        text = response.strip()
        
        # If it starts with a quote, extract the quoted content
        if text.startswith('"'):
            # Find the end quote, but not if it's followed by more text
            end_quote_idx = text.find('"', 1)
            if end_quote_idx != -1:
                quoted_content = text[1:end_quote_idx].strip()
                # Only use quoted content if it looks like a summary (not too short)
                if len(quoted_content) > 10:
                    text = quoted_content
        
        # Split by common separators and take the first meaningful part
        # Stop at various markers that indicate unwanted content
        stop_markers = [
            '  #',  # Hashtags with space
            '\n#',  # Hashtags on new line
            'Bookmark',  # Spam content
            'Click here',  # Spam content
            '(Note:',  # Meta-commentary
            'Did I get it right?',  # Conversational
            'Word count:',  # Meta-analysis
            'Analysis:',  # Meta-analysis
            'Analyze:',  # Meta-analysis
            '#Summary',  # Analysis tags
            '#EmotionalImpact',  # Analysis tags
            '(I\'m looking',  # Meta-commentary
        ]
        
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Clean up any remaining formatting
        text = text.strip()
        
        # Remove trailing punctuation repetition
        while text.endswith('..') or text.endswith('  '):
            text = text.rstrip('. ')
        
        # Ensure it ends with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _build_scenario_prompt(self, user_situation: str, abstract: str) -> str:
        """Build the prompt for scenario generation using the abstract."""
        
        prompt = f"""Generate a scenario JSON from this situation and abstract:

Situation: "{user_situation}"
Abstract: "{abstract}"

Example output:
{{
  "id": "scenario_relationship_conflict",
  "description": "Relationship_Misunderstanding",
  "context": "frustration and sadness",
  "tags": ["relationship", "conflict", "emotional"]
}}

Now generate similar JSON for the given situation and abstract:
{{"""

        return prompt
    
    def _generate_scenario_from_input_only(self, user_situation: str) -> Dict[str, Any]:
        """Generate scenario using only user input (no abstract)."""
        prompt = f"""Generate a scenario JSON for this situation:

Situation: "{user_situation}"

Example output:
{{
  "id": "scenario_work_stress",
  "description": "Work_Presentation_Anxiety",
  "context": "nervous anticipation",
  "tags": ["work", "anxiety"]
}}

Now generate similar JSON for the given situation:
{{"""
        
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_json(
                prompt,
                component="scenario_generator",
                interaction_type="input_only_scenario"
            )
            return {
                'id': response.get('id', 'scenario_input_only'),
                'description': response.get('description', user_situation),
                'context': response.get('context', ''),
                'tags': response.get('tags', []),
                'method': 'input_only'
            }
        else:
            return self._create_fallback_scenario(user_situation)
    
    def _generate_scenario_with_abstract(self, user_situation: str, abstract: str) -> Dict[str, Any]:
        """Generate scenario using both user input and abstract (current pipeline approach)."""
        prompt = self._build_scenario_prompt(user_situation, abstract)
        
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_json(
                prompt,
                component="scenario_generator",
                interaction_type="input_plus_abstract_scenario"
            )
            return {
                'id': response.get('id', 'scenario_input_abstract'),
                'description': response.get('description', user_situation),
                'context': response.get('context', ''),
                'tags': response.get('tags', []),
                'abstract': abstract,
                'method': 'input_plus_abstract'
            }
        else:
            return self._create_fallback_scenario(user_situation)
    
    def _generate_scenario_from_abstract_only(self, abstract: str) -> Dict[str, Any]:
        """Generate scenario using only the abstract (no original user input)."""
        prompt = f"""Generate a scenario JSON from this abstract:

Abstract: "{abstract}"

Example output:
{{
  "id": "scenario_career_milestone",
  "description": "Career_Achievement_Celebration",
  "context": "pride and accomplishment",
  "tags": ["career", "success"]
}}

Now generate similar JSON for the given abstract:
{{"""
        
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_json(
                prompt,
                component="scenario_generator",
                interaction_type="abstract_only_scenario"
            )
            return {
                'id': response.get('id', 'scenario_abstract_only'),
                'description': response.get('description', abstract),
                'context': response.get('context', ''),
                'tags': response.get('tags', []),
                'method': 'abstract_only'
            }
        else:
            return {
                'id': 'fallback_abstract_only',
                'description': f"Scenario based on: {abstract}",
                'context': 'Generated from abstract only',
                'tags': ['abstract_based'],
                'method': 'abstract_only'
            }
    
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
