"""
Tag generation system for emotion scenarios using structured vocabulary with LLM flexibility.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

# Core structured vocabulary for consistent tagging
CORE_TAG_VOCABULARY = {
    "actor_relationships": [
        "acquaintance", "parent", "child", "friend", "coworker_peer", "coworker_subordinate", "coworker_superior",
        "stranger", "authority_figure", "romantic_partner", "roommate", "neighbor", "customer", "peer",
        "service_worker", "crowd_member", "family_member", "business_partner", "doctor", "nurse"
    ],
    "trigger": [
        "betrayal", "lied", "threat", "neglect", "achievement", 
        "taking_property", "breaking_promise", "ignoring", "interrupting", 
        "humiliation", "criticism", "exclusion", "competition", "abandonment",
        "ignored", "blocked_goal", "property_damaged", "physical_harm", "mocked", "unexpected_gift",
        "unforseen_circumstances"
    ],
    "domains": ["work", "personal", "health", "financial", "social", "family", "academic"],

    "setting": [
        "home_private", "home_shared", "workplace_private", "workplace_shared","school", "public_transport",
         "street_public", "online_group", "social_media_public", "social_media_private"
    ],

    "temporal": ["sudden", "ongoing", "repetitive", "anticipated", "past", "future"],

    "control": ["controllable", "uncontrollable", "preventable", "inevitable"],

    "outcome": [
        "minor_inconvenience", "major_inconvenience", "material_loss_small", "material_loss_large",
        "physical_harm", "social_damage", "emotional_damage", "financial_damage", "safety_risk", "relationship_damage",
        "oportunity_gain"
    ]
}

TAG_GENERATION_PROMPT = """You are a tag generation system for emotion scenarios. Your task is to generate relevant tags that capture the key semantic elements of a situation.

CORE VOCABULARY (prefer these when applicable):
{core_vocabulary}

INSTRUCTIONS:
1. Generate 3-8 tags that best describe the situation
2. Prefer tags from the core vocabulary when they fit
3. Add specific descriptive tags if needed for important nuances not covered by core vocabulary
4. Focus on: actor relationships, trigger, domain, setting, and temporal aspects
5. Keep tags concise (1-3 words max)
6. Return ONLY a JSON array of tags, nothing else

SITUATION: "{situation}"

Tags:"""

class TagGenerator:
    """Generates structured tags for emotion scenarios."""
    
    def __init__(self):
        """Initialize the tag generator with LLM model."""
        self.model_name = config.LLM_MODEL_NAME
        self.device = config.get_device()
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model for tag generation."""
        print(f"Loading LLM for tag generation: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Tag generation LLM loaded successfully!")
    
    def _format_core_vocabulary(self):
        """Format the core vocabulary for the prompt."""
        formatted = []
        for category, tags in CORE_TAG_VOCABULARY.items():
            formatted.append(f"{category.upper()}: {', '.join(tags)}")
        return "\n".join(formatted)
    
    def generate_tags(self, situation: str) -> list:
        """
        Generate tags for a given situation.
        
        Args:
            situation: The situation description to generate tags for
            
        Returns:
            List of tags describing the situation
        """
        core_vocab_str = self._format_core_vocabulary()
        prompt = TAG_GENERATION_PROMPT.format(
            core_vocabulary=core_vocab_str,
            situation=situation
        )
        
        # Prepare input for the model
        messages = [{"role": "user", "content": prompt}]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,  # Lower temperature for more consistent tagging
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated text
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse tags from response
        tags = self._parse_tags_from_response(response)
        return tags
    
    def _parse_tags_from_response(self, response: str) -> list:
        """Parse tags from the LLM response."""
        print(f"DEBUG: Raw LLM response: '{response}'")
        
        try:
            # Handle potential markdown code blocks
            if "```" in response:
                start_block = response.find("```")
                end_block = response.rfind("```")
                if start_block != -1 and end_block != -1 and start_block != end_block:
                    code_block = response[start_block + 3 : end_block]
                    # Remove potential language specifier (like 'json')
                    if code_block.startswith('json'):
                        code_block = code_block[4:]
                    json_start = code_block.find('[')
                    if json_start != -1:
                        code_block = code_block[json_start:]
                    response = code_block.strip()
                    print(f"DEBUG: Extracted from code block: '{response}'")
            
            # Find JSON array in response
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index+1]
                print(f"DEBUG: Attempting to parse JSON: '{json_str}'")
                tags = json.loads(json_str)
                
                # Validate and clean tags
                if isinstance(tags, list):
                    print(f"DEBUG: Successfully parsed tags: {tags}")
                    # Clean and normalize tags
                    cleaned_tags = []
                    for tag in tags:
                        if isinstance(tag, str):
                            # Handle simple string tags
                            clean_tag = tag.strip().lower().replace(" ", "_")
                            if clean_tag and len(clean_tag) <= 30:  # Reasonable length limit
                                cleaned_tags.append(clean_tag)
                        elif isinstance(tag, dict):
                            # Handle object tags like {"actor_relationship": "roommate"}
                            for key, value in tag.items():
                                if isinstance(value, str):
                                    clean_tag = value.strip().lower().replace(" ", "_")
                                    if clean_tag and len(clean_tag) <= 30:
                                        cleaned_tags.append(clean_tag)
                    print(f"DEBUG: Cleaned tags: {cleaned_tags}")
                    return cleaned_tags[:8]  # Limit to 8 tags max
                else:
                    print(f"DEBUG: Parsed result is not a list: {type(tags)}")
            else:
                print(f"DEBUG: No JSON array found in response. start_index={start_index}, end_index={end_index}")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing tags from response: {e}")
            print(f"Raw response: {response}")
        
        # Fallback: try to extract words that look like tags
        print("DEBUG: Using fallback tag extraction")
        fallback_tags = self._fallback_tag_extraction(response)
        print(f"DEBUG: Fallback tags: {fallback_tags}")
        return fallback_tags
    
    def _fallback_tag_extraction(self, response: str) -> list:
        """Fallback method to extract tags if JSON parsing fails."""
        print(f"DEBUG: Fallback extraction for response: '{response}'")
        
        # Try multiple fallback strategies
        words = []
        
        # Strategy 1: Look for comma-separated words
        for line in response.split('\n'):
            if ',' in line:
                potential_tags = [w.strip().lower().replace(" ", "_") 
                                for w in line.split(',')]
                valid_tags = [tag for tag in potential_tags 
                            if tag and len(tag) <= 30 and tag.replace('_', '').isalpha()]
                words.extend(valid_tags)
        
        # Strategy 2: Look for words from our core vocabulary
        if not words:
            response_lower = response.lower()
            for category, vocab_tags in CORE_TAG_VOCABULARY.items():
                for vocab_tag in vocab_tags:
                    if vocab_tag in response_lower:
                        words.append(vocab_tag)
        
        # Strategy 3: Extract quoted words or bracketed words
        if not words:
            import re
            # Look for quoted words
            quoted_matches = re.findall(r'["\']([^"\'\']+)["\']', response)
            for match in quoted_matches:
                clean_tag = match.strip().lower().replace(" ", "_")
                if clean_tag and len(clean_tag) <= 30:
                    words.append(clean_tag)
        
        print(f"DEBUG: Fallback extracted words: {words}")
        return words[:6] if words else ["untagged"]
    
    def generate_tags_batch(self, situations: list) -> list:
        """
        Generate tags for multiple situations.
        
        Args:
            situations: List of situation descriptions
            
        Returns:
            List of tag lists, one for each situation
        """
        return [self.generate_tags(situation) for situation in situations]

# Global tag generator instance (lazy-loaded)
_tag_generator_instance = None

def get_tag_generator() -> TagGenerator:
    """Get the global tag generator instance (singleton pattern)."""
    global _tag_generator_instance
    if _tag_generator_instance is None:
        _tag_generator_instance = TagGenerator()
    return _tag_generator_instance

def generate_tags(situation: str) -> list:
    """
    Convenience function for generating tags.
    
    Args:
        situation: The situation description
        
    Returns:
        List of tags describing the situation
    """
    generator = get_tag_generator()
    return generator.generate_tags(situation)
