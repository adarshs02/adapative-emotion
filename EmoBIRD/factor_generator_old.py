"""
Factor Generator for Emobird

Generates psychological factors from user input and abstract, similar to the poc_bird system.
This step occurs after abstract generation but before CPT generation.
"""

import json
from typing import Dict, Any, List


class FactorGenerator:
    """
    Generates psychological factors from user input and abstract.
    """
    
    def __init__(self, config):
        """Initialize the factor generator."""
        self.config = config
        self.vllm_wrapper = None
        
        # Default factor template (fallback)
        self.default_factors = [
            {
                'name': 'intensity',
                'description': 'Emotional intensity of the situation',
                'possible_values': ['low', 'high']
            },
            {
                'name': 'control',
                'description': 'Level of perceived control over the situation',
                'possible_values': ['no_control', 'full_control']
            },
            {
                'name': 'social_context',
                'description': 'Social setting of the situation',
                'possible_values': ['private', 'public']
            }
        ]
    
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper from parent class."""
        self.vllm_wrapper = vllm_wrapper
    
    def generate_factors(self, user_situation: str, abstract: str) -> Dict[str, Any]:
        """
        Generate 3 psychological factors from user input and abstract, with selected values.
        
        Args:
            user_situation: Original user input
            abstract: Generated abstract/summary of the situation
            
        Returns:
            Dictionary containing:
            - factors: List of 3 factor dictionaries with name, description, and possible_values (exactly 2 each)
            - selected_values: Dictionary mapping factor names to selected values for this situation
        """
        
        print("⚙️ Generating psychological factors and values...")
        
        prompt = self._build_factor_prompt(user_situation, abstract)
        
        if self.vllm_wrapper:
            response_data = self.vllm_wrapper.generate_json(
                prompt, 
                component="factor_generator", 
                interaction_type="factor_generation"
            )
            
            factors = response_data.get('factors', [])
            selected_values = response_data.get('selected_values', {})
            
            # Validate and ensure we have exactly 3 factors
            validated_factors = self._validate_factors(factors)
            validated_values = self._validate_factor_values(selected_values, validated_factors)
            
            return {
                'factors': validated_factors,
                'selected_values': validated_values
            }
        else:
            # Fallback to default factors
            print("⚠️ vLLM not available, using default factors")
            default_values = {factor['name']: factor['possible_values'][0] for factor in self.default_factors}
            return {
                'factors': self.default_factors,
                'selected_values': default_values
            }
    
    def _generate_factors_from_input_only(self, user_situation: str) -> Dict[str, Any]:
        """Generate factors using only user input (no abstract)."""
        
        print("⚙️ Generating factors from user input only...")
        
        # Use the new clean prompt template
        prompt = self._build_factor_prompt(user_situation, "")
        
        if self.vllm_wrapper:
            # Use regular text generation instead of JSON
            response_text = self.vllm_wrapper.generate(
                prompt, 
                component="factor_generator", 
                interaction_type="factor_generation_input_only"
            )
            
            # Parse factors from natural language response
            factors_data = self._parse_factors_from_text(response_text)
            factors = factors_data.get('factors', [])
            selected_values = factors_data.get('selected_values', {})
            
            validated_factors = self._validate_factors({'factors': factors})
            validated_values = self._validate_factor_values(selected_values, validated_factors)
            
            return {
                'factors': validated_factors,
                'selected_values': validated_values
            }
        else:
            default_values = {factor['name']: factor['possible_values'][0] for factor in self.default_factors}
            return {
                'factors': self.default_factors,
                'selected_values': default_values
            }
    
    def _generate_factors_with_abstract(self, user_situation: str, abstract: str) -> Dict[str, Any]:
        """Generate factors using both user input and abstract (current pipeline approach)."""
        
        print("⚙️ Generating factors with both input and abstract...")
        
        # This is the same as the main generate_factors method
        return self.generate_factors(user_situation, abstract)
    
    def _generate_factors_from_abstract_only(self, abstract: str) -> Dict[str, Any]:
        """Generate factors using only the abstract (no original user input)."""
        
        print("⚙️ Generating factors from abstract only...")
        
        prompt = f"""You are a psychological factor analysis expert. Given an abstract summary of an emotional situation, identify exactly 3 key psychological factors that would influence emotional outcomes AND select the appropriate value for each factor for this specific situation.

ABSTRACT SUMMARY:
"{abstract}"

Generate exactly 3 psychological factors that are:
1. Psychologically relevant to emotional outcomes
2. Inferable from the abstract summary
3. Have exactly 2 distinct value categories each
4. Cover different aspects (cognitive, social, situational, personal)

CRITICAL: Respond with ONLY valid JSON. No explanations, no extra text, no formatting outside JSON.

Return exactly this JSON structure:
{{
  "factors": [
    {{
      "name": "factor_name_1",
      "description": "Brief description of what this factor represents",
      "possible_values": ["value1", "value2"]
    }},
    {{
      "name": "factor_name_2", 
      "description": "Brief description of what this factor represents",
      "possible_values": ["value1", "value2"]
    }},
    {{
      "name": "factor_name_3",
      "description": "Brief description of what this factor represents", 
      "possible_values": ["value1", "value2"]
    }}
  ],
  "selected_values": {{
    "factor_name_1": "selected_value_1",
    "factor_name_2": "selected_value_2", 
    "factor_name_3": "selected_value_3"
  }}
}}

Requirements:
- Exactly 3 factors
- Each factor must have exactly 2 possible values
- Factor names should be lowercase with underscores
- Values should be lowercase with underscores
- Selected values must be from the possible_values list
- Focus on factors inferable from the abstract content"""
        
        if self.vllm_wrapper:
            response_data = self.vllm_wrapper.generate_json(
                prompt, 
                component="factor_generator", 
                interaction_type="factor_generation_abstract_only"
            )
            factors = response_data.get('factors', [])
            selected_values = response_data.get('selected_values', {})
            
            validated_factors = self._validate_factors(factors)
            validated_values = self._validate_factor_values(selected_values, validated_factors)
            
            return {
                'factors': validated_factors,
                'selected_values': validated_values
            }
        else:
            default_values = {factor['name']: factor['possible_values'][0] for factor in self.default_factors}
            return {
                'factors': self.default_factors,
                'selected_values': default_values
            }
    
    def _build_factor_prompt(self, user_situation: str, abstract: str) -> str:
        """Build the prompt for factor generation."""
        
        context = f"{user_situation}"
        if abstract:
            context += f"\n\nContext: {abstract}"
        
        prompt = f"""Situation: {context}

What are 3 key psychological factors that would influence someone's emotions in this situation? For each factor, describe what it represents and give 2 possible values.

For example:
- Anxiety level: can be low or high
- Social support: can be present or absent  
- Control: can be high or low

Please list 3 relevant factors for this specific situation:"""

        return prompt
    
    def _parse_factors_from_text(self, response_text: str) -> Dict[str, Any]:
        """Parse psychological factors from natural language LLM response."""
        
        factors = []
        selected_values = {}
        
        if not response_text or len(response_text.strip()) < 10:
            return {'factors': factors, 'selected_values': selected_values}
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('For example') or line.startswith('Please list'):
                continue
                
            # Look for patterns like:
            # "- Anxiety level: can be low or high"
            # "1. Control: high or low"
            # "Familiarity: familiar or unfamiliar"
            
            # Remove bullet points, numbers, etc.
            clean_line = line.lstrip('- 123456789.').strip()
            
            # Look for factor descriptions with colon
            if ':' in clean_line:
                parts = clean_line.split(':', 1)
                if len(parts) == 2:
                    factor_name = parts[0].strip().lower().replace(' ', '_')
                    description_part = parts[1].strip().lower()
                    
                    # Extract possible values from descriptions like "can be low or high"
                    possible_values = []
                    
                    # Common patterns to extract binary values
                    if ' or ' in description_part:
                        # "can be low or high" -> ["low", "high"]
                        value_part = description_part.replace('can be ', '').replace('either ', '')
                        values = [v.strip() for v in value_part.split(' or ')]
                        possible_values = [v.replace(' ', '_') for v in values if v]
                    
                    elif ' vs ' in description_part:
                        # "high vs low" -> ["high", "low"]
                        values = [v.strip() for v in description_part.split(' vs ')]
                        possible_values = [v.replace(' ', '_') for v in values if v]
                    
                    elif '/' in description_part:
                        # "present/absent" -> ["present", "absent"]
                        values = [v.strip() for v in description_part.split('/')]
                        possible_values = [v.replace(' ', '_') for v in values if v]
                    
                    # If we found valid values, create the factor
                    if len(possible_values) >= 2:
                        factor = {
                            'name': factor_name,
                            'description': f'Factor representing {factor_name.replace("_", " ")}',
                            'possible_values': possible_values[:2]  # Take first 2 for binary
                        }
                        factors.append(factor)
                        
                        # For now, select first value as default
                        selected_values[factor_name] = possible_values[0]
        
        return {'factors': factors, 'selected_values': selected_values}
    
    def _validate_factors(self, factors_input: Any) -> List[Dict[str, Any]]:
        """Validate and normalize the generated factors."""
        
        validated_factors = []
        
        # Handle new simple format: {"factors": {"name": ["val1", "val2"]}}
        if isinstance(factors_input, dict) and 'factors' in factors_input:
            factors_dict = factors_input['factors']
            if isinstance(factors_dict, dict):
                for factor_name, possible_values in factors_dict.items():
                    if isinstance(possible_values, list) and len(possible_values) >= 2:
                        validated_factor = {
                            'name': str(factor_name).strip().lower().replace(' ', '_'),
                            'description': f'Factor representing {factor_name}',
                            'possible_values': [str(v).strip().lower().replace(' ', '_') for v in possible_values[:2]]  # Take first 2 values
                        }
                        validated_factors.append(validated_factor)
        
        # Handle old array format: [{"name": ..., "possible_values": [...]}]
        elif isinstance(factors_input, list):
            for factor in factors_input:
                if not isinstance(factor, dict):
                    continue
                    
                # Ensure required fields
                if 'name' not in factor or 'possible_values' not in factor:
                    continue
                
                # Clean and validate factor
                validated_factor = {
                    'name': str(factor['name']).strip().lower().replace(' ', '_'),
                    'description': str(factor.get('description', '')).strip(),
                    'possible_values': []
                }
                
                # Validate possible values
                possible_values = factor.get('possible_values', [])
                if isinstance(possible_values, list):
                    for value in possible_values[:2]:  # Take first 2 values for binary
                        if isinstance(value, str) and value.strip():
                            validated_factor['possible_values'].append(value.strip().lower().replace(' ', '_'))
                
                # Ensure at least 2 values
                if len(validated_factor['possible_values']) >= 2:
                    validated_factors.append(validated_factor)
        
        # Ensure we have exactly 3 factors
        if len(validated_factors) >= 3:
            return validated_factors[:3]
        else:
            # Pad with default factors if needed
            print(f"⚠️ Only got {len(validated_factors)} valid factors, padding with defaults")
            while len(validated_factors) < 3:
                default_idx = len(validated_factors)
                if default_idx < len(self.default_factors):
                    validated_factors.append(self.default_factors[default_idx])
                else:
                    break
            
            return validated_factors[:3]
    
    def extract_factor_values(self, user_situation: str, abstract: str, 
                            factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract specific factor values for the given situation.
        
        Args:
            user_situation: Original user input
            abstract: Generated abstract
            factors: List of factor definitions
            
        Returns:
            Dictionary mapping factor names to selected values
        """
        
        print("🎯 Extracting factor values...")
        
        prompt = self._build_value_extraction_prompt(user_situation, abstract, factors)
        
        if self.vllm_wrapper:
            value_data = self.vllm_wrapper.generate_json(
                prompt, 
                component="factor_generator", 
                interaction_type="factor_value_extraction"
            )
            factor_values = value_data.get('factor_values', {})
            
            # Validate factor values against possible values
            return self._validate_factor_values(factor_values, factors)
        else:
            # Fallback: select first value for each factor
            return {factor['name']: factor['possible_values'][0] for factor in factors}
    
    def _build_value_extraction_prompt(self, user_situation: str, abstract: str, 
                                     factors: List[Dict[str, Any]]) -> str:
        """Build prompt for extracting specific factor values."""
        
        factors_json = json.dumps(factors, indent=2)
        
        prompt = f"""Situation: "{user_situation}"
Factors: {factors_json}

Select appropriate values. Output JSON only:

{{
  "factor_values": {{
    "factor_name": "selected_value"
  }}
}}"""

        return prompt
    
    def _validate_factor_values(self, factor_values: Dict[str, str], 
                              factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Validate that factor values are from the possible values list."""
        
        validated_values = {}
        
        for factor in factors:
            factor_name = factor['name']
            possible_values = factor['possible_values']
            
            # Get the selected value
            selected_value = factor_values.get(factor_name, '')
            
            # Validate it's in possible values
            if selected_value in possible_values:
                validated_values[factor_name] = selected_value
            else:
                # Fallback to first possible value
                print(f"⚠️ Invalid value '{selected_value}' for factor '{factor_name}', using '{possible_values[0]}'")
                validated_values[factor_name] = possible_values[0]
        
        return validated_values
    
    def get_factor_info(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary information about the generated factors."""
        
        return {
            'num_factors': len(factors),
            'factor_names': [f['name'] for f in factors],
            'total_combinations': self._calculate_combinations(factors),
            'factors': factors
        }
    
    def _calculate_combinations(self, factors: List[Dict[str, Any]]) -> int:
        """Calculate total number of possible factor combinations."""
        
        total = 1
        for factor in factors:
            total *= len(factor['possible_values'])
        return total
