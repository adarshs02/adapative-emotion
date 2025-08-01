"""
Factor Generator for Emobird

Dynamically generates psychological factors from user input and abstracts.
"""

import json
from typing import Dict, Any, List


class FactorGenerator:
    """
    Generates psychological factors dynamically from user input.
    """
    
    def __init__(self, config, vllm_wrapper=None):
        """Initialize the factor generator."""
        self.config = config
        self.vllm_wrapper = vllm_wrapper
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper from parent class."""
        self.vllm_wrapper = vllm_wrapper
    
    def generate_factors_from_situation(self, user_situation: str) -> List[Dict[str, Any]]:
        """
        Generate psychological factors directly from user input only.
        No abstracts or scenarios needed - factors come purely from the situation.
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            List of factor dictionaries with name, description, and possible values
        """
        print("🧠 Generating factors directly from user situation...")
        
        # Build prompt focused only on user input
        prompt = self._build_direct_factor_prompt(user_situation)
        
        # Generate factors using vLLM
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_abstract(
                prompt, 
                component="factor_generator", 
                interaction_type="direct_factor_generation"
            )
            print(f"🔍 Raw factor response: '{response}'")
            
            # Parse factors from the response
            factors_data = self._parse_factors_from_response(response)
            
            if factors_data['factors']:
                return factors_data['factors']
            else:
                print("⚠️ No valid factors found in LLM response, using fallback")
                return self._create_fallback_factors(user_situation)['factors']
        else:
            return self._create_fallback_factors(user_situation)['factors']
    
    def extract_factor_values_direct(self, user_situation: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract factor values directly from user situation without using abstracts or scenarios.
        
        Args:
            user_situation: User's description of their situation
            factors: List of factor dictionaries
            
        Returns:
            Dictionary mapping factor names to their determined values
        """
        print("🎯 Extracting factor values directly from user situation...")
        
        if not self.vllm_wrapper:
            return self._create_fallback_analysis(user_situation, factors)
        
        # Build prompt for direct situation analysis
        prompt = self._build_direct_analysis_prompt(user_situation, factors)
        
        response = self.vllm_wrapper.generate_abstract(
            prompt,
            component="factor_generator",
            interaction_type="direct_situation_analysis"
        )
        
        print(f"🔍 Direct analysis response: '{response}'")
        
        # Parse the analysis response
        analysis = self._parse_analysis_response(response, factors)
        
        return analysis
    
    def _build_direct_factor_prompt(self, user_situation: str) -> str:
        """
        Build prompt for generating factors directly from user input only.
        No abstracts or external context needed.
        """
        prompt = f"""Analyze this situation and identify the 3 most important psychological factors that would influence the person's emotions:

SITUATION: {user_situation}

For each psychological factor, provide:
- Factor name (short, descriptive)
- Brief description of what this factor represents
- Two possible values (like high/low, present/absent, strong/weak, etc.)

Format exactly like this:
1. Factor name: Description (value1/value2)
2. Factor name: Description (value1/value2) 
3. Factor name: Description (value1/value2)

Example format:
1. stress_level: How much stress the person is experiencing (low/high)
2. social_support: Whether the person has people to rely on (absent/present)
3. control_perception: How much control the person feels they have (low/high)

Analyze the situation and provide 3 key psychological factors:"""
        
        return prompt
    
    def _build_direct_analysis_prompt(self, user_situation: str, factors: List[Dict[str, Any]]) -> str:
        """
        Build prompt for analyzing factor values directly from user situation.
        """
        factor_descriptions = []
        for factor in factors:
            values = "/".join(factor['possible_values'])
            factor_descriptions.append(f"- {factor['name']}: {factor['description']} ({values})")
        
        factors_text = "\n".join(factor_descriptions)
        
        prompt = f"""Analyze this situation and determine the specific value for each psychological factor:

SITUATION: {user_situation}

PSYCHOLOGICAL FACTORS:
{factors_text}

For each factor, choose the most appropriate value based on what you can observe in the situation. Provide a brief explanation.

Format exactly like this:
factor_name: chosen_value - brief explanation
factor_name: chosen_value - brief explanation

Example:
stress_level: high - person mentions feeling overwhelmed and anxious
social_support: present - mentions having family and friends available

Your analysis:"""
        
        return prompt
    
    def generate_factors(self, user_situation: str, abstract: str = "") -> Dict[str, Any]:
        """
        Generate psychological factors from user input.
        
        Args:
            user_situation: User's description of their situation
            abstract: Optional abstract/summary of the situation
            
        Returns:
            Dictionary containing:
            - factors: List of factor dictionaries
            - selected_values: Current values for each factor
            - generation_method: Method used for generation
        """
        
        print("🧠 Generating psychological factors...")
        
        # Build the prompt for factor generation
        prompt = self._build_factor_prompt(user_situation, abstract)
        
        # Generate factors using vLLM
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate_abstract(
                prompt, 
                component="factor_generator", 
                interaction_type="factor_generation"
            )
            print(f"🔍 Raw factor response: '{response}'")
            print(f"🔍 Response length: {len(response)} chars")
            
            # Parse factors from the response
            factors_data = self._parse_factors_from_response(response)
            
            if factors_data['factors']:
                return {
                    'factors': factors_data['factors'],
                    'selected_values': factors_data['selected_values'],
                    'generation_method': 'llm_generated'
                }
            else:
                print("⚠️ No valid factors found in LLM response, using fallback")
                return self._create_fallback_factors(user_situation)
        else:
            return self._create_fallback_factors(user_situation)
    
    def _build_factor_prompt(self, user_situation: str, abstract: str) -> str:
        """Build the prompt for factor generation."""
        
        context = user_situation
        if abstract:
            context += f"\n\nSummary: {abstract}"
        
        prompt = f"""Situation: {context}

What are 3 key psychological factors that would influence someone's emotions in this situation? 

For each factor, give:
- The factor name
- A brief description  
- Two possible values (like high/low, present/absent, etc.)

Format like this:
1. Factor name: Description (value1/value2)
2. Factor name: Description (value1/value2) 
3. Factor name: Description (value1/value2)

Example:
1. Anxiety level: How anxious the person feels (low/high)
2. Social support: Whether support is available (absent/present)
3. Control: How much control they feel (low/high)

Please list 3 factors for this situation:"""
        
        return prompt
    
    def _parse_factors_from_response(self, response: str) -> Dict[str, Any]:
        """Parse psychological factors from LLM response."""
        
        factors = []
        selected_values = {}
        
        if not response or len(response.strip()) < 10:
            return {'factors': factors, 'selected_values': selected_values}
        
        # Clean the response
        response = self._extract_clean_response(response)
        lines = response.strip().split('\n')
        
        # First try multi-line bullet format parsing
        factors_multi = self._parse_multiline_bullet_format(lines)
        if factors_multi:
            for factor in factors_multi:
                factors.append(factor)
                selected_values[factor['name']] = factor['possible_values'][0]
        
        # Then try single-line format parsing for any remaining lines
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering and bullets
            clean_line = line.lstrip('123456789.- ').strip()
            
            # Look for pattern: "Factor name: Description (value1/value2)"
            if ':' in clean_line and '(' in clean_line and ')' in clean_line:
                try:
                    # Split on colon
                    name_part, desc_part = clean_line.split(':', 1)
                
                    # Clean factor name - remove markdown formatting and asterisks
                    raw_name = name_part.strip()
                    factor_name = raw_name.replace('*', '').replace('#', '').strip().lower().replace(' ', '_')
                    
                    # Skip if we already have this factor from multi-line parsing
                    if any(f['name'] == factor_name for f in factors):
                        continue
                    
                    # Extract values from parentheses
                    desc_part = desc_part.strip()
                    if '(' in desc_part and ')' in desc_part:
                        values_start = desc_part.rfind('(')
                        values_end = desc_part.rfind(')')
                        values_text = desc_part[values_start+1:values_end]
                        description = desc_part[:values_start].strip()
                        
                        # Parse values (value1/value2 or value1 or value2)
                        possible_values = []
                        if '/' in values_text:
                            values = [v.strip().lower().replace(' ', '_') for v in values_text.split('/')]
                            possible_values = [v for v in values if v]
                        elif ' or ' in values_text:
                            values = [v.strip().lower().replace(' ', '_') for v in values_text.split(' or ')]
                            possible_values = [v for v in values if v]
                        
                        # Create factor if we have valid values
                        if len(possible_values) >= 2:
                            factor = {
                                'name': factor_name,
                                'description': description or f'Factor representing {factor_name.replace("_", " ")}',
                                'possible_values': possible_values[:2]  # Take first 2 for binary
                            }
                            factors.append(factor)
                            
                            # Select first value as default
                            selected_values[factor_name] = possible_values[0]
                            
                except Exception as e:
                    print(f"⚠️ Error parsing factor line '{line}': {e}")
                    continue
        
        return {'factors': factors, 'selected_values': selected_values}
    
    def _parse_multiline_bullet_format(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse multi-line bullet format factors like:
        1. **Factor Name**: Description
                * High: explanation
                * Low: explanation
        """
        
        factors = []
        current_factor = None
        current_values = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering and bullets from start
            clean_line = line.lstrip('123456789.- ').strip()
            
            # Check if this is a factor name line (has colon, no bullet values)
            if (':' in clean_line and 
                not clean_line.strip().startswith('*') and 
                not clean_line.strip().startswith('-') and
                '(' not in clean_line):  # Not single-line format
                
                # Save previous factor if we have one
                if current_factor and len(current_values) >= 2:
                    factor = {
                        'name': current_factor['name'],
                        'description': current_factor['description'],
                        'possible_values': current_values[:2]  # Take first 2 for binary
                    }
                    factors.append(factor)
                
                # Start new factor
                try:
                    name_part, desc_part = clean_line.split(':', 1)
                    # Clean factor name (remove asterisks, make lowercase, replace spaces)
                    factor_name = name_part.strip().replace('*', '').strip().lower().replace(' ', '_')
                    description = desc_part.strip()
                    
                    current_factor = {
                        'name': factor_name,
                        'description': description
                    }
                    current_values = []
                    
                except Exception as e:
                    print(f"⚠️ Error parsing factor name line '{line}': {e}")
                    continue
            
            # Check if this is a value line (starts with * or -)
            elif (clean_line.startswith('*') or clean_line.startswith('-')) and current_factor:
                try:
                    # Extract value name
                    value_line = clean_line.lstrip('*- ').strip()
                    if ':' in value_line:
                        value_name = value_line.split(':', 1)[0].strip().lower().replace(' ', '_')
                        if value_name and value_name not in current_values:
                            current_values.append(value_name)
                            
                except Exception as e:
                    print(f"⚠️ Error parsing value line '{line}': {e}")
                    continue
        
        # Save the last factor if we have one
        if current_factor and len(current_values) >= 2:
            factor = {
                'name': current_factor['name'], 
                'description': current_factor['description'],
                'possible_values': current_values[:2]  # Take first 2 for binary
            }
            factors.append(factor)
        
        return factors
    
    def _extract_clean_response(self, response: str) -> str:
        """Extract clean response from potentially messy model output."""
        
        # Remove markdown code blocks if present
        if "```" in response:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
        
        # Remove common prefixes/suffixes
        response = response.strip()
        for prefix in ["Here are", "The factors are", "Factors:", "Answer:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _create_fallback_factors(self, user_situation: str) -> Dict[str, Any]:
        """Create fallback factors when LLM generation fails."""
        
        print("🔄 Creating fallback factors...")
        
        # Create basic psychological factors that apply to most emotional situations
        factors = [
            {
                'name': 'stress_level',
                'description': 'The level of stress experienced in the situation',
                'possible_values': ['low', 'high']
            },
            {
                'name': 'social_support',
                'description': 'Whether social support is available',
                'possible_values': ['absent', 'present']
            },
            {
                'name': 'control',
                'description': 'How much control the person feels they have',
                'possible_values': ['low', 'high']
            }
        ]
        
        # Set default values based on simple keyword analysis
        selected_values = {}
        situation_lower = user_situation.lower()
        
        # Stress level heuristic
        stress_keywords = ['stress', 'anxious', 'worried', 'nervous', 'pressure']
        if any(keyword in situation_lower for keyword in stress_keywords):
            selected_values['stress_level'] = 'high'
        else:
            selected_values['stress_level'] = 'low'
        
        # Social support heuristic  
        social_keywords = ['alone', 'isolated', 'by myself', 'no one']
        if any(keyword in situation_lower for keyword in social_keywords):
            selected_values['social_support'] = 'absent'
        else:
            selected_values['social_support'] = 'present'
        
        # Control heuristic
        control_keywords = ['can\'t', 'unable', 'helpless', 'stuck', 'no choice']
        if any(keyword in situation_lower for keyword in control_keywords):
            selected_values['control'] = 'low'
        else:
            selected_values['control'] = 'high'
        
        return {
            'factors': factors,
            'selected_values': selected_values,
            'generation_method': 'fallback'
        }
    
    def analyze_situation(self, user_situation: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Analyze the user situation to determine factor values.
        
        Args:
            user_situation: User's description of their situation
            factors: List of factor dictionaries
            
        Returns:
            Dictionary mapping factor names to their determined values
        """
        
        print("🔍 Analyzing situation for factor values...")
        
        if not self.vllm_wrapper:
            return self._create_fallback_analysis(user_situation, factors)
        
        # Build prompt for situation analysis
        prompt = self._build_analysis_prompt(user_situation, factors)
        
        response = self.vllm_wrapper.generate_abstract(
            prompt,
            component="factor_generator",
            interaction_type="situation_analysis"
        )
        
        print(f"🔍 Analysis response: '{response}'")
        
        # Parse the analysis response
        analysis = self._parse_analysis_response(response, factors)
        
        return analysis
    
    def _build_analysis_prompt(self, user_situation: str, factors: List[Dict[str, Any]]) -> str:
        """Build prompt for analyzing factor values in the situation."""
        
        factor_descriptions = []
        for factor in factors:
            values = "/".join(factor['possible_values'])
            factor_descriptions.append(f"- {factor['name']}: {factor['description']} ({values})")
        
        factors_text = "\n".join(factor_descriptions)
        
        prompt = f"""Situation: {user_situation}

Based on this situation, determine the value for each psychological factor:

{factors_text}

For each factor, choose one of the possible values and briefly explain why.

Format:
factor_name: chosen_value - explanation
factor_name: chosen_value - explanation

Example:
stress_level: high - person shows clear signs of stress and worry
social_support: present - mentions having friends to talk to

Your analysis:"""
        
        return prompt
    
    def _parse_analysis_response(self, response: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Parse factor value analysis from LLM response."""
        
        analysis = {}
        
        if not response:
            return self._create_fallback_analysis("", factors)
        
        lines = response.strip().split('\n')
        factor_names = {f['name'] for f in factors}
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    factor_name = parts[0].strip().lower().replace(' ', '_')
                    value_part = parts[1].strip()
                    
                    # Extract just the value (before any explanation)
                    if ' - ' in value_part:
                        value = value_part.split(' - ')[0].strip().lower().replace(' ', '_')
                    else:
                        value = value_part.split()[0].strip().lower().replace(' ', '_')
                    
                    # Validate factor name and value
                    if factor_name in factor_names:
                        # Find the factor and check if value is valid
                        for factor in factors:
                            if factor['name'] == factor_name:
                                if value in factor['possible_values']:
                                    analysis[factor_name] = value
                                break
        
        # Fill in missing values with fallback analysis
        fallback = self._create_fallback_analysis("", factors)
        for factor in factors:
            if factor['name'] not in analysis:
                analysis[factor['name']] = fallback.get(factor['name'], factor['possible_values'][0])
        
        return analysis
    
    def _create_fallback_analysis(self, user_situation: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create fallback analysis when LLM analysis fails."""
        
        analysis = {}
        
        # Default to first value for each factor
        for factor in factors:
            analysis[factor['name']] = factor['possible_values'][0]
        
        return analysis
