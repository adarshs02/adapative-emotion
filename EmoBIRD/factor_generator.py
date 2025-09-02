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
    
    def _sanitize_situation_text(self, user_situation: str) -> str:
        """Sanitize situation text to avoid contaminating prompts with unrelated formatting blocks.

        - If the situation contains a block starting with 'Respond in exactly this output format:',
          drop everything from that marker onward.
        - Also truncate before headers like "# I'm thinking & feeling" if present.
        - Preserve the original dialogue/content before those markers.
        """
        if not user_situation:
            return user_situation
        markers = [
            "Respond in exactly this output format:",
            "# I'm thinking & feeling",
        ]
        cleaned = user_situation
        for m in markers:
            idx = cleaned.find(m)
            if idx != -1:
                cleaned = cleaned[:idx].strip()
        return cleaned
    
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
        
        # Generate factors using vLLM (general generator, not abstract-constrained)
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate(
                prompt, 
                component="factor_generator", 
                interaction_type="direct_factor_generation",
                max_tokens_override=512,
                stop=[],
            )
            print(f"🔍 Raw factor response: '{response}'")
            # If the model starts with the sentinel, retry with a clearer instruction
            if response and response.strip().startswith("END_OF_FACTORS"):
                retry_prompt = (
                    prompt
                    + "\nDo not start with END_OF_FACTORS. Start the list immediately with '1.' and only output END_OF_FACTORS after the list."
                )
                print("♻️ Response started with sentinel; retrying with delayed-sentinel instruction...")
                response = self.vllm_wrapper.generate(
                    retry_prompt,
                    component="factor_generator",
                    interaction_type="direct_factor_generation",
                    max_tokens_override=512,
                    stop=[],
                )
                print(f"🔁 Retry raw factor response (delayed sentinel): '{response}'")
            
            # Guard: if response is empty (e.g., early stop at sentinel), retry once with a clearer start hint
            if not response or not response.strip():
                retry_prompt = prompt + "\nStart the list immediately with '1.' on its own line."
                print("♻️ Empty response detected; retrying factor generation once with a start hint...")
                response = self.vllm_wrapper.generate(
                    retry_prompt,
                    component="factor_generator",
                    interaction_type="direct_factor_generation",
                    max_tokens_override=512,
                    stop=[],
                )
                print(f"🔁 Retry raw factor response: '{response}'")
            
            # Parse factors from the response
            factors_data = self._parse_factors_from_response(response)
            
            if factors_data['factors'] and len(factors_data['factors']) >= 3:
                return factors_data['factors']
            elif factors_data['factors'] and len(factors_data['factors']) < 3:
                print(f"⚠️ Only {len(factors_data['factors'])} factors found, adding fallback factors to reach minimum of 3")
                # Add fallback factors to reach minimum of 3
                existing_names = {f['name'] for f in factors_data['factors']}
                fallback_data = self._create_fallback_factors(user_situation)
                
                for fallback_factor in fallback_data['factors']:
                    if fallback_factor['name'] not in existing_names and len(factors_data['factors']) < 3:
                        factors_data['factors'].append(fallback_factor)
                        
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
        
        response = self.vllm_wrapper.generate(
            prompt,
            component="factor_generator",
            interaction_type="direct_situation_analysis",
            max_tokens_override=512,
            stop=[],
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
        situation = self._sanitize_situation_text(user_situation)
        prompt = f"""You are an expert at analyzing situations and identifying psychological/social factors that would influence the person's emotions. Analyze this situation and identify 3 important factors that would influence the person's emotions:

SITUATION: {situation}

For each factor, provide:
- Factor name (short, descriptive)
- Two possible values (like high/low, present/absent, strong/weak, etc.)

Provide AT LEAST 3 factors:
1. Factor name: Description (value1/value2)
2. Factor name: Description (value1/value2) 
3. Factor name: Description (value1/value2)
 Do not include any explanations after the factors. Do not start with END_OF_FACTORS.
 After you list the factors, output a single line exactly: END_OF_FACTORS
 Never output END_OF_FACTORS before the list. Only place it after the final factor line.


Example:
1. Self-efficacy: Person's belief in their ability to handle challenges (low/high)
2. Social support: Availability of emotional support from others (absent/present)
3. Stress level: Amount of psychological pressure experienced (low/high)
 
 Now list the factors:"""
        
        return prompt
    
    def _build_direct_analysis_prompt(self, user_situation: str, factors: List[Dict[str, Any]]) -> str:
        """
        Build prompt for analyzing factor values directly from user situation.
        """
        situation = self._sanitize_situation_text(user_situation)
        factor_descriptions = []
        for factor in factors:
            values = "/".join(factor['possible_values'])
            factor_descriptions.append(f"- {factor['name']}: {factor['description']} ({values})")
        
        factors_text = "\n".join(factor_descriptions)
        
        prompt = f"""Analyze this situation and determine the specific value for each psychological factor:

SITUATION: {situation}

PSYCHOLOGICAL FACTORS:
{factors_text}

For each factor, choose the most appropriate value based on what you can observe in the situation. Provide a brief explanation.

Format exactly like this:
factor_name: chosen_value - brief explanation
factor_name: chosen_value - brief explanation

Example:
stress_level: high - person mentions feeling overwhelmed and anxious
social_support: present - mentions having family and friends available

Only output the lines in the specified format above. Do not add headings, prefaces, or trailing commentary.
After your last line, output a single line exactly: END_OF_ANALYSIS

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
        
        # Build the prompt for factor generation (standardized on direct flow)
        prompt = self._build_direct_factor_prompt(user_situation)
        
        # Generate factors using vLLM (general generator, not abstract-constrained)
        if self.vllm_wrapper:
            response = self.vllm_wrapper.generate(
                prompt,
                component="factor_generator",
                interaction_type="direct_factor_generation",
                max_tokens_override=512,
                stop=[],
            )
            print(f"🔍 Raw factor response: '{response}'")
            print(f"🔍 Response length: {len(response)} chars")
            # If the model starts with the sentinel, retry with a clearer instruction
            if response and response.strip().startswith("END_OF_FACTORS"):
                retry_prompt = (
                    prompt
                    + "\nDo not start with END_OF_FACTORS. Start the list immediately with '1.' and only output END_OF_FACTORS after the list."
                )
                print("♻️ Response started with sentinel; retrying with delayed-sentinel instruction...")
                response = self.vllm_wrapper.generate(
                    retry_prompt,
                    component="factor_generator",
                    interaction_type="direct_factor_generation",
                    max_tokens_override=512,
                    stop=[],
                )
                print(f"🔁 Retry raw factor response (delayed sentinel): '{response}'")
            
            # Guard: if response is empty (e.g., early stop at sentinel), retry once with a clearer start hint
            if not response or not response.strip():
                retry_prompt = prompt + "\nStart the list immediately with '1.' on its own line."
                print("♻️ Empty response detected; retrying factor generation once with a start hint...")
                response = self.vllm_wrapper.generate(
                    retry_prompt,
                    component="factor_generator",
                    interaction_type="direct_factor_generation",
                    max_tokens_override=512,
                    stop=[],
                )
                print(f"🔁 Retry raw factor response: '{response}'")
            
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
    
    def _parse_factors_from_response(self, response: str) -> Dict[str, Any]:
        """Parse psychological factors from LLM response."""
        
        factors = []
        selected_values = {}
        
        if not response or len(response.strip()) < 10:
            return {'factors': factors, 'selected_values': selected_values}
        
        # Clean the response
        response = self._extract_clean_response(response)
        # Drop empty and sentinel-only lines to reduce noise
        lines = [
            l for l in response.strip().split('\n')
            if l.strip() and "end_of_factors" not in l.strip().lower()
        ]
        
        # First try multi-line bullet format parsing
        factors_multi = self._parse_multiline_bullet_format(lines)
        if factors_multi:
            for factor in factors_multi:
                factors.append(factor)
                if len(factor['possible_values']) > 0:
                    selected_values[factor['name']] = factor['possible_values'][0]
        
        # Then try single-line format parsing for any remaining lines
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
                
            # Remove numbering and bullets
            clean_line = line.lstrip('123456789.- ').strip()
            
            # Look for pattern: "Factor name: Description (value1/value2)" or
            #                     "Factor name: <ActualName> (low/high)"
            if ':' in clean_line and '(' in clean_line and ')' in clean_line:
                try:
                    # Split on colon
                    name_part, desc_part = clean_line.split(':', 1)
                
                    # Derive factor name
                    raw_left = name_part.strip()
                    left_clean = raw_left.replace('*', '').replace('#', '').strip().lower()
                    desc_part = desc_part.strip()
                    # Extract the text before '(' on the right side
                    pre_paren = desc_part[:desc_part.rfind('(')].strip() if '(' in desc_part and ')' in desc_part else desc_part
                    # If left side is generic like "factor name", use right side pre-paren as the actual name
                    if left_clean in ("factor name", "factor", "name") and pre_paren:
                        raw_name_source = pre_paren
                    else:
                        raw_name_source = raw_left
                    # Final normalized factor_name
                    factor_name = raw_name_source.replace('*', '').replace('#', '').strip().lower().replace(' ', '_')
                    if not factor_name:
                        # Skip if we couldn't derive a name
                        continue
                    
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
                        vt = values_text.lower()
                        # Remove common labels like 'value1', 'value2', 'value'
                        for token in ["value1", "value2", "value 1", "value 2", "value:", "value"]:
                            vt = vt.replace(token, "")
                        # Normalize separators to '/'
                        vt = vt.replace(",", "/")
                        vt = vt.replace(" or ", "/")
                        # Split and normalize tokens
                        values = [v.strip().replace(' ', '_') for v in vt.split('/')]
                        # De-duplicate while preserving order
                        seen = set()
                        possible_values = []
                        for v in values:
                            if v and v not in seen:
                                seen.add(v)
                                possible_values.append(v)
                        
                        # Create factor if we have valid values
                        if len(possible_values) >= 2:
                            factor = {
                                'name': factor_name,
                                'description': description or f'Factor representing {factor_name.replace("_", " ")}',
                                # Keep up to 3 options to accommodate "medium" style outputs
                                'possible_values': possible_values[:3]
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

        Also supports headers like:
        1. Factor name: Sense of identity
           Description: ...
           (low/high)
        """
        
        factors: List[Dict[str, Any]] = []
        current_factor: Dict[str, Any] = None  # {'name': str, 'description': str}
        current_values: List[str] = []
        
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            
            # Remove numbering/bullets
            clean_line = line.lstrip('1234567890.- ').strip()
            
            # New factor header (multi-line style): contains ':' but not parentheses and not a bullet
            if (':' in clean_line and not clean_line.startswith('*') and not clean_line.startswith('-') and '(' not in clean_line):
                # Save previous factor if complete
                if current_factor and len(current_values) >= 2:
                    factors.append({
                        'name': current_factor['name'],
                        'description': current_factor.get('description', ''),
                        'possible_values': current_values[:2]
                    })
                # Start a new factor
                try:
                    name_part, desc_part = clean_line.split(':', 1)
                    raw_left = name_part.strip()
                    left_clean = raw_left.replace('*', '').replace('#', '').strip().lower()
                    right_text = desc_part.strip()
                    # If generic left label, use right side as the name
                    if left_clean in ("factor name", "factor", "name") and right_text:
                        name_source = right_text
                        description = ""
                    else:
                        name_source = raw_left
                        description = right_text
                    factor_name = name_source.replace('*', '').replace('#', '').strip().lower().replace(' ', '_')
                    current_factor = {'name': factor_name, 'description': description}
                    current_values = []
                except Exception as e:
                    print(f"⚠️ Error parsing factor name line '{line}': {e}")
                    current_factor = None
                    current_values = []
                continue
            
            # Bullet value lines: e.g., '* High: ...', '- Low: ...', '* Value1: High (...)'
            if (clean_line.startswith('*') or clean_line.startswith('-')) and current_factor:
                try:
                    value_line = clean_line.lstrip('*- ').strip()
                    candidate = ""
                    if ':' in value_line:
                        left, right = value_line.split(':', 1)
                        left_norm = left.strip().lower().replace(' ', '_')
                        right_text = right.strip()
                        # If left label is a generic placeholder (value1/value2/etc),
                        # use the right-hand side as the candidate value; otherwise use the left.
                        if left_norm in {"value1", "value2", "value_1", "value_2", "value", "option1", "option_1", "option2", "option_2"}:
                            candidate = right_text
                        else:
                            candidate = left
                    else:
                        candidate = value_line

                    # Extract the value token from candidate by cutting at common separators
                    # like ' - ', '(', ',', ';' and taking the first token.
                    cut_points = []
                    for sep in [" - ", "(", ",", ";"]:
                        idx = candidate.find(sep)
                        if idx != -1:
                            cut_points.append(idx)
                    if cut_points:
                        candidate = candidate[:min(cut_points)]
                    value_name = candidate.strip().strip("-:()[]{}.,").lower().replace(' ', '_')
                    if value_name and value_name not in current_values:
                        current_values.append(value_name)
                except Exception as e:
                    print(f"⚠️ Error parsing value line '{line}': {e}")
                continue
            
            # Description override line
            if current_factor and clean_line.lower().startswith('description:'):
                try:
                    current_factor['description'] = clean_line.split(':', 1)[1].strip()
                except Exception:
                    pass
                continue
            
            # Parenthesized values on a separate line: "(value1/low, value2/high)"
            if current_factor and '(' in clean_line and ')' in clean_line and ':' not in clean_line:
                try:
                    values_start = clean_line.rfind('(')
                    values_end = clean_line.rfind(')')
                    values_text = clean_line[values_start+1:values_end]
                    vt = values_text.lower()
                    for token in ["value1", "value2", "value 1", "value 2", "value:", "value"]:
                        vt = vt.replace(token, "")
                    vt = vt.replace(",", "/").replace(" or ", "/")
                    for v in [t.strip().replace(' ', '_') for t in vt.split('/')]:
                        if v and v not in current_values:
                            current_values.append(v)
                except Exception as e:
                    print(f"⚠️ Error parsing parenthesized values line '{line}': {e}")
                continue
        
        # Save last factor
        if current_factor and len(current_values) >= 2:
            factors.append({
                'name': current_factor['name'],
                'description': current_factor.get('description', ''),
                'possible_values': current_values[:2]
            })
        
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
        for prefix in [
            "Here are",
            "The factors are",
            "Factors:",
            "Answer:",
            "Here is the answer:",
            "Here is the answer",
        ]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        # Handle sentinel if present (parse-only delimiter) and repeated blocks
        # Choose the segment that actually contains the clearest factor-like content.
        if "END_OF_FACTORS" in response:
            segments = [seg.strip() for seg in response.split("END_OF_FACTORS")]
            def looks_like_factors(text: str) -> int:
                if not text:
                    return 0
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                score = 0
                for l in lines[:10]:
                    # Numbered or bulleted
                    if l[:2].isdigit() or l.startswith(('-', '*')):
                        score += 1
                    # Factor-style pattern
                    if (":" in l) and ("(" in l) and (")" in l):
                        score += 2
                return score
            # Pick the segment with highest score; tie-breaker by length
            best = ""
            best_score = -1
            for seg in segments:
                sc = looks_like_factors(seg)
                if sc > best_score or (sc == best_score and len(seg) > len(best)):
                    best = seg
                    best_score = sc
            response = (best or response).strip()
        
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
        
        response = self.vllm_wrapper.generate(
            prompt,
            component="factor_generator",
            interaction_type="situation_analysis",
            max_tokens_override=512,
            stop=[],
        )
        
        print(f"🔍 Analysis response: '{response}'")
        
        # Parse the analysis response
        analysis = self._parse_analysis_response(response, factors)
        
        return analysis
    
    def _build_analysis_prompt(self, user_situation: str, factors: List[Dict[str, Any]]) -> str:
        """Build prompt for analyzing factor values in the situation."""
        
        factor_descriptions = []
        for factor in factors:
            try:
                pv = factor.get('possible_values') or []
                values = "/".join(pv)
                factor_descriptions.append(f"- {factor.get('name', 'unknown')}: {factor.get('description', '')} ({values})")
            except Exception as e:
                print(f"⚠️ Error building factor description for prompt: {e}")
        
        factors_text = "\n".join(factor_descriptions)
        
        prompt = f"""Situation: {user_situation}

Based on this situation, determine the value for each factor:

{factors_text}

For each factor, choose one of the possible values and briefly explain why.

Format:
factor_name: chosen_value - explanation
factor_name: chosen_value - explanation

Example:
stress_level: high - person shows clear signs of stress and worry
social_support: present - mentions having friends to talk to

Only output the lines in the specified format above. Do not add headings, prefaces, or trailing commentary.
After your last line, output a single line exactly: END_OF_ANALYSIS

Your analysis:"""
        
        return prompt
    
    def _parse_analysis_response(self, response: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Parse factor value analysis from LLM response."""
        
        analysis = {}
        
        if not response:
            return self._create_fallback_analysis("", factors)
        
        # Trim at analysis sentinel to drop trailing chatter
        cut = response.split("END_OF_ANALYSIS", 1)[0]
        lines = cut.strip().split('\n')
        factor_names = {f['name'] for f in factors}
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    factor_name = parts[0].strip().lower().replace(' ', '_')
                    value_part = parts[1].strip()
                    
                    # Extract just the value (before any explanation)
                    value = None
                    if ' - ' in value_part:
                        candidate = value_part.split(' - ')[0].strip()
                    else:
                        # Guard against empty strings to avoid IndexError
                        tokens = value_part.split()
                        if not tokens:
                            continue
                        candidate = tokens[0]
                    # Normalize and strip trailing punctuation
                    value = candidate.strip().strip("-:()[]{}.,").lower().replace(' ', '_')
                    
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
            name = factor.get('name', 'unknown')
            if name not in analysis:
                if name in fallback:
                    analysis[name] = fallback[name]
                else:
                    pv = factor.get('possible_values') or []
                    if pv:
                        analysis[name] = pv[0]
                    else:
                        print(f"⚠️ Factor '{name}' has no possible_values; defaulting to 'unknown'")
                        analysis[name] = 'unknown'
        
        return analysis
    
    def _create_fallback_analysis(self, user_situation: str, factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create fallback analysis when LLM analysis fails."""
        
        analysis = {}
        
        # Default to first value for each factor (if available), else 'unknown'
        for factor in factors:
            try:
                name = factor.get('name', 'unknown')
                pv = factor.get('possible_values') or []
                if pv:
                    analysis[name] = pv[0]
                else:
                    print(f"⚠️ Factor '{name}' has empty possible_values; using 'unknown' as fallback")
                    analysis[name] = 'unknown'
            except Exception as e:
                print(f"⚠️ Error creating fallback for factor: {e}")
                analysis[factor.get('name', 'unknown')] = 'unknown'
        
        return analysis
