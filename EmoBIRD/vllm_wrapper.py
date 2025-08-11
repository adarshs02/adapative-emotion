"""
vLLM Wrapper for Emobird

Provides a high-performance LLM interface using vLLM for inference.
"""

import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from logger import get_logger


class VLLMWrapper:
    """
    Wrapper class for vLLM inference engine.
    Provides a unified interface for LLM generation.
    """
    
    def __init__(self, config):
        """Initialize the vLLM wrapper."""
        self.config = config
        self.model = None
        self.sampling_params = None
        
        if config.use_vllm:
            self._initialize_vllm()
        else:
            raise ValueError("vLLM is not enabled in configuration")
    
    def _initialize_vllm(self):
        """Initialize the vLLM engine."""
        print(f"🚀 Initializing vLLM with model: {self.config.llm_model_name}")
        
        try:
            # Initialize vLLM engine
            self.model = LLM(
                model=self.config.llm_model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                trust_remote_code=True,
                enforce_eager=False,  # Use CUDA graphs for better performance
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None
            )
            
            # Configure JSON-specific sampling parameters to prevent repetitive generation
            self.json_sampling_params = SamplingParams(
                temperature=0.6,
                max_tokens=256,   # Shorter for JSON responses
                top_p=0.9,
                frequency_penalty=0.8,  # Prevent repetition
                presence_penalty=0.3,   # Encourage diversity
                stop=None  # Remove problematic stop tokens that truncate JSON
            )
            
            # Configure ultra-constrained sampling for abstracts to prevent hallucination
            self.abstract_sampling_params = SamplingParams(
                temperature=0.1,        # Very low temperature for deterministic output
                max_tokens=64,          # Short abstracts only
                top_p=0.7,              # More focused sampling
                frequency_penalty=0.0,  # No repetition penalty
                presence_penalty=0.0,   # No creativity penalty
                stop=None  # Remove problematic stop tokens that halt generation immediately
            )
            
            print("✅ vLLM initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize vLLM: {e}")
            raise
    
    def generate(self, prompt: str, component: str = "vllm", interaction_type: str = "generation") -> str:
        """
        Generate a single response for a prompt.
        
        Args:
            prompt: Input prompt string
            component: Component name for logging
            interaction_type: Type of interaction for logging
            
        Returns:
            Generated response string
        """
        # Use extended tokens for conversational output generation
        if component == "output_generator" and interaction_type == "conversational_response":
            return self.generate_conversational([prompt], component, interaction_type)[0]
        
        response = self.generate_batch([prompt], component, interaction_type)[0]
        return response
    
    def generate_conversational(self, prompts: List[str], component: str = "output_generator", interaction_type: str = "conversational_response") -> List[str]:
        """
        Generate conversational responses with extended token limits (up to 512 tokens).
        
        Args:
            prompts: List of input prompt strings
            component: Component name for logging
            interaction_type: Type of interaction for logging
            
        Returns:
            List of generated response strings
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Create extended sampling parameters for conversational responses
            conversational_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=220,  # Extended token limit for conversational responses
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None
            )
            
            # Generate responses
            outputs = self.model.generate(prompts, conversational_params)
            
            # Extract responses
            responses = []
            for output in outputs:
                response = output.outputs[0].text.strip()
                responses.append(response)
            
            # Log the interactions
            logger = get_logger()
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                logger.log_interaction(
                    component=component,
                    interaction_type=interaction_type,
                    prompt=prompt,
                    response=response,
                    metadata={
                        "batch_index": i,
                        "batch_size": len(prompts),
                        "sampling_method": "conversational_extended",
                        "temperature": conversational_params.temperature,
                        "max_tokens": conversational_params.max_tokens
                    }
                )
            
            return responses
            
        except Exception as e:
            # Log the error
            logger = get_logger()
            logger.log_error(
                component=component,
                error_type="conversational_generation_failed",
                error_message=str(e),
                context={"prompts_count": len(prompts), "interaction_type": interaction_type}
            )
            print(f"❌ Error during conversational generation: {e}")
            return [""] * len(prompts)  # Return empty strings as fallback
    
    def _generate_strict_json(self, prompt: str, component: str, interaction_type: str, use_temp_zero: bool = False) -> str:
        """
        Generate response with strict JSON parameters and stop tokens.
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Create strict JSON sampling params with balanced stop tokens
            # Use shorter max_tokens and smarter stop tokens to prevent rambling
            json_max_tokens = min(200, self.json_sampling_params.max_tokens)  # Limit for JSON responses
            
            if use_temp_zero:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=json_max_tokens,
                    top_p=0.95,
                    stop=["\n}\n", "```", "\n\nHuman:", "\n\nAssistant:", "\n---"],  # Stop after JSON completion or rambling starts
                    logprobs=None
                )
            else:
                sampling_params = SamplingParams(
                    temperature=0.1,  # Very low temperature for consistency
                    max_tokens=json_max_tokens,
                    top_p=0.95,
                    stop=["\n}\n", "```", "\n\nHuman:", "\n\nAssistant:", "\n---"],  # Stop after JSON completion or rambling starts
                    logprobs=None
                )
            
            # Generate response
            outputs = self.model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # Log the interaction
            logger = get_logger()
            logger.log_interaction(
                component=component,
                interaction_type=interaction_type,
                prompt=prompt,
                response=response,
                metadata={
                    "sampling_method": "strict_json",
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                    "stop_tokens": sampling_params.stop
                }
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Error during strict JSON generation: {e}")
            raise
    
    def _extract_first_json(self, response: str) -> str:
        """
        Extract the first valid JSON object from response, stripping everything after.
        """
        if not response:
            raise ValueError("Empty response")
        
        # Remove markdown code blocks if present
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Find first { and matching }
        start = response.find('{')
        if start == -1:
            raise ValueError("No JSON object found in response")
        
        # Find matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(response[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Unmatched braces in JSON")
        
        return response[start:end]
    
    def _validate_json_schema(self, parsed_json: dict, schema: dict) -> None:
        """
        Basic schema validation for required keys.
        """
        if "required" in schema:
            for key in schema["required"]:
                if key not in parsed_json:
                    raise ValueError(f"Missing required key: {key}")
        
        if "properties" in schema:
            for key, value in parsed_json.items():
                if key in schema["properties"]:
                    expected_type = schema["properties"][key].get("type")
                    if expected_type == "array" and not isinstance(value, list):
                        raise ValueError(f"Key '{key}' must be an array")
                    elif expected_type == "string" and not isinstance(value, str):
                        raise ValueError(f"Key '{key}' must be a string")
    
    def _generate_with_json_params(self, prompt: str, component: str, interaction_type: str) -> str:
        """
        Generate a single response using JSON-specific sampling parameters.
        
        Args:
            prompt: Input prompt string
            component: Component name for logging
            interaction_type: Type of interaction for logging
            
        Returns:
            Generated response string
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Generate response with JSON-specific parameters
            outputs = self.model.generate([prompt], self.json_sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            response = generated_text.strip()
            
            # Log the interaction
            logger = get_logger()
            logger.log_interaction(
                component=component,
                interaction_type=interaction_type,
                prompt=prompt,
                response=response,
                metadata={
                    "sampling_method": "json_specific",
                    "temperature": self.json_sampling_params.temperature,
                    "max_tokens": self.json_sampling_params.max_tokens,
                    "frequency_penalty": self.json_sampling_params.frequency_penalty,
                    "presence_penalty": self.json_sampling_params.presence_penalty,
                    "stop_tokens": self.json_sampling_params.stop
                }
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Error during JSON generation: {e}")
            return ""  # Return empty string as fallback
    
    def generate_abstract(self, prompt: str, component: str = "vllm", interaction_type: str = "abstract_generation") -> str:
        """
        Generate an abstract/summary using ultra-constrained sampling to prevent hallucination.
        
        Args:
            prompt: Input prompt string
            component: Component name for logging
            interaction_type: Type of interaction for logging
            
        Returns:
            Generated abstract string
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Generate response with abstract-specific constrained parameters
            outputs = self.model.generate([prompt], self.abstract_sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            response = generated_text.strip()
            
            # Log the interaction
            logger = get_logger()
            logger.log_interaction(
                component=component,
                interaction_type=interaction_type,
                prompt=prompt,
                response=response,
                metadata={
                    "sampling_method": "abstract_constrained",
                    "temperature": self.abstract_sampling_params.temperature,
                    "max_tokens": self.abstract_sampling_params.max_tokens,
                    "top_p": self.abstract_sampling_params.top_p,
                    "stop_tokens": self.abstract_sampling_params.stop
                }
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Error during abstract generation: {e}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            return ""  # Return empty string as fallback
    
    def generate_batch(self, prompts: List[str], component: str = "vllm", interaction_type: str = "batch_generation") -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompt strings
            component: Component name for logging
            interaction_type: Type of interaction for logging
            
        Returns:
            List of generated response strings
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Generate responses
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Extract generated text
            responses = []
            for output in outputs:
                generated_text = output.outputs[0].text
                responses.append(generated_text.strip())
            
            # Log each prompt-response pair
            logger = get_logger()
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                logger.log_interaction(
                    component=component,
                    interaction_type=f"{interaction_type}_{i+1}" if len(prompts) > 1 else interaction_type,
                    prompt=prompt,
                    response=response,
                    metadata={
                        "batch_size": len(prompts),
                        "batch_index": i,
                        "temperature": self.sampling_params.temperature,
                        "max_tokens": self.sampling_params.max_tokens
                    }
                )
            
            return responses
            
        except Exception as e:
            # Log the error
            logger = get_logger()
            logger.log_error(
                component=component,
                error_type="generation_failed",
                error_message=str(e),
                context={"prompts_count": len(prompts), "interaction_type": interaction_type}
            )
            print(f"❌ Error during generation: {e}")
            return [""] * len(prompts)  # Return empty strings as fallback
    
    def json_call(self, prompt: str, schema: dict = None, component: str = "vllm", interaction_type: str = "json_generation", max_retries: int = 2) -> Dict[str, Any]:
        """
        Enforce strict JSON generation with schema validation and retry logic.
        
        Args:
            prompt: Input prompt string
            schema: Optional JSON schema dict for validation
            component: Component name for logging
            interaction_type: Type of interaction for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON parsing fails after all retries
        """
        logger = get_logger()
        
        for attempt in range(max_retries + 1):
            try:
                # Use temperature=0 for retry attempts to get more deterministic output
                use_temp_zero = attempt > 0
                
                # Generate response with strict JSON parameters
                response = self._generate_strict_json(prompt, component, f"{interaction_type}_attempt_{attempt+1}", use_temp_zero)
                
                # Extract and parse JSON strictly
                cleaned_response = self._extract_first_json(response)
                parsed_json = json.loads(cleaned_response)
                
                # Validate against schema if provided
                if schema:
                    self._validate_json_schema(parsed_json, schema)
                
                # Log successful JSON parsing
                logger.log_interaction(
                    component=component,
                    interaction_type=f"{interaction_type}_success",
                    prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    response=json.dumps(parsed_json)[:200] + "..." if len(json.dumps(parsed_json)) > 200 else json.dumps(parsed_json),
                    metadata={
                        "attempt_number": attempt + 1,
                        "schema_validated": schema is not None,
                        "temperature_zero": use_temp_zero
                    }
                )
                
                return parsed_json
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Log parsing error
                logger.log_error(
                    component=component,
                    error_type="strict_json_parse_error",
                    error_message=str(e),
                    context={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "raw_response": response[:300] if 'response' in locals() else "No response",
                        "interaction_type": interaction_type
                    }
                )
                
                if attempt < max_retries:
                    print(f"⚠️ JSON parse error (attempt {attempt + 1}): {e}")
                    continue
                else:
                    # Final failure - raise error
                    error_msg = f"Failed to generate valid JSON after {max_retries + 1} attempts: {e}"
                    logger.log_error(
                        component=component,
                        error_type="strict_json_generation_failed",
                        error_message=error_msg,
                        context={
                            "final_response": response if 'response' in locals() else "No response",
                            "interaction_type": interaction_type,
                            "prompt": prompt[:200]
                        }
                    )
                    raise ValueError(error_msg)
        
        # Should never reach here, but just in case
        raise ValueError("Unexpected error in json_call")
    
    def generate_json(self, prompt: str, component: str = "vllm", interaction_type: str = "json_generation", max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a JSON response, with retry logic for malformed JSON.
        
        Args:
            prompt: Input prompt string
            component: Component name for logging
            interaction_type: Type of interaction for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON dictionary, or empty dict if parsing fails
        """
        logger = get_logger()
        
        for attempt in range(max_retries):
            # Use JSON-specific sampling parameters for better JSON generation
            response = self._generate_with_json_params(prompt, component, f"{interaction_type}_attempt_{attempt+1}")
            
            # DEBUG: Print raw response to see what we're getting
            print(f"🔍 Raw vLLM response length: {len(response)} chars")
            print(f"🔍 Raw vLLM response: {response[:1000]}")
            if len(response) > 1000:
                print(f"🔍 [TRUNCATED - showing first 1000 chars of {len(response)} total]")
            
            # Try to extract and parse JSON
            try:
                # Handle potential markdown code blocks
                cleaned_response = self._clean_json_response(response)
                parsed_json = json.loads(cleaned_response)
                
                # Log successful JSON parsing
                logger.log_interaction(
                    component=component,
                    interaction_type=f"{interaction_type}_success",
                    prompt=f"[JSON PARSING SUCCESS] Original prompt: {prompt[:100]}...",
                    response=f"[PARSED JSON] {json.dumps(parsed_json)[:200]}...",
                    metadata={
                        "attempt_number": attempt + 1,
                        "json_keys": list(parsed_json.keys()),
                        "cleaned_response_length": len(cleaned_response),
                        "original_response_length": len(response)
                    }
                )
                
                return parsed_json
                
            except json.JSONDecodeError as e:
                # Log JSON parsing error
                logger.log_error(
                    component=component,
                    error_type="json_parse_error",
                    error_message=str(e),
                    context={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "raw_response": response[:500],
                        "cleaned_response": self._clean_json_response(response)[:500],
                        "interaction_type": interaction_type
                    }
                )
                
                if attempt < max_retries - 1:
                    print(f"⚠️ JSON parse error (attempt {attempt + 1}): {e}")
                    print(f"Raw response: {response[:200]}...")
                    continue
                else:
                    print(f"❌ Failed to parse JSON after {max_retries} attempts")
                    print(f"Final response: {response}")
                    
                    # Log final failure
                    logger.log_error(
                        component=component,
                        error_type="json_generation_failed",
                        error_message=f"Failed to generate valid JSON after {max_retries} attempts",
                        context={
                            "final_response": response,
                            "interaction_type": interaction_type,
                            "prompt": prompt[:200]
                        }
                    )
                    
                    return {}
        
        return {}
    
    def _clean_json_response(self, response: str) -> str:
        """
        Robust JSON extraction from model responses.
        Handles various formats: markdown blocks, extra text, multiple JSONs.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        response = response.strip()
        
        # Method 1: Try to find JSON within markdown code blocks
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        json_blocks = re.findall(json_block_pattern, response)
        
        for block in json_blocks:
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                try:
                    # Test if it's valid JSON
                    json.loads(block)
                    return block
                except:
                    continue
        
        # Method 2: Find JSON objects in the response (handles extra text)
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        json_matches = re.findall(json_pattern, response)
        
        for match in json_matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        
        # Find complete JSON objects with balanced braces
        start_idx = response.find('{')
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = response[start_idx:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except:
                            continue
        
        # Last resort: empty JSON
        return "{}"
    
    def _reconstruct_partial_json(self, response: str) -> str:
        """
        Attempt to reconstruct a minimal valid JSON from partial content.
        
        Args:
            response: Raw model response
            
        Returns:
            Reconstructed JSON string
        """
        response = response.strip()
        
        # If response starts with a quote, assume it's partial JSON content
        if response.startswith('"'):
            # Add opening brace and try to close it
            response = '{' + response
            if not response.endswith('}'):
                response += '}'
        
        # Try to find the first opening brace
        start_idx = response.find('{')
        if start_idx == -1:
            return "{}"
        
        # Try to find the last closing brace
        end_idx = response.rfind('}')
        if end_idx == -1:
            # If no closing brace, try to add one
            response += '}'
            end_idx = len(response) - 1
        
        # Extract the content between the braces
        content = response[start_idx:end_idx+1]
        
        # Try to parse the content as JSON
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # If all else fails, return an empty JSON object
        return "{}"
    
    def update_sampling_params(self, **kwargs):
        """Update sampling parameters dynamically."""
        if self.sampling_params:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    setattr(self.sampling_params, key, value)
                    print(f"Updated {key} to {value}")
                else:
                    print(f"⚠️ Unknown sampling parameter: {key}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.config.llm_model_name,
            "max_model_len": self.config.vllm_max_model_len,
            "gpu_memory_utilization": self.config.vllm_gpu_memory_utilization,
            "tensor_parallel_size": self.config.vllm_tensor_parallel_size,
            "sampling_temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens
        }
    
    def __del__(self):
        """Cleanup when wrapper is destroyed."""
        if hasattr(self, 'model') and self.model:
            # vLLM handles cleanup automatically
            pass
