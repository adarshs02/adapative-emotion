"""
vLLM Wrapper for Emobird

Provides a high-performance LLM interface using vLLM for inference.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Type
from vllm import LLM, SamplingParams
try:
    from EmoBIRD.logger import get_logger
    from EmoBIRD.validation import (
        extract_first_json as _first_json,
        parse_and_validate as _parse_and_validate,
        strip_noise as _strip_noise,
    )
except ImportError:
    # Allow running scripts from the EmoBIRD/ directory without package context
    from logger import get_logger
    from validation import (
        extract_first_json as _first_json,
        parse_and_validate as _parse_and_validate,
        strip_noise as _strip_noise,
    )


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
        self.last_json_call_meta = {}
        
        if config.use_vllm:
            self._initialize_vllm()
        else:
            raise ValueError("vLLM is not enabled in configuration")
    
    def _initialize_vllm(self):
        """Initialize the vLLM engine."""
        print(f"🚀 Initializing vLLM with model: {self.config.llm_model_name}")
        
        try:
            # Respect optional custom download/cache directory to avoid shared-permission issues
            dl_dir = getattr(self.config, "vllm_download_dir", None)
            if dl_dir:
                # Ensure directory exists
                try:
                    os.makedirs(dl_dir, exist_ok=True)
                except Exception:
                    pass
                # Point Hugging Face caches to this directory as well
                # Force override to avoid inherited shared read-only paths
                os.environ["HF_HOME"] = dl_dir
                # A subdir for transformers cache to avoid clutter if desired
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(dl_dir, "transformers")
                # Also direct Hugging Face Hub cache to this directory
                os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(dl_dir, "hub")
                # Ensure subdirectories exist to prevent permission errors during lazy creation
                try:
                    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
                    os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
                except Exception:
                    pass

            # Initialize vLLM engine
            self.model = LLM(
                model=self.config.llm_model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                trust_remote_code=True,
                enforce_eager=False,  # Use CUDA graphs for better performance
                download_dir=dl_dir if dl_dir else None,
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=self.config.stop_seqs
            )
            
            # Configure JSON-specific sampling parameters
            self.json_sampling_params = SamplingParams(
                temperature=self.config.json_temperature,
                max_tokens=self.config.json_max_tokens,   # Shorter for JSON responses
                top_p=0.9,
                frequency_penalty=0.8,  # Prevent repetition
                presence_penalty=0.3,
                stop=self.config.stop_seqs
            )
            
            # Configure ultra-constrained sampling for abstracts to prevent hallucination
            self.abstract_sampling_params = SamplingParams(
                temperature=0.1,        # Very low temperature for deterministic output
                max_tokens=128,          # Short abstracts only
                top_p=0.7,              # More focused sampling
                frequency_penalty=0.0,  # No repetition penalty
                presence_penalty=0.0,   # No creativity penalty
                stop=self.config.stop_seqs
            )
            
            if dl_dir:
                print(f"✅ vLLM initialized successfully! Using download/cache dir: {dl_dir}")
                # Print resolved cache environment variables for debugging
                try:
                    print(
                        "Cache envs -> HF_HOME=", os.environ.get("HF_HOME"),
                        ", TRANSFORMERS_CACHE=", os.environ.get("TRANSFORMERS_CACHE"),
                        ", HUGGINGFACE_HUB_CACHE=", os.environ.get("HUGGINGFACE_HUB_CACHE")
                    )
                except Exception:
                    pass
            else:
                print("✅ vLLM initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize vLLM: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        component: str = "vllm",
        interaction_type: str = "generation",
        stop: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
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
        
        response = self.generate_batch(
            [prompt],
            component,
            interaction_type,
            stop=stop,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override,
        )[0]
        return response
    
    def generate_conversational(self, prompts: List[str], component: str = "output_generator", interaction_type: str = "conversational_response") -> List[str]:
        """
        Generate conversational responses with extended token limits (up to 1500 tokens).
        
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
                max_tokens=1500,  # Extended token limit for conversational responses
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=self.config.stop_seqs
            )
            
            # Generate responses
            outputs = self.model.generate(prompts, conversational_params)
            
            # Extract responses safely
            responses = []
            if not outputs or len(outputs) == 0:
                responses = [""] * len(prompts)
            else:
                for output in outputs:
                    try:
                        txt = (output.outputs[0].text if getattr(output, "outputs", None) else "")
                        responses.append((txt or "").strip())
                    except Exception:
                        responses.append("")
            
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
    
    def _generate_strict_json(
        self,
        prompt: str,
        component: str,
        interaction_type: str,
        use_temp_zero: bool = False,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        """
        Generate response with strict JSON parameters and stop tokens.
        """
        if not self.model:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            # Create strict JSON sampling params with balanced stop tokens
            # Use shorter max_tokens and smarter stop tokens to prevent rambling
            # Respect per-call override and cap
            effective_max = max_tokens_override if max_tokens_override is not None else self.json_sampling_params.max_tokens
            json_max_tokens = effective_max
            # Avoid stop tokens during JSON; rely on brace-aware extraction to capture the first complete object.
            # Any early stop (e.g., '}\n') can cut inside nested objects and corrupt JSON.
            json_stop = None
            
            if use_temp_zero:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=json_max_tokens,
                    top_p=0.95,
                    stop=json_stop,
                    logprobs=None
                )
            else:
                # Choose temperature: override > config.json_temperature
                eff_temp = self.config.json_temperature if temperature_override is None else temperature_override
                sampling_params = SamplingParams(
                    temperature=eff_temp,
                    max_tokens=json_max_tokens,
                    top_p=0.95,
                    stop=json_stop,
                    logprobs=None
                )
            
            # Generate response
            outputs = self.model.generate([prompt], sampling_params)
            if outputs and len(outputs) > 0 and getattr(outputs[0], "outputs", None):
                response = (outputs[0].outputs[0].text or "").strip()
            else:
                response = ""
            
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
    def _normalize_unified_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize unified JSON object in-place-safe manner.
        - Drop emotion_probs keys not in emotions; renormalize to 1.0.
        - If missing probs, back off to uniform over up to 3 emotions (or defaults).
        - Enforce allowed factor taxonomy from config if provided.
        - Coerce boolean strings to booleans for boolean-typed factors.
        - Rebuild factors list to match allowed definitions for present keys.
        """
        if not isinstance(data, dict):
            return data

        # Emotions and probs
        emotions = data.get("emotions") or []
        if not isinstance(emotions, list):
            emotions = []
        emotions = [e for e in emotions if isinstance(e, str) and e]
        data["emotions"] = emotions

        probs = data.get("emotion_probs") or {}
        if not isinstance(probs, dict):
            probs = {}

        # Keep only keys in emotions
        probs = {k: float(v) for k, v in probs.items() if k in emotions and isinstance(v, (int, float)) and v >= 0}

        def uniform_over(keys: list) -> Dict[str, float]:
            if not keys:
                return {}
            w = 1.0 / float(len(keys))
            return {k: w for k in keys}

        if not probs:
            # Back-off: choose up to 3 emotions if available, else defaults
            if emotions:
                chosen = emotions[:3]
            else:
                defaults = getattr(self.config, "default_emotions", []) or ["joy", "sadness", "anger"]
                chosen = defaults[:3]
                # also set emotions if empty
                if not emotions:
                    data["emotions"] = chosen
            probs = uniform_over(chosen)
        else:
            s = float(sum(probs.values()))
            if s <= 0:
                keys = list(probs.keys()) if probs else (emotions[:3] if emotions else [])
                probs = uniform_over(keys)
            elif abs(s - 1.0) > 1e-6:
                for k in list(probs.keys()):
                    probs[k] = probs[k] / s
        data["emotion_probs"] = probs

        # Factors normalization
        allowed = getattr(self.config, "allowed_factors", None)
        fv = data.get("factor_values") or {}
        if not isinstance(fv, dict):
            fv = {}

        if isinstance(allowed, dict) and allowed:
            normalized_fv: Dict[str, Any] = {}
            for name, val in fv.items():
                if name not in allowed:
                    continue  # reject unknown factor
                spec = allowed[name]
                vtype = spec.get("value_type")
                pvals = spec.get("possible_values")
                v = val
                if vtype == "boolean":
                    if isinstance(v, str):
                        lv = v.strip().lower()
                        if lv in ("true", "yes", "1"): v = True
                        elif lv in ("false", "no", "0"): v = False
                    v = bool(v) if isinstance(v, (bool, int, str)) else False
                if pvals is not None and len(pvals) > 0 and v not in pvals:
                    # Reject values not in allowed set
                    continue
                normalized_fv[name] = v
            fv = normalized_fv
            data["factor_values"] = fv

            # Rebuild factors list to match allowed set for present keys
            factors_list = []
            for name in fv.keys():
                spec = allowed[name]
                factors_list.append({
                    "name": name,
                    "value_type": spec.get("value_type"),
                    "possible_values": spec.get("possible_values"),
                    "description": spec.get("description", "allowed factor")
                })
            data["factors"] = factors_list

        return data
    
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

    def _clean_and_parse_json_two_pass(
        self,
        raw_text: str,
        schema: Optional[dict] = None,
        schema_model: Optional[Type] = None,
        context_meta: Optional[Dict[str, Any]] = None,
        component: str = "vllm",
        interaction_type: str = "json_generation",
    ) -> Optional[Dict[str, Any]]:
        """Two-pass cleaning + parsing routine.
        Pass 1: strict brace-aware first JSON.
        Pass 2: light noise stripping before first JSON.
        Returns parsed dict or None if both passes fail.
        """
        logger = get_logger()
        # Pass 1
        try:
            cleaned = _first_json(raw_text)
            data = json.loads(cleaned)
            data = self._normalize_unified_json(data)
            if schema_model is not None:
                _, _ = _parse_and_validate(json.dumps(data), schema_model)
            if schema:
                self._validate_json_schema(data, schema)
            return data
        except Exception as e1:
            logger.log_error(
                component=component,
                error_type="json_parse_first_pass_failed",
                error_message=str(e1),
                context={**(context_meta or {}), "stage": "first_pass"},
            )

        # Pass 2: noise strip then extract first JSON
        try:
            stripped = _strip_noise(raw_text)
            cleaned2 = _first_json(stripped)
            data2 = json.loads(cleaned2)
            data2 = self._normalize_unified_json(data2)
            if schema_model is not None:
                _, _ = _parse_and_validate(json.dumps(data2), schema_model)
            if schema:
                self._validate_json_schema(data2, schema)
            logger.log_interaction(
                component=component,
                interaction_type=f"{interaction_type}_success_after_clean",
                prompt=(context_meta or {}).get("prompt_preview"),
                response=json.dumps(data2)[:200] + ("..." if len(json.dumps(data2)) > 200 else ""),
                metadata={**(context_meta or {}), "stage": "second_pass"},
            )
            return data2
        except Exception as e2:
            logger.log_error(
                component=component,
                error_type="json_parse_second_pass_failed",
                error_message=str(e2),
                context={**(context_meta or {}), "stage": "second_pass"},
            )
            return None

    def _fallback_minimal_json(self, prompt: str, component: str, interaction_type: str) -> Dict[str, Any]:
        """Hard-gate fallback: produce minimal empathetic JSON using base LLM.
        Returns a consistent minimal structure without partial insights.
        """
        try:
            # Ask the model for one concise empathetic sentence only (deterministic)
            instruct = (
                "Write one concise, empathetic sentence acknowledging the user's feelings "
                "and offering brief support. No lists."
            )
            base_prompt = f"{instruct}\n\nContext:\n{prompt[:800]}"
            outputs = self.model.generate(
                [base_prompt],
                SamplingParams(temperature=0.0, max_tokens=64, top_p=0.9, stop=None),
            )
            text = outputs[0].outputs[0].text.strip()
            # Build minimal JSON
            data = {
                "emotions": ["neutral"],
                "emotion_probs": {"neutral": 1.0},
                "message": text[:500],
            }
        except Exception:
            # Absolute minimal fallback on catastrophic error
            data = {
                "emotions": ["neutral"],
                "emotion_probs": {"neutral": 1.0},
                "message": "I'm here with you. You're not alone in this.",
            }
        # Log fallback usage
        logger = get_logger()
        logger.log_interaction(
            component=component,
            interaction_type=f"{interaction_type}_fallback_minimal",
            prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
            response=json.dumps(data),
            metadata={"fallback": True},
        )
        return data
    
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
            
            # Extract generated text safely
            if outputs and len(outputs) > 0 and getattr(outputs[0], "outputs", None):
                generated_text = outputs[0].outputs[0].text
            else:
                generated_text = ""
            response = (generated_text or "").strip()
            
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
    
    def generate_batch(
        self,
        prompts: List[str],
        component: str = "vllm",
        interaction_type: str = "batch_generation",
        stop: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> List[str]:
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
            # Choose effective sampling params (allow per-call overrides)
            if stop is None and max_tokens_override is None and temperature_override is None:
                effective_params = self.sampling_params
            else:
                base = self.sampling_params
                effective_params = SamplingParams(
                    temperature=base.temperature if temperature_override is None else temperature_override,
                    max_tokens=base.max_tokens if max_tokens_override is None else max_tokens_override,
                    top_p=getattr(base, "top_p", 0.9),
                    frequency_penalty=getattr(base, "frequency_penalty", 0.0),
                    presence_penalty=getattr(base, "presence_penalty", 0.0),
                    stop=base.stop if stop is None else stop,
                )

            # Generate responses
            outputs = self.model.generate(prompts, effective_params)
            
            # Extract generated text safely
            responses = []
            if not outputs or len(outputs) == 0:
                responses = [""] * len(prompts)
            else:
                for output in outputs:
                    try:
                        generated_text = (output.outputs[0].text if getattr(output, "outputs", None) else "")
                        responses.append((generated_text or "").strip())
                    except Exception:
                        responses.append("")
            
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
                        "temperature": (effective_params.temperature if 'effective_params' in locals() else self.sampling_params.temperature),
                        "max_tokens": (effective_params.max_tokens if 'effective_params' in locals() else self.sampling_params.max_tokens),
                        "stop_tokens": (effective_params.stop if 'effective_params' in locals() else self.sampling_params.stop),
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
    
    def json_call(
        self,
        prompt: str,
        schema: dict = None,
        component: str = "vllm",
        interaction_type: str = "json_generation",
        max_retries: int = 0,
        schema_model: Optional[Type] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        meta = {"attempts": 0, "final_status": "unknown"}
        self.last_json_call_meta = meta

        # Step 1: single strict generation (honor override if provided)
        meta["attempts"] += 1
        response = self._generate_strict_json(
            prompt,
            component,
            f"{interaction_type}_attempt_1",
            use_temp_zero=(temperature_override is None),
            temperature_override=(0.0 if temperature_override is None else temperature_override),
            max_tokens_override=max_tokens_override,
        )

        # Optional single regenerate if empty or no JSON brace
        if (not response) or ("{" not in response):
            logger.log_error(
                component=component,
                error_type="empty_or_no_brace_response",
                error_message="Empty or no '{' found; one-time regenerate",
                context={"attempt": 1, "interaction_type": interaction_type}
            )
            meta["attempts"] += 1
            base_budget = max_tokens_override if max_tokens_override is not None else self.config.json_max_tokens
            bumped_budget = min(int(base_budget * 2), 2048)
            # Force JSON-only directive on regenerate
            force_json_prompt = (
                f"{prompt}\n\nReturn only a valid JSON object. "
                f"Do not include any explanations, comments, or code fences."
            )
            response = self._generate_strict_json(
                force_json_prompt,
                component,
                f"{interaction_type}_attempt_2_regenerate",
                use_temp_zero=(temperature_override is None),
                temperature_override=(0.0 if temperature_override is None else temperature_override),
                max_tokens_override=bumped_budget,
            )

        # Parse once, with two-pass cleaning and targeted unmatched-brace retry
        try:
            parsed = self._clean_and_parse_json_two_pass(
                response,
                schema=schema,
                schema_model=schema_model,
                context_meta={
                    "attempt": meta["attempts"],
                    "interaction_type": interaction_type,
                    "prompt_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
                },
                component=component,
                interaction_type=interaction_type,
            )
            if parsed is not None:
                logger.log_interaction(
                    component=component,
                    interaction_type=f"{interaction_type}_success",
                    prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                    response=json.dumps(parsed)[:200] + ("..." if len(json.dumps(parsed)) > 200 else ""),
                    metadata={"attempt_number": meta["attempts"]},
                )
                meta["final_status"] = "success"
                self.last_json_call_meta = meta
                return parsed
        except Exception as e:
            # If braces were unmatched or JSON was truncated, do a one-time higher-budget regenerate.
            err_str = str(e).lower()
            unmatched_like = ("unmatched braces" in err_str)
            truncated_like = ("unterminated" in err_str) or ("expecting" in err_str and "delimiter" in err_str)
            if unmatched_like or truncated_like:
                logger.log_error(
                    component=component,
                    error_type="unmatched_braces_retry",
                    error_message=str(e),
                    context={
                        "first_response": response if response else "No response",
                        "interaction_type": interaction_type,
                        "prompt": prompt[:200]
                    }
                )
                meta["attempts"] += 1
                base_budget = max_tokens_override if max_tokens_override is not None else self.config.json_max_tokens
                bumped_budget = min(int(base_budget * 2), 2048)
                # Keep same prompt but emphasize JSON-only
                force_json_prompt2 = (
                    f"{prompt}\n\nReturn only a valid JSON object. "
                    f"Do not include any explanations, comments, or code fences."
                )
                response2 = self._generate_strict_json(
                    force_json_prompt2,
                    component,
                    f"{interaction_type}_attempt_{meta['attempts']}_regenerate_unmatched",
                    use_temp_zero=(temperature_override is None),
                    temperature_override=(0.0 if temperature_override is None else temperature_override),
                    max_tokens_override=bumped_budget,
                )
                try:
                    parsed2 = self._clean_and_parse_json_two_pass(
                        response2,
                        schema=schema,
                        schema_model=schema_model,
                        context_meta={
                            "attempt": meta["attempts"],
                            "interaction_type": interaction_type,
                            "prompt_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
                        },
                        component=component,
                        interaction_type=interaction_type,
                    )
                    if parsed2 is not None:
                        logger.log_interaction(
                            component=component,
                            interaction_type=f"{interaction_type}_success_after_retry",
                            prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                            response=json.dumps(parsed2)[:200] + ("..." if len(json.dumps(parsed2)) > 200 else ""),
                            metadata={"attempt_number": meta["attempts"]},
                        )
                        meta["final_status"] = "success"
                        self.last_json_call_meta = meta
                        return parsed2
                except Exception as e2:
                    # Hard-gate fallback
                    logger.log_error(
                        component=component,
                        error_type="strict_json_generation_failed_after_retry",
                        error_message=str(e2),
                        context={
                            "first_response": response if response else "No response",
                            "second_response": response2 if response2 else "No response",
                            "interaction_type": interaction_type,
                            "prompt": prompt[:200],
                        },
                    )
                    meta["final_status"] = "fallback"
                    self.last_json_call_meta = meta
                    return self._fallback_minimal_json(prompt, component, interaction_type)
            # Non-brace-related failure: log and raise
            # Before giving up, attempt second-pass salvage on the first response
            salvaged = self._clean_and_parse_json_two_pass(
                response,
                schema=schema,
                schema_model=schema_model,
                context_meta={
                    "attempt": meta["attempts"],
                    "interaction_type": interaction_type,
                    "prompt_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
                },
                component=component,
                interaction_type=interaction_type,
            )
            if salvaged is not None:
                logger.log_interaction(
                    component=component,
                    interaction_type=f"{interaction_type}_success_after_clean",
                    prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                    response=json.dumps(salvaged)[:200] + ("..." if len(json.dumps(salvaged)) > 200 else ""),
                    metadata={"attempt_number": meta["attempts"]},
                )
                meta["final_status"] = "success"
                self.last_json_call_meta = meta
                return salvaged
            # Hard-gate fallback
            logger.log_error(
                component=component,
                error_type="strict_json_generation_failed",
                error_message=str(e),
                context={
                    "final_response": response if response else "No response",
                    "interaction_type": interaction_type,
                    "prompt": prompt[:200],
                },
            )
            meta["final_status"] = "fallback"
            self.last_json_call_meta = meta
            return self._fallback_minimal_json(prompt, component, interaction_type)
    
    def generate_json(self, prompt: str, component: str = "vllm", interaction_type: str = "json_generation", max_retries: int = 0) -> Dict[str, Any]:
        """
        Minimal JSON generation. Single strict call with optional one-time regenerate if empty.
        Returns parsed dict or {} on failure. Kept for backward compatibility.
        """
        logger = get_logger()
        # Single strict gen
        response = self._generate_strict_json(
            prompt,
            component,
            f"{interaction_type}_attempt_1",
            use_temp_zero=True,
            temperature_override=0.0,
        )
        if (not response) or ("{" not in response):
            # One-time regenerate with higher budget
            base_budget = self.config.json_max_tokens
            bumped_budget = min(int(base_budget * 2), 2048)
            response = self._generate_strict_json(
                prompt,
                component,
                f"{interaction_type}_attempt_2_regenerate",
                use_temp_zero=True,
                temperature_override=0.0,
                max_tokens_override=bumped_budget,
            )
        try:
            cleaned = _first_json(response)
            return json.loads(cleaned)
        except Exception as e:
            logger.log_error(
                component=component,
                error_type="json_generation_failed",
                error_message=str(e),
                context={"interaction_type": interaction_type, "prompt": prompt[:200]}
            )
            return {}
    
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
