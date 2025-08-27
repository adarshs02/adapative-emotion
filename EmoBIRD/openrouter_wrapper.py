"""
OpenRouter Wrapper for Emobird

Provides an HTTP-based LLM interface against OpenRouter's Chat Completions API.
Matches the VLLMWrapper interface: generate(), generate_batch(), generate_conversational(),
json_call(), and generate_abstract().
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type

import requests

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


class OpenRouterWrapper:
    """
    Wrapper class for OpenRouter chat-completions API.
    Provides a unified interface for LLM generation consistent with VLLMWrapper.
    """

    def __init__(self, config):
        self.config = config
        self.base_url: str = getattr(config, "openrouter_base_url", "https://openrouter.ai/api/v1/chat/completions")
        self.api_key: Optional[str] = getattr(config, "openrouter_api_key", None)
        self.provider: Optional[str] = getattr(config, "openrouter_provider", None)
        self.timeout: int = int(getattr(config, "openrouter_timeout", 60))
        self.max_retries: int = int(getattr(config, "openrouter_max_retries", 1))
        self.last_json_call_meta: Dict[str, Any] = {}

        if not self.api_key:
            print("⚠️ OpenRouter API key is not set. Set OPENROUTER_API_KEY or config.openrouter_api_key.")

        # Basic session for connection reuse
        self._session = requests.Session()

    # -------------- Public API --------------
    def generate(
        self,
        prompt: str,
        component: str = "openrouter",
        interaction_type: str = "generation",
        stop: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
        """
        Generate a single response for a prompt.
        Mirrors VLLMWrapper.generate behavior including conversational routing.
        """
        # Route extended conversational responses to dedicated method
        if component == "output_generator" and interaction_type == "conversational_response":
            return self.generate_conversational([prompt], component, interaction_type)[0]

        return self.generate_batch(
            [prompt],
            component,
            interaction_type,
            stop=stop,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override,
        )[0]

    def generate_conversational(
        self,
        prompts: List[str],
        component: str = "output_generator",
        interaction_type: str = "conversational_response",
    ) -> List[str]:
        """
        Generate conversational responses with extended token limits (~1500 tokens).
        """
        responses: List[str] = []
        logger = get_logger()
        for i, p in enumerate(prompts):
            resp_text = self._complete(
                prompt=p,
                temperature=self.config.temperature,
                max_tokens=1500,
                stop=self.config.stop_seqs,
                metadata={
                    "component": component,
                    "interaction_type": interaction_type,
                    "batch_index": i,
                    "batch_size": len(prompts),
                    "sampling_method": "conversational_extended",
                },
            )
            responses.append(resp_text)
            logger.log_interaction(
                component=component,
                interaction_type=interaction_type,
                prompt=p,
                response=resp_text,
                metadata={
                    "batch_index": i,
                    "batch_size": len(prompts),
                    "temperature": self.config.temperature,
                    "max_tokens": 1500,
                    "stop_tokens": self.config.stop_seqs,
                },
            )
        return responses

    def generate_batch(
        self,
        prompts: List[str],
        component: str = "openrouter",
        interaction_type: str = "batch_generation",
        stop: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts against OpenRouter.
        """
        responses: List[str] = []
        logger = get_logger()
        eff_stop = self.config.stop_seqs if stop is None else stop
        eff_temp = self.config.temperature if temperature_override is None else temperature_override
        eff_max_tokens = self.config.max_new_tokens if max_tokens_override is None else max_tokens_override

        for i, p in enumerate(prompts):
            try:
                text = self._complete(
                    prompt=p,
                    temperature=eff_temp,
                    max_tokens=eff_max_tokens,
                    stop=eff_stop,
                    metadata={
                        "component": component,
                        "interaction_type": interaction_type,
                        "batch_index": i,
                        "batch_size": len(prompts),
                    },
                )
            except Exception as e:
                text = ""
                # Log error
                logger.log_error(
                    component=component,
                    error_type="generation_failed",
                    error_message=str(e),
                    context={"prompts_count": len(prompts), "interaction_type": interaction_type},
                )
            responses.append((text or "").strip())
            logger.log_interaction(
                component=component,
                interaction_type=f"{interaction_type}_{i+1}" if len(prompts) > 1 else interaction_type,
                prompt=p,
                response=responses[-1],
                metadata={
                    "batch_size": len(prompts),
                    "batch_index": i,
                    "temperature": eff_temp,
                    "max_tokens": eff_max_tokens,
                    "stop_tokens": eff_stop,
                },
            )
        return responses

    def json_call(
        self,
        prompt: str,
        schema: dict = None,
        component: str = "openrouter",
        interaction_type: str = "json_generation",
        max_retries: int = 0,
        schema_model: Optional[Type] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Enforce strict JSON generation with schema validation and retry logic.
        Mirrors VLLMWrapper.json_call behavior using OpenRouter.
        """
        logger = get_logger()
        meta = {"attempts": 0, "final_status": "unknown"}
        self.last_json_call_meta = meta

        # Step 1: single strict generation (temp zero by default unless override)
        meta["attempts"] += 1
        response = self._complete(
            prompt=prompt,
            temperature=(0.0 if temperature_override is None else temperature_override),
            max_tokens=(self.config.json_max_tokens if max_tokens_override is None else max_tokens_override),
            stop=None,  # avoid cutting JSON early
            force_json_only=False,  # prompt should already instruct JSON-only
            metadata={"component": component, "interaction_type": f"{interaction_type}_attempt_1"},
        )

        # Optional single regenerate if empty or no JSON brace
        if (not response) or ("{" not in response):
            logger.log_error(
                component=component,
                error_type="empty_or_no_brace_response",
                error_message="Empty or no '{' found; one-time regenerate",
                context={"attempt": 1, "interaction_type": interaction_type},
            )
            meta["attempts"] += 1
            base_budget = self.config.json_max_tokens if max_tokens_override is None else max_tokens_override
            bumped_budget = min(int(base_budget * 2), 2048)
            force_json_prompt = (
                f"{prompt}\n\nReturn only a valid JSON object. Do not include any explanations, comments, or code fences."
            )
            response = self._complete(
                prompt=force_json_prompt,
                temperature=(0.0 if temperature_override is None else temperature_override),
                max_tokens=bumped_budget,
                stop=None,
                force_json_only=False,
                metadata={"component": component, "interaction_type": f"{interaction_type}_attempt_2_regenerate"},
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
                        "prompt": prompt[:200],
                    },
                )
                meta["attempts"] += 1
                base_budget = self.config.json_max_tokens if max_tokens_override is None else max_tokens_override
                bumped_budget = min(int(base_budget * 2), 2048)
                force_json_prompt2 = (
                    f"{prompt}\n\nReturn only a valid JSON object. Do not include any explanations, comments, or code fences."
                )
                response2 = self._complete(
                    prompt=force_json_prompt2,
                    temperature=(0.0 if temperature_override is None else temperature_override),
                    max_tokens=bumped_budget,
                    stop=None,
                    force_json_only=False,
                    metadata={
                        "component": component,
                        "interaction_type": f"{interaction_type}_attempt_{meta['attempts']}_regenerate_unmatched",
                    },
                )
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
                        interaction_type=f"{interaction_type}_success_after_regen",
                        prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                        response=json.dumps(parsed2)[:200] + ("..." if len(json.dumps(parsed2)) > 200 else ""),
                        metadata={"attempt_number": meta["attempts"]},
                    )
                    meta["final_status"] = "success"
                    self.last_json_call_meta = meta
                    return parsed2

        # Optional final reformat-only attempt(s)
        retries_left = max(0, int(max_retries)) if max_retries is not None else 0
        allow_format_retry = int(getattr(self.config, "allow_format_only_retry", 1))
        retries_left = max(retries_left, allow_format_retry)
        while retries_left > 0:
            retries_left -= 1
            meta["attempts"] += 1
            try:
                reform_prompt = (
                    "You will be given content that approximately resembles JSON. "
                    "Rewrite it into valid JSON only (one JSON object). Do not add commentary or code fences.\n\n"
                    f"CONTENT:\n{response[:1200]}\n\nJSON ONLY:"
                )
                reform = self._complete(
                    prompt=reform_prompt,
                    temperature=0.0,
                    max_tokens=min(1024, (self.config.json_max_tokens if max_tokens_override is None else max_tokens_override) * 2),
                    stop=None,
                    force_json_only=False,
                    metadata={"component": component, "interaction_type": f"{interaction_type}_format_only_attempt"},
                )
                parsed3 = self._clean_and_parse_json_two_pass(
                    reform,
                    schema=schema,
                    schema_model=schema_model,
                    context_meta={"attempt": meta["attempts"], "interaction_type": interaction_type},
                    component=component,
                    interaction_type=interaction_type,
                )
                if parsed3 is not None:
                    logger.log_interaction(
                        component=component,
                        interaction_type=f"{interaction_type}_success_after_format_retry",
                        prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                        response=json.dumps(parsed3)[:200] + ("..." if len(json.dumps(parsed3)) > 200 else ""),
                        metadata={"attempt_number": meta["attempts"]},
                    )
                    meta["final_status"] = "success"
                    self.last_json_call_meta = meta
                    return parsed3
            except Exception as _:
                pass

        # Hard-gate fallback: produce minimal empathetic JSON
        data = self._fallback_minimal_json(prompt, component, interaction_type)
        meta["final_status"] = "fallback"
        self.last_json_call_meta = meta
        return data

    def generate_abstract(
        self,
        prompt: str,
        component: str = "openrouter",
        interaction_type: str = "abstract_generation",
    ) -> str:
        """
        Generate an abstract/summary using constrained sampling to prevent hallucination.
        """
        logger = get_logger()
        try:
            resp_text = self._complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=128,
                stop=self.config.stop_seqs,
                metadata={"component": component, "interaction_type": interaction_type},
            )
            logger.log_interaction(
                component=component,
                interaction_type=interaction_type,
                prompt=prompt,
                response=resp_text,
                metadata={
                    "sampling_method": "abstract_constrained",
                    "temperature": 0.1,
                    "max_tokens": 128,
                    "stop_tokens": self.config.stop_seqs,
                },
            )
            return (resp_text or "").strip()
        except Exception as e:
            logger.log_error(
                component=component,
                error_type="abstract_generation_failed",
                error_message=str(e),
                context={"interaction_type": interaction_type},
            )
            return ""

    # -------------- Internal helpers --------------
    def _complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        metadata: Optional[Dict[str, Any]] = None,
        force_json_only: bool = False,
    ) -> str:
        """
        Call OpenRouter chat completions with retries and return text content.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
        # Optional provider routing hint
        if self.provider:
            headers["HTTP-Referer"] = self.provider  # OpenRouter sometimes uses Referer/X-Title; keep minimal

        payload: Dict[str, Any] = {
            "model": self.config.llm_model_name,
            "messages": [
                {"role": "system", "content": ("Return only JSON." if force_json_only else "You are a helpful assistant.")},
                {"role": "user", "content": prompt},
            ],
            "temperature": max(0.0, float(temperature)),
            "max_tokens": int(max_tokens),
            "top_p": 0.9,
        }
        if stop:
            payload["stop"] = stop

        attempt = 0
        backoff = 1.0
        last_err: Optional[Exception] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                resp = self._session.post(
                    self.base_url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except Exception:
                        content = ""
                    return (content or "").strip()
                # Retry on 429/5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue
                # Non-retryable HTTP error
                resp.raise_for_status()
            except Exception as e:
                last_err = e
                # Retry on network/timeout
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)
                continue
        # Exhausted retries
        raise RuntimeError(f"OpenRouter completion failed after {attempt} attempts: {last_err}")

    def _clean_and_parse_json_two_pass(
        self,
        raw_text: str,
        schema: Optional[dict] = None,
        schema_model: Optional[Type] = None,
        context_meta: Optional[Dict[str, Any]] = None,
        component: str = "openrouter",
        interaction_type: str = "json_generation",
    ) -> Optional[Dict[str, Any]]:
        """
        Two-pass cleaning + parsing routine (shared with vLLM wrapper semantics).
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
                context={(context_meta or {}) | {"stage": "first_pass"}},
            )

        # Pass 2
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
                metadata={(context_meta or {}) | {"stage": "second_pass"}},
            )
            return data2
        except Exception as e2:
            logger.log_error(
                component=component,
                error_type="json_parse_second_pass_failed",
                error_message=str(e2),
                context={(context_meta or {}) | {"stage": "second_pass"}},
            )
            return None

    def _fallback_minimal_json(self, prompt: str, component: str, interaction_type: str) -> Dict[str, Any]:
        """
        Hard-gate fallback: produce minimal empathetic JSON using base LLM.
        Returns a consistent minimal structure without partial insights.
        """
        try:
            instruct = (
                "Write one concise, empathetic sentence acknowledging the user's feelings "
                "and offering brief support. No lists."
            )
            base_prompt = f"{instruct}\n\nContext:\n{prompt[:800]}"
            text = self._complete(
                prompt=base_prompt,
                temperature=0.0,
                max_tokens=64,
                stop=None,
                metadata={"component": component, "interaction_type": f"{interaction_type}_fallback_minimal"},
            )
            data = {
                "emotions": ["neutral"],
                "emotion_probs": {"neutral": 1.0},
                "message": (text or "").strip()[:500],
            }
        except Exception:
            data = {
                "emotions": ["neutral"],
                "emotion_probs": {"neutral": 1.0},
                "message": "I'm here with you. You're not alone in this.",
            }
        logger = get_logger()
        logger.log_interaction(
            component=component,
            interaction_type=f"{interaction_type}_fallback_minimal",
            prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
            response=json.dumps(data),
            metadata={"fallback": True},
        )
        return data

    def _normalize_unified_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize unified JSON object in-place-safe manner (copy of VLLM behavior).
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
                    continue
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
                    "description": spec.get("description", "allowed factor"),
                })
            data["factors"] = factors_list

        return data

    def _validate_json_schema(self, parsed_json: dict, schema: dict) -> None:
        """
        Basic schema validation for required keys.
        """
        if not schema:
            return
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
