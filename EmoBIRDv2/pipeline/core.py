#!/usr/bin/env python3
"""
EmoBIRDv2 Pipeline Core

Reusable orchestration of the multi-step EmoBIRDv2 pipeline with a clean API.

Exports:
- EmoBIRDConfig: configuration dataclass
- EmoBIRDPipeline: class with step_* methods and run_for_text()
- run_pipeline_for_text(): functional wrapper

This mirrors the logic in `EmoBIRDv2/eval_scripts/run_emopatient_emobirdv2.py` while
keeping it importable from other scripts or notebooks.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure repo root is importable when this file is executed directly
import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# EmoBIRDv2 imports
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    ABSTRACT_MAX_TOKENS,
    FACTOR_MAX_TOKENS,
    EMOTION_MAX_TOKENS,
    LIKERT_MAX_TOKENS,
    OUTPUT_MAX_TOKENS,
)
from EmoBIRDv2.utils.utils import robust_json_loads
import EmoBIRDv2.scripts.abstract_generator as AG
import EmoBIRDv2.scripts.factor_generator as FG
import EmoBIRDv2.scripts.factor_value_selector as FVS
import EmoBIRDv2.scripts.emotion_generator as EG
import EmoBIRDv2.scripts.likert_matcher as LM
import EmoBIRDv2.scripts.final_output_generator as FOG
import EmoBIRDv2.scripts.fallback_responder as FBR


@dataclass
class EmoBIRDConfig:
    model: str = MODEL_NAME
    temperature: float = MODEL_TEMPERATURE

    abs_max_tokens: int = ABSTRACT_MAX_TOKENS
    fac_max_tokens: int = FACTOR_MAX_TOKENS
    sel_max_tokens: int = FACTOR_MAX_TOKENS
    emo_max_tokens: int = EMOTION_MAX_TOKENS
    likert_max_tokens: int = LIKERT_MAX_TOKENS
    out_max_tokens: int = OUTPUT_MAX_TOKENS

    attempts: int = 5
    with_emotions: bool = True
    log_raw: bool = False

    # Optional overrides
    api_key: Optional[str] = field(default_factory=lambda: OPENROUTER_API_KEY or "")
    openrouter_connect_timeout: Optional[int] = None
    openrouter_read_timeout: Optional[int] = None

    # Fallback behavior
    fallback_enabled: bool = True
    fallback_model: Optional[str] = None  # default to primary model when None
    fallback_temperature: Optional[float] = 0.6
    fallback_max_tokens: int = OUTPUT_MAX_TOKENS


class EmoBIRDPipeline:
    def __init__(self, config: Optional[EmoBIRDConfig] = None) -> None:
        self.config = config or EmoBIRDConfig()
        # Apply timeout overrides by rebinding the module-level values used by AG.call_openrouter
        if self.config.openrouter_connect_timeout is not None:
            try:
                AG.OPENROUTER_CONNECT_TIMEOUT = int(self.config.openrouter_connect_timeout)  # type: ignore[attr-defined]
            except Exception:
                pass
        if self.config.openrouter_read_timeout is not None:
            try:
                AG.OPENROUTER_READ_TIMEOUT = int(self.config.openrouter_read_timeout)  # type: ignore[attr-defined]
            except Exception:
                pass

    # ------------------------ Steps ------------------------
    def step_abstract(self, *, situation: str) -> Optional[Dict[str, Any]]:
        tpl = AG.load_prompt()
        prompt = AG.build_user_prompt(tpl, situation)
        raw = ""
        for i in range(1, int(self.config.attempts) + 1):
            try:
                raw = AG.call_openrouter(
                    prompt,
                    self.config.api_key or "",
                    self.config.model,
                    float(self.config.temperature),
                    int(self.config.abs_max_tokens),
                )
                if self.config.log_raw and raw:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"[abstract] raw: {trunc}", file=sys.stderr)
            except Exception as e:
                print(f"[abstract] attempt {i}/{self.config.attempts} failed: {e}", file=sys.stderr)
                raw = ""
            if not raw:
                continue
            try:
                obj = robust_json_loads(raw)
                if isinstance(obj, dict) and obj.get("abstract"):
                    return obj
            except Exception as e:
                print(
                    f"[abstract] attempt {i}/{self.config.attempts} JSON parse failed: {e}",
                    file=sys.stderr,
                )
        return None

    def step_factors(self, *, abstract_text: str) -> Optional[List[Dict[str, Any]]]:
        tpl = FG.load_prompt()
        prompt = FG.build_user_prompt(tpl, abstract_text)
        raw = ""
        for i in range(1, int(self.config.attempts) + 1):
            try:
                raw = AG.call_openrouter(
                    prompt,
                    self.config.api_key or "",
                    self.config.model,
                    float(self.config.temperature),
                    int(self.config.fac_max_tokens),
                )
                if self.config.log_raw and raw:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"[factors] raw: {trunc}", file=sys.stderr)
            except Exception as e:
                print(f"[factors] attempt {i}/{self.config.attempts} failed: {e}", file=sys.stderr)
                raw = ""
            if not raw:
                continue
            parsed = FG.parse_factor_block(raw)
            if parsed:
                return parsed
        return None

    def step_select(self, *, situation: str, factors: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
        tpl = FVS.load_prompt()
        prompt = FVS.build_user_prompt(tpl, situation, factors)
        raw = ""
        for i in range(1, int(self.config.attempts) + 1):
            try:
                raw = AG.call_openrouter(
                    prompt,
                    self.config.api_key or "",
                    self.config.model,
                    float(self.config.temperature),
                    int(self.config.sel_max_tokens),
                )
                if self.config.log_raw and raw:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"[select] raw: {trunc}", file=sys.stderr)
            except Exception as e:
                print(f"[select] attempt {i}/{self.config.attempts} failed: {e}", file=sys.stderr)
                raw = ""
            if not raw:
                continue
            parsed = FVS.parse_selection_block(raw)
            if parsed:
                return parsed
        return None

    def step_emotions(self, *, situation: str) -> Optional[List[str]]:
        tpl = EG.load_prompt()
        prompt = EG.build_user_prompt(tpl, situation)
        raw = ""
        for i in range(1, int(self.config.attempts) + 1):
            try:
                raw = AG.call_openrouter(
                    prompt,
                    self.config.api_key or "",
                    self.config.model,
                    float(self.config.temperature),
                    int(self.config.emo_max_tokens),
                )
                if self.config.log_raw and raw:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"[emotions] raw: {trunc}", file=sys.stderr)
            except Exception as e:
                print(f"[emotions] attempt {i}/{self.config.attempts} failed: {e}", file=sys.stderr)
                raw = ""
            if not raw:
                continue
            parsed = EG.parse_emotion_lines(raw)
            if parsed and len(parsed) >= 3:
                return parsed
        return None

    def step_likert(self, *, situation: str, factors: List[Dict[str, Any]], emotions: List[str]) -> Optional[List[Dict[str, Any]]]:
        tpl = LM.load_prompt()
        prompt = LM.build_user_prompt(tpl, situation, factors, emotions)
        raw = ""
        for i in range(1, int(self.config.attempts) + 1):
            try:
                raw = AG.call_openrouter(
                    prompt,
                    self.config.api_key or "",
                    self.config.model,
                    float(self.config.temperature),
                    int(self.config.likert_max_tokens),
                )
                if self.config.log_raw and raw:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"[likert] raw: {trunc}", file=sys.stderr)
            except Exception as e:
                print(f"[likert] attempt {i}/{self.config.attempts} failed: {e}", file=sys.stderr)
                raw = ""
            if not raw:
                continue
            parsed = LM.parse_likert_lines(raw)
            if parsed:
                return parsed
        return None

    def step_final_output(
        self,
        *,
        situation: str,
        abstract: Optional[str],
        selections: List[Dict[str, str]],
        likert_items: List[Dict[str, Any]],
    ) -> Optional[str]:
        try:
            return FOG.generate_final_output(
                situation=situation,
                abstract=abstract,
                selections=selections,
                likert_items=likert_items,
                model=self.config.model,
                temperature=float(self.config.temperature),
                max_tokens=int(self.config.out_max_tokens),
            )
        except Exception as e:
            print(f"[final_output] failed: {e}", file=sys.stderr)
            return None

    def step_fallback(
        self,
        *,
        situation: str,
        failed_at: str,
    ) -> str:
        """Best-effort empathetic response when a stage fails.

        Returns a non-empty string. Uses a static last-resort message if the model call fails.
        """
        static_msg = (
            "I'm sorry you're going through this. Based on what you shared, it's understandable "
            "to feel this way. If you can, take a slow breath and focus on one small, supportive "
            "step for yourself right now. If safety is a concern, please reach out to someone you "
            "trust or local support services."
        )
        try:
            text = FBR.generate_fallback(
                situation=situation,
                failed_at=failed_at,
                model=self.config.fallback_model or self.config.model,
                temperature=(self.config.fallback_temperature if self.config.fallback_temperature is not None else self.config.temperature),
                max_tokens=int(self.config.fallback_max_tokens),
                api_key=self.config.api_key or "",
            )
            return text.strip() if text and text.strip() else static_msg
        except Exception as e:
            print(f"[fallback] failed: {e}", file=sys.stderr)
            return static_msg

    # ------------------------ Orchestration ------------------------
    def run_for_text(self, *, situation: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "abstract": None,
            "factors": None,
            "selections": None,
            "emotions": None,
            "likert": None,
            "final_output": None,
            "status": "ok",
            "failed_at": None,
            "fallback_used": False,
            "fallback_reason": None,
            "fallback_model": None,
        }

        abs_obj = self.step_abstract(situation=situation)

        # Abstract fallback to ensure downstream steps can proceed
        abstract: Optional[str]
        if not abs_obj:
            abstract = situation
            result["abstract_fallback"] = True
        else:
            abstract = str(abs_obj.get("abstract", "")).strip()
        result["abstract"] = abstract

        factors = self.step_factors(abstract_text=abstract)
        if not factors:
            if self.config.fallback_enabled:
                fb = self.step_fallback(situation=situation, failed_at="factors")
                result.update({
                    "status": "fallback",
                    "failed_at": "factors",
                    "final_output": fb,
                    "fallback_used": True,
                    "fallback_reason": "stage_failed",
                    "fallback_model": self.config.fallback_model or self.config.model,
                })
                return result
            result["status"] = "error"
            result["failed_at"] = "factors"
            return result
        result["factors"] = factors

        selections = self.step_select(situation=situation, factors=factors)
        if not selections:
            if self.config.fallback_enabled:
                fb = self.step_fallback(
                    situation=situation,
                    failed_at="select",
                )
                result.update({
                    "status": "fallback",
                    "failed_at": "select",
                    "final_output": fb,
                    "fallback_used": True,
                    "fallback_reason": "stage_failed",
                    "fallback_model": self.config.fallback_model or self.config.model,
                })
                return result
            result["status"] = "error"
            result["failed_at"] = "select"
            return result
        result["selections"] = selections

        # Emotions are required
        emotions: Optional[List[str]] = self.step_emotions(situation=situation)
        if not emotions:
            if self.config.fallback_enabled:
                fb = self.step_fallback(
                    situation=situation,
                    failed_at="emotions",
                )
                result.update({
                    "status": "fallback",
                    "failed_at": "emotions",
                    "final_output": fb,
                    "fallback_used": True,
                    "fallback_reason": "stage_failed",
                    "fallback_model": self.config.fallback_model or self.config.model,
                })
                return result
            result["status"] = "error"
            result["failed_at"] = "emotions"
            return result
        result["emotions"] = emotions

        # Likert is required
        likert: Optional[List[Dict[str, Any]]] = self.step_likert(
            situation=situation, factors=factors, emotions=emotions
        )
        if not likert:
            if self.config.fallback_enabled:
                fb = self.step_fallback(
                    situation=situation,
                    failed_at="likert",
                )
                result.update({
                    "status": "fallback",
                    "failed_at": "likert",
                    "final_output": fb,
                    "fallback_used": True,
                    "fallback_reason": "stage_failed",
                    "fallback_model": self.config.fallback_model or self.config.model,
                })
                return result
            result["status"] = "error"
            result["failed_at"] = "likert"
            return result
        result["likert"] = likert

        # Generate final output (likert is guaranteed above)
        final_text = self.step_final_output(
            situation=situation,
            abstract=abstract,
            selections=selections,
            likert_items=likert,
        )
        if final_text:
            result["final_output"] = final_text
            return result
        # If final generation failed, attempt fallback if enabled
        if self.config.fallback_enabled:
            fb = self.step_fallback(
                situation=situation,
                failed_at="final_output",
            )
            result.update({
                "status": "fallback",
                "failed_at": "final_output",
                "final_output": fb,
                "fallback_used": True,
                "fallback_reason": "stage_failed",
                "fallback_model": self.config.fallback_model or self.config.model,
            })
            return result

        return result


def run_pipeline_for_text(
    *,
    situation: str,
    config: Optional[EmoBIRDConfig] = None,
) -> Dict[str, Any]:
    """
    Functional wrapper that constructs an `EmoBIRDPipeline` and runs it for a single text.
    """
    pipe = EmoBIRDPipeline(config=config)
    return pipe.run_for_text(situation=situation)
