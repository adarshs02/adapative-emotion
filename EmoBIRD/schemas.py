"""
Pydantic schemas for strict output validation.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


ValueType = Literal["boolean", "categorical", "ordinal"]


class FactorDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value_type: ValueType
    possible_values: Optional[List[Union[str, int, float]]] = None
    description: str

    @field_validator("possible_values")
    @classmethod
    def _non_empty_if_provided(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("possible_values, if provided, must be non-empty")
        return v


class UnifiedEmotionAnalysis(BaseModel):
    """Unified output object combining factors, factor values, and emotions.

    All fields are strictly validated. Extra keys are forbidden.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    situation: str
    scenario_summary: str = Field(min_length=1, max_length=400)

    factors: List[FactorDef] = Field(default_factory=list)
    factor_values: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    emotions: List[str] = Field(default_factory=list)
    emotion_probs: Dict[str, float] = Field(default_factory=dict)

    version: Literal["uea_v1"] = "uea_v1"

    @model_validator(mode="after")
    def _cross_validate(self):
        # Validate factor_values keys exist in factors
        factor_names = {f.name for f in self.factors}
        for k in self.factor_values.keys():
            if k not in factor_names:
                raise ValueError(f"factor_values contains unknown factor '{k}'")

        # Validate factor value types against factor definitions
        factor_index = {f.name: f for f in self.factors}
        for k, v in self.factor_values.items():
            f = factor_index[k]
            if f.value_type == "boolean" and not isinstance(v, bool):
                raise ValueError(f"factor '{k}' expects boolean value")
            if f.value_type in ("categorical", "ordinal"):
                # Allow numbers or strings. If possible_values provided, enforce membership (as str or number)
                if f.possible_values is not None and v not in f.possible_values:
                    raise ValueError(
                        f"factor '{k}' value '{v}' not in allowed set {f.possible_values}"
                    )

        # Validate emotions and emotion_probs
        if not self.emotions:
            raise ValueError("emotions list must not be empty")

        for e in self.emotions:
            if not isinstance(e, str) or not e:
                raise ValueError("every emotion must be a non-empty string")

        # All probs in [0,1] and keys subset of emotions
        emotion_set = set(self.emotions)
        if not self.emotion_probs:
            raise ValueError("emotion_probs must not be empty")

        for k, p in self.emotion_probs.items():
            if k not in emotion_set:
                raise ValueError(f"emotion_probs contains unknown emotion '{k}'")
            if not isinstance(p, (float, int)) or p < 0 or p > 1:
                raise ValueError(f"emotion_probs['{k}'] must be a probability in [0,1]")

        # Sum close to 1
        total = float(sum(self.emotion_probs.values()))
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"emotion_probs must sum to 1.0 (±1e-3); got {total}")

        return self
