"""
Tests for FactorEntailment class.
"""

import pytest
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_entailment import FactorEntailment


class MockVLLMWrapper:
    """Mock vLLM wrapper for testing."""
    
    def __init__(self, responses: Dict[str, Dict[str, Any]] = None):
        self.responses = responses or {}
        self.default_response = {"applies": "neutral", "reasoning": "test reasoning"}
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Mock JSON generation that returns predefined responses."""
        # Simple pattern matching for test responses
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default_response


@pytest.fixture
def dummy_factors():
    """Dummy factor definitions for testing."""
    return [
        {
            "name": "relationship",
            "values": ["stranger", "friend", "family"]
        },
        {
            "name": "fairness", 
            "values": ["fair", "unfair"]
        }
    ]


@pytest.fixture
def mock_vllm():
    """Mock vLLM wrapper with predefined responses."""
    responses = {
        "relationship": {"applies": "likely", "reasoning": "test relationship"},
        "fairness": {"applies": "very-likely", "reasoning": "test fairness"},
        "stranger": {"applies": "unlikely", "reasoning": "not a stranger"},
        "friend": {"applies": "very-likely", "reasoning": "clearly a friend"},
        "family": {"applies": "neutral", "reasoning": "not family"},
        "fair": {"applies": "unlikely", "reasoning": "situation seems unfair"},
        "unfair": {"applies": "very-likely", "reasoning": "clearly unfair"}
    }
    return MockVLLMWrapper(responses)


def test_factor_entailment_initialization(dummy_factors, mock_vllm):
    """Test FactorEntailment initialization."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    assert entailment.vllm_wrapper == mock_vllm
    assert entailment.factors == dummy_factors
    assert len(entailment.VERBAL2P) == 5  # 5 scale points


def test_entail_values_basic(dummy_factors, mock_vllm):
    """Test basic factor value entailment."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    story = "My friend betrayed my trust and treated me unfairly"
    
    result = entailment.entail_values(story)
    
    # Should return one value per factor
    assert len(result) == 2
    assert "relationship" in result
    assert "fairness" in result
    
    # Based on mock responses, should choose high-scoring values
    assert result["relationship"] in ["stranger", "friend", "family"]
    assert result["fairness"] in ["fair", "unfair"]


def test_assess_factor_value(dummy_factors, mock_vllm):
    """Test individual factor value assessment."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    story = "test story"
    
    # Test known responses
    score_friend = entailment._assess_factor_value(story, "relationship", "friend")
    score_unfair = entailment._assess_factor_value(story, "fairness", "unfair")
    
    # Based on mock responses
    assert score_friend == 0.95  # "very-likely" → 0.95
    assert score_unfair == 0.95  # "very-likely" → 0.95


def test_empty_factors():
    """Test behavior with empty factors list."""
    mock_vllm = MockVLLMWrapper()
    entailment = FactorEntailment(mock_vllm, [])
    
    result = entailment.entail_values("test story")
    
    assert result == {}


def test_factor_without_values(mock_vllm):
    """Test behavior with factor that has no values."""
    factors_no_values = [
        {"name": "empty_factor", "values": []},
        {"name": "missing_values"}  # No 'values' key
    ]
    
    entailment = FactorEntailment(mock_vllm, factors_no_values)
    result = entailment.entail_values("test story")
    
    assert result == {}


def test_get_factor_definitions(dummy_factors, mock_vllm):
    """Test getting factor definitions."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    definitions = entailment.get_factor_definitions()
    
    assert definitions == dummy_factors


def test_get_scale_info(dummy_factors, mock_vllm):
    """Test getting scale information."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    scale_info = entailment.get_scale_info()
    
    assert "scale" in scale_info
    assert "threshold_for_true" in scale_info
    assert scale_info["threshold_for_true"] == 0.75
    assert len(scale_info["scale"]) == 5


def test_verbal2p_scale():
    """Test that the verbal to probability scale is correct."""
    expected_scale = {
        'very-unlikely': 0.05,
        'unlikely': 0.25,
        'neutral': 0.50,
        'likely': 0.75,
        'very-likely': 0.95
    }
    
    assert FactorEntailment.VERBAL2P == expected_scale


def test_build_entailment_prompt(dummy_factors, mock_vllm):
    """Test prompt building for entailment."""
    entailment = FactorEntailment(mock_vllm, dummy_factors)
    
    story = "Test story"
    prompt = entailment._build_entailment_prompt(story, "relationship", "friend")
    
    assert "Test story" in prompt
    assert "relationship" in prompt
    assert "friend" in prompt
    assert "very-unlikely" in prompt
    assert "very-likely" in prompt
    assert "JSON" in prompt


def test_error_handling_in_assessment(dummy_factors):
    """Test error handling during factor assessment."""
    class ErrorMockVLLM:
        def generate_json(self, prompt):
            raise Exception("Mock error")
    
    entailment = FactorEntailment(ErrorMockVLLM(), dummy_factors)
    
    # Should return neutral score on error
    score = entailment._assess_factor_value("story", "relationship", "friend")
    assert score == 0.50  # Neutral fallback


if __name__ == "__main__":
    pytest.main([__file__])
