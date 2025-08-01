"""
Tests for EmotionPredictor class.
"""

import pytest
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_predictor import EmotionPredictor
from factor_entailment import FactorEntailment
from logistic_pooler import LogisticPooler
from cpt_utils import create_dummy_cpt


class MockVLLMWrapper:
    """Mock vLLM wrapper for testing."""
    
    def __init__(self, responses: Dict[str, Dict[str, Any]] = None):
        self.responses = responses or {}
        self.default_response = {"applies": "neutral", "reasoning": "test reasoning"}
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Mock JSON generation that returns predefined responses."""
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default_response


@pytest.fixture
def simple_cpt_data():
    """Simple CPT data with known probabilities for testing."""
    return {
        "factors": [
            {"name": "relationship", "values": ["friend", "stranger"]},
            {"name": "intensity", "values": ["mild", "strong"]}
        ],
        "emotions": ["anger", "joy", "sadness"],
        "cpt": {
            "relationship=friend": {"anger": 0.2, "joy": 0.7, "sadness": 0.1},
            "relationship=stranger": {"anger": 0.6, "joy": 0.3, "sadness": 0.1},
            "intensity=mild": {"anger": 0.3, "joy": 0.5, "sadness": 0.2},
            "intensity=strong": {"anger": 0.8, "joy": 0.1, "sadness": 0.1}
        }
    }


@pytest.fixture
def mock_vllm():
    """Mock vLLM wrapper with predefined responses."""
    responses = {
        "relationship": {"applies": "likely", "reasoning": "test relationship"},
        "intensity": {"applies": "very-likely", "reasoning": "test intensity"},
        "friend": {"applies": "very-likely", "reasoning": "clearly a friend"},
        "stranger": {"applies": "unlikely", "reasoning": "not a stranger"},
        "mild": {"applies": "unlikely", "reasoning": "situation is intense"},
        "strong": {"applies": "very-likely", "reasoning": "very intense situation"}
    }
    return MockVLLMWrapper(responses)


@pytest.fixture
def emotion_predictor_setup(simple_cpt_data, mock_vllm):
    """Set up EmotionPredictor with mocked components."""
    factors = simple_cpt_data["factors"]
    entailment = FactorEntailment(mock_vllm, factors)
    pooler = LogisticPooler()
    predictor = EmotionPredictor(entailment, pooler, simple_cpt_data)
    return predictor


def test_emotion_predictor_initialization(simple_cpt_data, mock_vllm):
    """Test EmotionPredictor initialization."""
    factors = simple_cpt_data["factors"]
    entailment = FactorEntailment(mock_vllm, factors)
    pooler = LogisticPooler()
    predictor = EmotionPredictor(entailment, pooler, simple_cpt_data)
    
    assert predictor.entailment == entailment
    assert predictor.pooler == pooler
    assert predictor.cpt_data == simple_cpt_data


def test_predict_basic(emotion_predictor_setup):
    """Test basic emotion prediction."""
    predictor = emotion_predictor_setup
    
    story = "My friend really hurt me with their intense actions"
    
    result = predictor.predict(story, top_k=2)
    
    # Should return top-2 emotions
    assert len(result) == 2
    
    # Should return valid emotions from the CPT
    for emotion in result.keys():
        assert emotion in ["anger", "joy", "sadness"]
    
    # Should return valid probabilities
    for prob in result.values():
        assert 0 <= prob <= 1


def test_predict_all_emotions(emotion_predictor_setup):
    """Test predicting all emotions without top-k filtering."""
    predictor = emotion_predictor_setup
    
    story = "Test story for all emotions"
    
    result = predictor.predict_all(story)
    
    # Should return all emotions from CPT
    assert len(result) == 3  # anger, joy, sadness
    assert set(result.keys()) == {"anger", "joy", "sadness"}
    
    # Should return valid probabilities
    for prob in result.values():
        assert 0 <= prob <= 1


def test_predict_soft_pooling(emotion_predictor_setup):
    """Test prediction with soft pooling."""
    predictor = emotion_predictor_setup
    
    story = "Test story for soft pooling"
    
    result = predictor.predict_soft(story, top_k=2, use_soft_pooling=True)
    
    # Should return top-2 emotions
    assert len(result) == 2
    
    # Should return valid probabilities
    for prob in result.values():
        assert 0 <= prob <= 1


def test_predict_soft_fallback(emotion_predictor_setup):
    """Test that predict_soft falls back to hard prediction when disabled."""
    predictor = emotion_predictor_setup
    
    story = "Test story"
    
    # With soft pooling disabled, should behave like regular predict
    result_soft_disabled = predictor.predict_soft(story, top_k=3, use_soft_pooling=False)
    result_regular = predictor.predict(story, top_k=3)
    
    # Should have same structure (though values might differ due to randomness in mock)
    assert len(result_soft_disabled) == len(result_regular)
    assert set(result_soft_disabled.keys()) == set(result_regular.keys())


def test_get_prediction_info(emotion_predictor_setup):
    """Test getting prediction configuration information."""
    predictor = emotion_predictor_setup
    
    info = predictor.get_prediction_info()
    
    assert "factors" in info
    assert "emotions" in info
    assert "num_factor_combinations" in info
    assert "entailment_scale" in info
    assert "pooling_method" in info
    assert "supports_soft_pooling" in info
    
    assert info["supports_soft_pooling"] is True
    assert "relationship" in info["factors"]
    assert "intensity" in info["factors"]
    assert set(info["emotions"]) == {"anger", "joy", "sadness"}


def test_get_soft_factor_scores(emotion_predictor_setup):
    """Test getting soft factor scores."""
    predictor = emotion_predictor_setup
    
    story = "Test story"
    
    factor_scores = predictor._get_soft_factor_scores(story)
    
    # Should return scores for all factors
    assert "relationship" in factor_scores
    assert "intensity" in factor_scores
    
    # Each factor should have scores for all its values
    assert len(factor_scores["relationship"]) == 2  # friend, stranger
    assert len(factor_scores["intensity"]) == 2  # mild, strong
    
    # All scores should be valid probabilities
    for factor_name, value_scores in factor_scores.items():
        for score in value_scores.values():
            assert 0 <= score <= 1


def test_predict_with_different_top_k(emotion_predictor_setup):
    """Test prediction with different top_k values."""
    predictor = emotion_predictor_setup
    
    story = "Test story"
    
    # Test different top_k values
    result_1 = predictor.predict(story, top_k=1)
    result_2 = predictor.predict(story, top_k=2)
    result_3 = predictor.predict(story, top_k=3)
    
    assert len(result_1) == 1
    assert len(result_2) == 2
    assert len(result_3) == 3
    
    # Results should be sorted by probability (descending)
    if len(result_2) == 2:
        probs = list(result_2.values())
        assert probs[0] >= probs[1]
    
    if len(result_3) == 3:
        probs = list(result_3.values())
        assert probs[0] >= probs[1] >= probs[2]


def test_predict_empty_story(emotion_predictor_setup):
    """Test prediction with empty story."""
    predictor = emotion_predictor_setup
    
    result = predictor.predict("", top_k=2)
    
    # Should still return valid results
    assert len(result) == 2
    for prob in result.values():
        assert 0 <= prob <= 1


def test_predictor_consistency(emotion_predictor_setup):
    """Test that predictions are consistent for the same story."""
    predictor = emotion_predictor_setup
    
    story = "Consistent test story"
    
    # Run multiple predictions - should get same results due to deterministic mock
    results = []
    for _ in range(3):
        result = predictor.predict(story, top_k=2)
        results.append(result)
    
    # All results should be identical
    for i in range(1, len(results)):
        assert results[i] == results[0]


def test_predictor_with_minimal_cpt():
    """Test predictor with minimal CPT data."""
    minimal_cpt = {
        "factors": [{"name": "test_factor", "values": ["low", "high"]}],
        "emotions": ["happy", "sad"],
        "cpt": {
            "test_factor=low": {"happy": 0.3, "sad": 0.7},
            "test_factor=high": {"happy": 0.8, "sad": 0.2}
        }
    }
    
    mock_vllm = MockVLLMWrapper({
        "low": {"applies": "very-likely", "reasoning": "test"},
        "high": {"applies": "unlikely", "reasoning": "test"}
    })
    
    factors = minimal_cpt["factors"]
    entailment = FactorEntailment(mock_vllm, factors)
    pooler = LogisticPooler()
    predictor = EmotionPredictor(entailment, pooler, minimal_cpt)
    
    result = predictor.predict("test story", top_k=2)
    
    assert len(result) == 2
    assert set(result.keys()) == {"happy", "sad"}


if __name__ == "__main__":
    pytest.main([__file__])
