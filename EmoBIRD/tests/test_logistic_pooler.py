"""
Tests for LogisticPooler class.
"""

import pytest
import math
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logistic_pooler import LogisticPooler
from cpt_utils import create_dummy_cpt


@pytest.fixture
def dummy_cpt_data():
    """Create dummy CPT data for testing."""
    return create_dummy_cpt(
        factors=["relationship", "fairness"],
        emotions=["anger", "sadness", "joy"],
        factor_values={
            "relationship": ["friend", "stranger"],
            "fairness": ["fair", "unfair"]
        }
    )


@pytest.fixture
def simple_cpt_data():
    """Simple CPT data with known probabilities for testing."""
    return {
        "factors": [
            {"name": "relationship", "values": ["friend", "stranger"]},
            {"name": "fairness", "values": ["fair", "unfair"]}
        ],
        "emotions": ["anger", "joy"],
        "cpt": {
            "relationship=friend": {"anger": 0.3, "joy": 0.7},
            "relationship=stranger": {"anger": 0.6, "joy": 0.4},
            "fairness=fair": {"anger": 0.2, "joy": 0.8},
            "fairness=unfair": {"anger": 0.9, "joy": 0.1}
        }
    }


def test_basic_pooling(simple_cpt_data):
    """Test basic BIRD pooling formula."""
    chosen_values = {"relationship": "friend", "fairness": "unfair"}
    
    # Pool anger emotion
    anger_prob = LogisticPooler.pool("anger", chosen_values, simple_cpt_data)
    
    # Expected calculation:
    # P(anger|friend) = 0.3, P(anger|unfair) = 0.9
    # prodP = 0.3 * 0.9 = 0.27
    # prodN = (1-0.3) * (1-0.9) = 0.7 * 0.1 = 0.07
    # result = 0.27 / (0.27 + 0.07) = 0.27 / 0.34 ≈ 0.794
    
    expected = 0.27 / (0.27 + 0.07)
    assert abs(anger_prob - expected) < 0.001


def test_pool_all_emotions(simple_cpt_data):
    """Test pooling all emotions at once."""
    chosen_values = {"relationship": "friend", "fairness": "fair"}
    
    all_probs = LogisticPooler.pool_all_emotions(chosen_values, simple_cpt_data)
    
    assert len(all_probs) == 2  # anger and joy
    assert "anger" in all_probs
    assert "joy" in all_probs
    
    # Probabilities should be between 0 and 1
    assert 0 <= all_probs["anger"] <= 1
    assert 0 <= all_probs["joy"] <= 1


def test_missing_factor_values(simple_cpt_data):
    """Test behavior when factor values are missing from CPT."""
    chosen_values = {"relationship": "unknown", "fairness": "unfair"}
    
    anger_prob = LogisticPooler.pool("anger", chosen_values, simple_cpt_data)
    
    # Should use 0.5 for missing "relationship=unknown"
    # So we have: P(anger|unknown)=0.5, P(anger|unfair)=0.9
    # prodP = 0.5 * 0.9 = 0.45
    # prodN = 0.5 * 0.1 = 0.05
    # result = 0.45 / (0.45 + 0.05) = 0.9
    
    expected = 0.45 / (0.45 + 0.05)
    assert abs(anger_prob - expected) < 0.001


def test_empty_chosen_values(simple_cpt_data):
    """Test behavior with empty chosen values."""
    anger_prob = LogisticPooler.pool("anger", {}, simple_cpt_data)
    
    # Should return neutral probability
    assert anger_prob == 0.5


def test_bird_pooling_formula():
    """Test the BIRD pooling formula directly."""
    # Test with known probabilities
    probs = [0.3, 0.9]
    
    result = LogisticPooler._bird_pooling_formula(probs)
    
    # prodP = 0.3 * 0.9 = 0.27
    # prodN = 0.7 * 0.1 = 0.07
    # result = 0.27 / (0.27 + 0.07) = 0.794...
    
    expected = 0.27 / (0.27 + 0.07)
    assert abs(result - expected) < 0.001


def test_bird_pooling_edge_cases():
    """Test BIRD pooling with edge cases."""
    # Empty list
    assert LogisticPooler._bird_pooling_formula([]) == 0.5
    
    # Single probability
    assert LogisticPooler._bird_pooling_formula([0.8]) == 0.8
    
    # All neutral probabilities
    result = LogisticPooler._bird_pooling_formula([0.5, 0.5, 0.5])
    assert abs(result - 0.5) < 0.001


def test_numerical_stability():
    """Test numerical stability with extreme probabilities."""
    # Test with probabilities close to 0 and 1
    extreme_probs = [0.001, 0.999]
    
    result = LogisticPooler._bird_pooling_formula(extreme_probs)
    
    # Should not crash and return reasonable result
    assert 0 <= result <= 1
    assert not math.isnan(result)
    assert not math.isinf(result)


def test_soft_pooling(simple_cpt_data):
    """Test soft pooling with factor scores."""
    factor_scores = {
        "relationship": {"friend": 0.8, "stranger": 0.2},
        "fairness": {"fair": 0.3, "unfair": 0.7}
    }
    
    anger_prob = LogisticPooler.pool_soft("anger", factor_scores, simple_cpt_data)
    
    # Should return a valid probability
    assert 0 <= anger_prob <= 1
    assert not math.isnan(anger_prob)


def test_soft_pooling_empty_scores(simple_cpt_data):
    """Test soft pooling with empty factor scores."""
    anger_prob = LogisticPooler.pool_soft("anger", {}, simple_cpt_data)
    
    # Should return neutral probability
    assert anger_prob == 0.5


def test_get_pooling_info():
    """Test getting pooling information."""
    info = LogisticPooler.get_pooling_info()
    
    assert "method" in info
    assert "formula" in info
    assert "missing_dial_default" in info
    assert info["missing_dial_default"] == 0.5
    assert info["method"] == "BIRD pooling formula"


def test_pooling_consistency():
    """Test that pooling is consistent and deterministic."""
    chosen_values = {"relationship": "friend", "fairness": "unfair"}
    
    # Run multiple times - should get same result
    results = []
    for _ in range(5):
        result = LogisticPooler.pool("anger", chosen_values, {
            "cpt": {
                "relationship=friend": {"anger": 0.3},
                "fairness=unfair": {"anger": 0.9}
            }
        })
        results.append(result)
    
    # All results should be identical
    assert len(set(results)) == 1


def test_missing_emotion_in_cpt(simple_cpt_data):
    """Test behavior when emotion is missing from CPT entries."""
    chosen_values = {"relationship": "friend", "fairness": "fair"}
    
    # Try to pool an emotion not in the CPT
    prob = LogisticPooler.pool("nonexistent", chosen_values, simple_cpt_data)
    
    # Should handle gracefully and return neutral
    assert prob == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
