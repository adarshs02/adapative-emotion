"""
Unit tests for caching, key normalization, and logistic pooling functionality.

These tests verify the core functionality added in the EmoBIRD codebase patch:
- CPT caching and retrieval
- Key normalization consistency  
- Logistic pooling (BIRD formula) correctness
- Rating validation
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import math

# Import the modules we're testing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dial_cache import save_cpt, load_cpt, clear_cache, cache_exists, get_cache_info, CACHE_PATH
from utils import norm_key, validate_rating, pool_logistic, RATING_SCALE


class TestDialCache(unittest.TestCase):
    """Test dial cache functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear any existing cache before each test
        if cache_exists():
            clear_cache()
    
    def tearDown(self):
        """Clean up after each test."""
        # Clear cache after each test
        if cache_exists():
            clear_cache()
    
    def test_save_and_load_cpt_round_trip(self):
        """Test that save_cpt and load_cpt work correctly together."""
        # Create test CPT data
        test_cpt = {
            "factors": [
                {"name": "stress_level", "possible_values": ["high", "low"]},
                {"name": "social_support", "possible_values": ["present", "absent"]}
            ],
            "emotions": ["anxiety", "relief", "anger"],
            "combinations": {
                "stress_level=high|social_support=present": {
                    "anxiety": 0.75,
                    "relief": 0.25,
                    "anger": 0.50
                },
                "stress_level=low|social_support=absent": {
                    "anxiety": 0.25,
                    "relief": 0.75,
                    "anger": 0.30
                }
            },
            "metadata": {
                "method": "test_method",
                "num_combinations": 2,
                "num_factors": 2,
                "num_emotions": 3
            }
        }
        
        # Save CPT
        save_cpt(test_cpt)
        
        # Verify cache exists
        self.assertTrue(cache_exists())
        
        # Load CPT
        loaded_cpt = load_cpt()
        
        # Verify loaded data matches saved data
        self.assertIsNotNone(loaded_cpt)
        self.assertEqual(loaded_cpt["factors"], test_cpt["factors"])
        self.assertEqual(loaded_cpt["emotions"], test_cpt["emotions"])
        self.assertEqual(loaded_cpt["combinations"], test_cpt["combinations"])
        self.assertEqual(loaded_cpt["metadata"], test_cpt["metadata"])
    
    def test_load_cpt_no_cache(self):
        """Test load_cpt returns None when no cache exists."""
        # Ensure no cache exists
        self.assertFalse(cache_exists())
        
        # Load should return None
        result = load_cpt()
        self.assertIsNone(result)
    
    def test_cache_info(self):
        """Test cache info functionality."""
        # Test when no cache exists
        info = get_cache_info()
        self.assertFalse(info["exists"])
        self.assertEqual(info["combinations"], 0)
        self.assertEqual(info["emotions"], 0)
        
        # Create and save test data
        test_cpt = {
            "factors": [{"name": "test_factor", "possible_values": ["val1", "val2"]}],
            "emotions": ["emotion1", "emotion2", "emotion3"],
            "combinations": {"test_key": {"emotion1": 0.5}},
            "metadata": {"method": "test"}
        }
        save_cpt(test_cpt)
        
        # Test when cache exists
        info = get_cache_info()
        self.assertTrue(info["exists"])
        self.assertEqual(info["combinations"], 1)
        self.assertEqual(info["emotions"], 3)
        self.assertGreater(info["size_bytes"], 0)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Create test cache
        test_cpt = {"factors": [], "emotions": [], "combinations": {}, "metadata": {}}
        save_cpt(test_cpt)
        self.assertTrue(cache_exists())
        
        # Clear cache
        clear_cache()
        self.assertFalse(cache_exists())


class TestKeyNormalization(unittest.TestCase):
    """Test key normalization functionality."""
    
    def test_norm_key_basic(self):
        """Test basic key normalization."""
        result = norm_key("Attachment Style", "Strong")
        expected = "attachment_style=strong"
        self.assertEqual(result, expected)
    
    def test_norm_key_with_spaces(self):
        """Test key normalization with multiple spaces."""
        result = norm_key("  Social Support Level  ", "  Very High  ")
        expected = "social_support_level=very_high"
        self.assertEqual(result, expected)
    
    def test_norm_key_mixed_case(self):
        """Test key normalization with mixed case."""
        result = norm_key("StReSs LeVeL", "ExTrEmElY HiGh")
        expected = "stress_level=extremely_high"
        self.assertEqual(result, expected)
    
    def test_norm_key_consistency(self):
        """Test that equivalent inputs produce identical keys."""
        key1 = norm_key("Stress Level", "High")
        key2 = norm_key(" stress level ", " HIGH ")
        key3 = norm_key("STRESS LEVEL", "high")
        
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
        self.assertEqual(key1, "stress_level=high")


class TestRatingValidation(unittest.TestCase):
    """Test rating validation functionality."""
    
    def test_validate_rating_valid_inputs(self):
        """Test validation with valid rating inputs."""
        valid_ratings = ["very-unlikely", "unlikely", "neutral", "likely", "very-likely"]
        
        for rating in valid_ratings:
            # Test exact match
            result = validate_rating(rating)
            self.assertEqual(result, rating)
            
            # Test with extra whitespace
            result = validate_rating(f"  {rating}  ")
            self.assertEqual(result, rating)
            
            # Test with different case
            result = validate_rating(rating.upper())
            self.assertEqual(result, rating)
    
    def test_validate_rating_invalid_inputs(self):
        """Test validation raises errors for invalid inputs."""
        invalid_ratings = ["high", "low", "moderate", "maybe", "probably", "never", "always"]
        
        for rating in invalid_ratings:
            with self.assertRaises(ValueError) as context:
                validate_rating(rating)
            self.assertIn("Illegal rating", str(context.exception))
    
    def test_rating_scale_mapping(self):
        """Test that rating scale mapping is correct."""
        expected_mappings = {
            "very-unlikely": 0.05,
            "unlikely": 0.25,
            "neutral": 0.50,
            "likely": 0.75,
            "very-likely": 0.95
        }
        
        self.assertEqual(RATING_SCALE, expected_mappings)


class TestLogisticPooling(unittest.TestCase):
    """Test logistic pooling (BIRD formula) functionality."""
    
    def test_pool_logistic_single_probability(self):
        """Test pooling with a single probability."""
        result = pool_logistic([0.7])
        self.assertEqual(result, 0.7)
    
    def test_pool_logistic_empty_list(self):
        """Test pooling with empty list returns neutral."""
        result = pool_logistic([])
        self.assertEqual(result, 0.5)
    
    def test_pool_logistic_two_probabilities(self):
        """Test pooling with two probabilities."""
        # Test case: p1=0.8, p2=0.6
        # BIRD formula: (0.8 * 0.6) / (0.8 * 0.6 + 0.2 * 0.4) = 0.48 / (0.48 + 0.08) = 0.48 / 0.56 ≈ 0.857
        result = pool_logistic([0.8, 0.6])
        expected = (0.8 * 0.6) / (0.8 * 0.6 + 0.2 * 0.4)
        self.assertAlmostEqual(result, expected, places=6)
        self.assertAlmostEqual(result, 0.857142857, places=6)
    
    def test_pool_logistic_three_probabilities(self):
        """Test pooling with three probabilities."""
        # Test case from requirements: [0.8, 0.6, 0.7] ≈ 0.88
        result = pool_logistic([0.8, 0.6, 0.7])
        
        # Manual calculation:
        # prodP = 0.8 * 0.6 * 0.7 = 0.336
        # prodN = 0.2 * 0.4 * 0.3 = 0.024
        # result = 0.336 / (0.336 + 0.024) = 0.336 / 0.36 ≈ 0.933
        prod_p = 0.8 * 0.6 * 0.7
        prod_n = 0.2 * 0.4 * 0.3
        expected = prod_p / (prod_p + prod_n)
        
        self.assertAlmostEqual(result, expected, places=6)
        # Note: The requirement says ≈ 0.88, but the actual BIRD formula gives ≈ 0.933
        # This is mathematically correct for the BIRD formula
    
    def test_pool_logistic_extreme_values(self):
        """Test pooling with extreme probability values."""
        # Test with very high probabilities
        result = pool_logistic([0.99, 0.95, 0.98])
        self.assertGreater(result, 0.9)  # Should be very high
        
        # Test with very low probabilities  
        result = pool_logistic([0.01, 0.05, 0.02])
        self.assertLess(result, 0.1)  # Should be very low
        
        # Test with mixed extreme values
        result = pool_logistic([0.01, 0.99])
        # Should be dominated by the low probability in BIRD formula
        self.assertLess(result, 0.5)
    
    def test_pool_logistic_neutral_probabilities(self):
        """Test pooling with neutral probabilities."""
        result = pool_logistic([0.5, 0.5, 0.5])
        self.assertAlmostEqual(result, 0.5, places=6)
    
    def test_pool_logistic_zero_division_protection(self):
        """Test that pooling handles edge cases without division by zero."""
        # This shouldn't happen in practice, but test edge case protection
        result = pool_logistic([0.0, 1.0])
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
