"""
Tests for the Llama-based scenario router.
"""

import pytest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_router import route, route_top_k, LlamaRouter
import config


class TestLlamaRouter:
    """Test cases for LlamaRouter functionality."""
    
    @pytest.fixture(scope="class")
    def router(self):
        """Fixture to provide a router instance for testing."""
        # Skip tests if index file doesn't exist
        if not os.path.exists(config.HNSW_INDEX_PATH):
            pytest.skip("HNSW index not found - run build_index.py first")
        
        return LlamaRouter()
    
    def test_router_initialization(self, router):
        """Test that router initializes correctly."""
        assert router.llm is not None
        assert router.index is not None
        assert router.scenario_mapping is not None
    
    def test_embed_text(self, router):
        """Test text embedding functionality."""
        text = "I am feeling happy today"
        embedding = router.embed_text(text)
        
        assert embedding is not None
        assert embedding.shape == (1, router.llm.llm_engine.model_config.get_hidden_size())
        assert embedding.dtype.name.startswith('float')
    
    def test_route_familiar_scenario(self, router):
        """Test routing with a scenario that should match well."""
        # Use a situation that should match a known scenario
        situation = "My friend just got a promotion at work and I'm happy for them"
        # Use a lower threshold to ensure we get a match
        scenario_id, confidence = router.route(situation, threshold=0.25)
        
        # With raw scores, expect a reasonably high confidence for good matches
        assert confidence > 0.26
        assert scenario_id is not None
    
    def test_route_unfamiliar_scenario(self, router):
        """Test routing with an unfamiliar situation."""
        # Use a very specific technical situation unlikely to match well
        situation = "I am debugging a segmentation fault in my CUDA kernel implementation"
        scenario_id, confidence = router.route(situation)
        
        # Should have low confidence and potentially no match
        assert confidence < 0.28
        # scenario_id might be None if below threshold
    
    def test_route_with_custom_threshold(self, router):
        """Test routing with custom threshold."""
        situation = "I received unexpected good news today"
        
        # High threshold - might not match
        scenario_id_high, conf_high = router.route(situation, threshold=0.9)
        
        # Low threshold - should match
        scenario_id_low, conf_low = router.route(situation, threshold=0.1)
        
        assert conf_high == conf_low  # Same situation, same confidence
        if conf_high < 0.9:
            assert scenario_id_high is None
        if conf_low >= 0.1:
            assert scenario_id_low is not None
    
    def test_route_top_k(self, router):
        """Test getting top-k scenario matches."""
        situation = "I'm feeling nervous about an upcoming presentation"
        results = router.route_top_k(situation, k=3)
        
        assert len(results) <= 3
        assert all('scenario_id' in result for result in results)
        assert all('description' in result for result in results)
        assert all('confidence' in result for result in results)
        
        # Results should be sorted by confidence (highest first)
        confidences = [result['confidence'] for result in results]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_get_scenario_info(self, router):
        """Test getting scenario information by ID."""
        # Get a scenario ID first
        situation = "I'm excited about a vacation"
        scenario_id, _ = router.route(situation, threshold=0.0)  # Low threshold to ensure match
        
        if scenario_id:
            info = router.get_scenario_info(scenario_id)
            assert info is not None
            assert info['id'] == scenario_id
            assert 'description' in info
        
        # Test with non-existent ID
        fake_info = router.get_scenario_info("non_existent_scenario")
        assert fake_info is None


class TestRouterFunctions:
    """Test the convenience functions."""
    
    def test_route_function(self):
        """Test the standalone route function."""
        if not os.path.exists(config.HNSW_INDEX_PATH):
            pytest.skip("Index file not found")
        
        situation = "I'm feeling grateful for my family"
        scenario_id, confidence = route(situation)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_route_top_k_function(self):
        """Test the standalone route_top_k function."""
        if not os.path.exists(config.HNSW_INDEX_PATH):
            pytest.skip("Index file not found")
        
        situation = "I'm worried about my health"
        results = route_top_k(situation, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, dict) for result in results)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
