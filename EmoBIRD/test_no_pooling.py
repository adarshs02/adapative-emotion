#!/usr/bin/env python3
"""
Test script to verify the no-pooling version of EmoBIRD works correctly.
"""

import sys
from emobird_poc import Emobird
from utils import print_gpu_info

def test_basic_functionality():
    """Test basic functionality of the no-pooling version."""
    
    print("="*70)
    print("TESTING EMOBIRD WITHOUT POOLING")
    print("="*70)
    
    # Initialize system
    print("\n🔄 Initializing EmoBIRD...")
    emobird = Emobird()
    emobird.verbose = False  # Reduce output for testing
    
    # Test situations
    test_cases = [
        {
            "situation": "I got promoted at work and my team celebrated with me.",
            "expected_emotions": ["joy", "happiness", "pride"],
            "name": "Positive scenario"
        },
        {
            "situation": "My friend betrayed my trust by lying to me.",
            "expected_emotions": ["anger", "disappointment", "sadness"],
            "name": "Negative scenario"
        },
        {
            "situation": "I'm moving to a new city for a great job opportunity.",
            "expected_emotions": ["excitement", "anxiety", "anticipation"],
            "name": "Mixed scenario"
        }
    ]
    
    for test in test_cases:
        print(f"\n\n📝 Test: {test['name']}")
        print(f"   Situation: {test['situation'][:50]}...")
        
        # Analyze
        result = emobird.analyze_emotion(test['situation'])
        
        # Check for required fields
        assert 'emotions' in result, "Missing 'emotions' in result"
        assert 'factors' in result, "Missing 'factors' in result"
        assert 'explanation' in result, "Missing 'explanation' in result"
        assert 'metadata' in result, "Missing 'metadata' in result"
        
        # Check metadata
        assert result['metadata']['method'] == 'direct_llm_assessment', "Wrong method in metadata"
        assert result['metadata']['pooling'] == 'none', "Pooling should be 'none'"
        
        # Display results
        print(f"\n   ✅ All required fields present")
        print(f"   Method: {result['metadata']['method']}")
        print(f"   Pooling: {result['metadata']['pooling']}")
        
        if result['emotions']:
            print(f"\n   Top emotions detected:")
            sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, prob in sorted_emotions:
                print(f"     - {emotion}: {prob:.3f}")
        
        if 'explanation' in result and result['explanation']:
            print(f"\n   Explanation: {result['explanation'][:100]}...")
    
    print("\n\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nThe EmoBIRD system is now using direct LLM assessment")
    print("without BIRD pooling. The LLM considers all factors")
    print("holistically when determining emotion probabilities.")

if __name__ == "__main__":
    print_gpu_info()
    test_basic_functionality()
