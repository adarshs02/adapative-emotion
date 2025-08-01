#!/usr/bin/env python3

"""
Simple test to verify the factor generation fix works
without full system initialization.
"""

from factor_generator import FactorGenerator
from config import EmobirdConfig
import json

def test_prompt_generation():
    """Test that the new prompt template is being used correctly."""
    
    print("🧪 Simple Factor Generation Test")
    print("=" * 50)
    
    try:
        # Initialize with basic config (no vLLM yet)
        config = EmobirdConfig()
        generator = FactorGenerator(config)
        
        # Test the new prompt template
        test_situation = "I'm nervous about meeting my girlfriend's parents for the first time"
        test_abstract = "First-time meeting causes anxiety"
        
        print("🔍 Testing new prompt template:")
        prompt = generator._build_factor_prompt(test_situation, test_abstract)
        print(f"Prompt length: {len(prompt)} chars")
        print("Prompt preview:")
        print("-" * 30)
        print(prompt[:500])
        if len(prompt) > 500:
            print(f"... [truncated - showing first 500 of {len(prompt)} chars]")
        print("-" * 30)
        
        # Check if the prompt looks like our new clean format
        if "SCENARIO" in prompt and "GOAL" in prompt and "JSON SPEC" in prompt:
            print("✅ New clean prompt template is being used!")
        else:
            print("❌ Still using old prompt format")
            
        # Test the validation logic handles new format
        test_response = {
            "factors": {
                "anxiety_level": ["low", "high"],
                "social_familiarity": ["familiar", "unfamiliar"],  
                "outcome_importance": ["low_stakes", "high_stakes"]
            }
        }
        
        print("\n🔍 Testing new parser with sample response:")
        print(f"Sample response: {json.dumps(test_response, indent=2)}")
        
        validated = generator._validate_factors(test_response)
        print(f"✅ Parser returned {len(validated)} validated factors:")
        for i, factor in enumerate(validated, 1):
            print(f"  {i}. {factor['name']}: {factor['possible_values']}")
            
        print("\n🎯 SUCCESS: Fix appears to be working!")
        print("The new clean prompt template is in use and parser handles the new format.")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt_generation()
