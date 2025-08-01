#!/usr/bin/env python3
"""
Test the new factor generator implementation
"""

import sys
import os
import json
sys.path.append('/mnt/shared/adarsh/EmoBIRD')

from config import EmobirdConfig
from factor_generator import FactorGenerator
from vllm_wrapper import VLLMWrapper

def test_new_factor_generator():
    """Test the new factor generator."""
    
    print("🧪 Testing New Factor Generator")
    print("=" * 50)
    
    # Initialize components
    config = EmobirdConfig()
    vllm_wrapper = VLLMWrapper(config)  # vLLM initializes automatically
    factor_generator = FactorGenerator(config, vllm_wrapper)
    
    # Load examples from JSON file
    print("📂 Loading examples from examples.json...")
    with open('/mnt/shared/adarsh/EmoBIRD/examples/examples1.json', 'r') as f:
        examples_data = json.load(f)
    
    examples = examples_data['examples']
    print(f"Found {len(examples)} examples to test\n")
    
    # Test each example
    for i, user_situation in enumerate(examples, 1):
        print(f"{'='*60}")
        print(f"📝 EXAMPLE {i}/{len(examples)}")
        print(f"{'='*60}")
        print(f"Situation: {user_situation}")
        print("\n🧠 Generating factors...")
        
        # Generate factors
        result = factor_generator.generate_factors(user_situation)
        
        print(f"\n✅ Generation method: {result['generation_method']}")
        print(f"📊 Number of factors generated: {len(result['factors'])}")
        
        print("\n🎯 Generated Factors:")
        for j, factor in enumerate(result['factors'], 1):
            print(f"{j}. {factor['name']}")
            print(f"   Description: {factor['description']}")
            print(f"   Values: {factor['possible_values']}")
            print(f"   Selected: {result['selected_values'].get(factor['name'], 'N/A')}")
            print()
        
        # Test situation analysis
        print("🔍 Testing situation analysis...")
        analysis = factor_generator.analyze_situation(user_situation, result['factors'])
        
        print("📈 Situation Analysis:")
        for factor_name, value in analysis.items():
            print(f"  {factor_name}: {value}")
        
        print(f"\n✅ Example {i} completed!\n")
    
    print("\n🎉 All examples tested successfully!")

if __name__ == "__main__":
    test_new_factor_generator()
