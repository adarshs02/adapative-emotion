#!/usr/bin/env python3
"""
Test and compare factor generation with and without abstracts
"""

import sys
import os
import json
sys.path.append('/mnt/shared/adarsh/EmoBIRD')

from config import EmobirdConfig
from factor_generator import FactorGenerator
from scenario_generator import ScenarioGenerator
from vllm_wrapper import VLLMWrapper

def compare_factor_generation():
    """Compare factor generation with and without abstracts."""
    
    print("🧪 Testing Factor Generation Comparison")
    print("🔬 With vs Without Abstracts")
    print("=" * 60)
    
    # Initialize components
    config = EmobirdConfig()
    vllm_wrapper = VLLMWrapper(config)  # vLLM initializes automatically
    factor_generator = FactorGenerator(config, vllm_wrapper)
    scenario_generator = ScenarioGenerator(config, vllm_wrapper)
    
    # Load examples from JSON file
    print("📂 Loading examples from examples.json...")
    with open('/mnt/shared/adarsh/EmoBIRD/examples/examples1.json', 'r') as f:
        examples_data = json.load(f)
    
    examples = examples_data['examples']
    print(f"Found {len(examples)} examples to test\n")
    
    # Test each example
    for i, user_situation in enumerate(examples, 1):
        print(f"{'='*80}")
        print(f"📝 EXAMPLE {i}/{len(examples)}")
        print(f"{'='*80}")
        print(f"Situation: {user_situation}")
        print()
        
        # Step 1: Generate abstract first
        print("📋 Generating abstract...")
        abstract = scenario_generator._generate_abstract(user_situation)
        print(f"Abstract: {abstract}")
        print()
        
        # Step 2: Generate factors WITHOUT abstract
        print("🧠 Generating factors WITHOUT abstract...")
        factors_without_abstract = factor_generator.generate_factors(user_situation, "")
        
        print("🎯 Factors (WITHOUT Abstract):")
        print(f"  Method: {factors_without_abstract['generation_method']}")
        print(f"  Count: {len(factors_without_abstract['factors'])}")
        for j, factor in enumerate(factors_without_abstract['factors'], 1):
            print(f"  {j}. {factor['name']}: {factor['description']}")
            print(f"     Values: {factor['possible_values']}")
        print()
        
        # Step 3: Generate factors WITH abstract
        print("🧠 Generating factors WITH abstract...")
        factors_with_abstract = factor_generator.generate_factors(user_situation, abstract)
        
        print("🎯 Factors (WITH Abstract):")
        print(f"  Method: {factors_with_abstract['generation_method']}")
        print(f"  Count: {len(factors_with_abstract['factors'])}")
        for j, factor in enumerate(factors_with_abstract['factors'], 1):
            print(f"  {j}. {factor['name']}: {factor['description']}")
            print(f"     Values: {factor['possible_values']}")
        print()
        
        # Step 4: Compare the results
        print("⚖️  COMPARISON ANALYSIS:")
        
        # Compare factor counts
        count_without = len(factors_without_abstract['factors'])
        count_with = len(factors_with_abstract['factors'])
        print(f"  Factor Count: Without={count_without}, With={count_with}")
        
        # Compare factor names
        names_without = set(f['name'] for f in factors_without_abstract['factors'])
        names_with = set(f['name'] for f in factors_with_abstract['factors'])
        
        common_factors = names_without & names_with
        unique_without = names_without - names_with
        unique_with = names_with - names_without
        
        print(f"  Common Factors: {list(common_factors) if common_factors else 'None'}")
        print(f"  Unique to WITHOUT: {list(unique_without) if unique_without else 'None'}")
        print(f"  Unique to WITH: {list(unique_with) if unique_with else 'None'}")
        
        # Generation method comparison
        method_without = factors_without_abstract['generation_method']
        method_with = factors_with_abstract['generation_method']
        print(f"  Generation Success: Without={method_without}, With={method_with}")
        
        print()
        print("📊 QUALITY ASSESSMENT:")
        
        # Simple quality indicators
        fallback_without = method_without == 'fallback'
        fallback_with = method_with == 'fallback'
        
        if not fallback_without and not fallback_with:
            print("  ✅ Both methods generated factors successfully")
            if count_with > count_without:
                print("  📈 Abstract version generated MORE factors")
            elif count_without > count_with:
                print("  📉 Abstract version generated FEWER factors")
            else:
                print("  ⚖️  Both versions generated SAME number of factors")
                
        elif fallback_without and not fallback_with:
            print("  ✅ Abstract helped! Only WITH abstract succeeded")
        elif not fallback_without and fallback_with:
            print("  ⚠️  Abstract hindered! Only WITHOUT abstract succeeded")
        else:
            print("  ❌ Both methods fell back to defaults")
        
        print(f"\n✅ Example {i} comparison completed!\n")
    
    print("\n🎉 All comparisons completed!")

if __name__ == "__main__":
    compare_factor_generation()
