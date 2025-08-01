#!/usr/bin/env python3
"""
Factor Generation Comparison Test

Tests factor generation in three modes:
1. User input only
2. Abstract + user input (current pipeline)
3. Abstract only

Similar to scenario_generator testing approach.
"""

import json
import sys
import os

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EmobirdConfig
from factor_generator import FactorGenerator
from scenario_generator import ScenarioGenerator
from emobird import Emobird

def test_factor_generation_modes():
    """Test factor generation in all three modes."""
    
    print("🧪 FACTOR GENERATION MODE COMPARISON TEST")
    print("=" * 60)
    
    # Initialize components
    config = EmobirdConfig()
    emobird = Emobird()
    
    if not emobird.vllm_wrapper:
        print("❌ vLLM wrapper not available. Cannot run factor generation tests.")
        return
    
    factor_generator = FactorGenerator(config)
    factor_generator.set_vllm(emobird.vllm_wrapper)
    
    scenario_generator = ScenarioGenerator(config, emobird.vllm_wrapper)
    
    # Test situations
    test_situations = [
        "I just got promoted at work but I'm feeling overwhelmed by the new responsibilities and worried I might fail",
        "My best friend hasn't been responding to my messages for weeks and I'm starting to think they're avoiding me",
        "I'm about to give a presentation to 100 people and my heart is racing with anxiety",
    ]
    
    for i, user_situation in enumerate(test_situations, 1):
        print(f"\n🎯 TEST SITUATION {i}")
        print("-" * 40)
        print(f"Input: {user_situation}")
        print()
        
        # First generate an abstract for testing
        print("📋 Generating abstract...")
        abstract = scenario_generator._generate_abstract(user_situation)
        print(f"Abstract: {abstract}")
        print()
        
        # Test Mode 1: User input only
        print("🔸 MODE 1: USER INPUT ONLY")
        try:
            result_input_only = factor_generator._generate_factors_from_input_only(user_situation)
            factors_input_only = result_input_only.get('factors', [])
            values_input_only = result_input_only.get('selected_values', {})
            
            print(f"Generated {len(factors_input_only)} factors:")
            for j, factor in enumerate(factors_input_only, 1):
                selected_value = values_input_only.get(factor['name'], 'N/A')
                print(f"  {j}. {factor['name']}: {factor['description']}")
                print(f"     Values: {factor['possible_values']} | Selected: {selected_value}")
        except Exception as e:
            print(f"❌ Error in input-only mode: {e}")
            factors_input_only = []
            values_input_only = {}
        
        print()
        
        # Test Mode 2: Abstract + User input (current pipeline)
        print("🔸 MODE 2: ABSTRACT + USER INPUT")
        try:
            result_with_abstract = factor_generator._generate_factors_with_abstract(user_situation, abstract)
            factors_with_abstract = result_with_abstract.get('factors', [])
            values_with_abstract = result_with_abstract.get('selected_values', {})
            
            print(f"Generated {len(factors_with_abstract)} factors:")
            for j, factor in enumerate(factors_with_abstract, 1):
                selected_value = values_with_abstract.get(factor['name'], 'N/A')
                print(f"  {j}. {factor['name']}: {factor['description']}")
                print(f"     Values: {factor['possible_values']} | Selected: {selected_value}")
        except Exception as e:
            print(f"❌ Error in abstract+input mode: {e}")
            factors_with_abstract = []
            values_with_abstract = {}
        
        print()
        
        # Test Mode 3: Abstract only
        print("🔸 MODE 3: ABSTRACT ONLY")
        try:
            result_abstract_only = factor_generator._generate_factors_from_abstract_only(abstract)
            factors_abstract_only = result_abstract_only.get('factors', [])
            values_abstract_only = result_abstract_only.get('selected_values', {})
            
            print(f"Generated {len(factors_abstract_only)} factors:")
            for j, factor in enumerate(factors_abstract_only, 1):
                selected_value = values_abstract_only.get(factor['name'], 'N/A')
                print(f"  {j}. {factor['name']}: {factor['description']}")
                print(f"     Values: {factor['possible_values']} | Selected: {selected_value}")
        except Exception as e:
            print(f"❌ Error in abstract-only mode: {e}")
            factors_abstract_only = []
            values_abstract_only = {}
        
        print()
        
        # Comparison analysis
        print("📊 COMPARISON ANALYSIS")
        print("-" * 25)
        
        # Compare factor names across modes
        factor_names_1 = set(f['name'] for f in factors_input_only)
        factor_names_2 = set(f['name'] for f in factors_with_abstract)
        factor_names_3 = set(f['name'] for f in factors_abstract_only)
        
        all_factor_names = factor_names_1.union(factor_names_2).union(factor_names_3)
        
        print("Factor name comparison:")
        for name in sorted(all_factor_names):
            modes = []
            if name in factor_names_1:
                modes.append("Input")
            if name in factor_names_2:
                modes.append("Input+Abstract")
            if name in factor_names_3:
                modes.append("Abstract")
            print(f"  {name}: {', '.join(modes)}")
        
        # Common factors
        common_factors = factor_names_1.intersection(factor_names_2).intersection(factor_names_3)
        if common_factors:
            print(f"Common factors across all modes: {', '.join(sorted(common_factors))}")
        else:
            print("No factors common across all three modes")
        
        # Mode-specific factors
        input_only_factors = factor_names_1 - factor_names_2 - factor_names_3
        abstract_input_factors = factor_names_2 - factor_names_1 - factor_names_3
        abstract_only_factors = factor_names_3 - factor_names_1 - factor_names_2
        
        if input_only_factors:
            print(f"Input-only unique factors: {', '.join(sorted(input_only_factors))}")
        if abstract_input_factors:
            print(f"Abstract+Input unique factors: {', '.join(sorted(abstract_input_factors))}")
        if abstract_only_factors:
            print(f"Abstract-only unique factors: {', '.join(sorted(abstract_only_factors))}")
        
        print("\n" + "="*60)

def test_factor_value_extraction():
    """Test factor value extraction with the different generation methods."""
    
    print("\n🎯 FACTOR VALUE EXTRACTION TEST")
    print("=" * 40)
    
    # Initialize components
    config = EmobirdConfig()
    emobird = Emobird()
    
    if not emobird.vllm_wrapper:
        print("❌ vLLM wrapper not available. Cannot run extraction tests.")
        return
    
    factor_generator = FactorGenerator(config)
    factor_generator.set_vllm(emobird.vllm_wrapper)
    
    scenario_generator = ScenarioGenerator(config, emobird.vllm_wrapper)
    
    # Test situation
    user_situation = "I'm nervous about meeting my girlfriend's parents for the first time this weekend"
    abstract = scenario_generator._generate_abstract(user_situation)
    
    print(f"Situation: {user_situation}")
    print(f"Abstract: {abstract}")
    print()
    
    # Generate factors using different modes
    modes = [
        ("Input Only", factor_generator._generate_factors_from_input_only(user_situation)),
        ("Input+Abstract", factor_generator._generate_factors_with_abstract(user_situation, abstract)),
        ("Abstract Only", factor_generator._generate_factors_from_abstract_only(abstract))
    ]
    
    for mode_name, factors in modes:
        print(f"🔸 {mode_name.upper()} - FACTOR VALUE EXTRACTION")
        
        if not factors:
            print("❌ No factors generated")
            continue
            
        print("Generated factors:")
        for j, factor in enumerate(factors, 1):
            if isinstance(factor, dict) and 'name' in factor:
                possible_values = factor.get('possible_values', factor.get('values', 'N/A'))
                print(f"  {j}. {factor['name']}: {possible_values}")
            else:
                print(f"  {j}. {factor} (invalid format)")
        
        # Extract values for this situation
        try:
            factor_values = factor_generator.extract_factor_values(user_situation, abstract, factors)
            print("Extracted values:")
            for factor_name, value in factor_values.items():
                print(f"  {factor_name}: {value}")
        except Exception as e:
            print(f"❌ Error extracting values: {e}")
        
        print()

if __name__ == "__main__":
    print("Factor Generation Testing Suite")
    print("Testing different input modes for factor generation")
    print()
    
    try:
        test_factor_generation_modes()
        test_factor_value_extraction()
        
        print("\n✅ Factor generation testing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
