#!/usr/bin/env python3
"""
Test the complete EmoBIRD pipeline: Factor Generation → CPT Generation
"""

import sys
import os
import json
sys.path.append('/mnt/shared/adarsh/EmoBIRD')

from config import EmobirdConfig
from factor_generator import FactorGenerator
from cpt_generator import CPTGenerator
from vllm_wrapper import VLLMWrapper

def test_complete_pipeline():
    """Test the complete factor generation + CPT generation pipeline."""
    
    print("🧪 Testing Complete EmoBIRD Pipeline")
    print("🔗 Factor Generation → CPT Generation")
    print("=" * 60)
    
    # Initialize components
    config = EmobirdConfig()
    vllm_wrapper = VLLMWrapper(config)  # vLLM initializes automatically
    factor_generator = FactorGenerator(config, vllm_wrapper)
    cpt_generator = CPTGenerator(config)
    cpt_generator.set_vllm(vllm_wrapper)
    
    # Test with a specific example
    user_situation = "I'm nervous about giving a presentation to my boss tomorrow"
    
    print(f"📝 Test Situation: {user_situation}")
    print()
    
    # Step 1: Generate CPT (determining relevant factors internally)
    print("🎲 STEP 1: Generating CPT with internal factor determination...")
    
    # Create a simple scenario dict for the CPT generator
    scenario = {
        'id': 'dynamic_scenario',
        'description': f"Emotional scenario for: {user_situation}",
        'tags': ['user_generated', 'dynamic']
    }
    
    cpt_result = cpt_generator.generate_cpt(scenario, user_situation)
    
    print(f"✅ CPT generation method: {cpt_result.get('metadata', {}).get('generation_method', 'unknown')}")
    print(f"🔢 Generated CPT rows: {len(cpt_result.get('cpt', []))}")
    print(f"🎭 Factors determined by CPT: {len(cpt_result.get('factors', {}))}")
    
    # Display the factors that were determined by CPT generation
    print("\n🎯 Factors Determined by CPT:")
    cpt_factors = cpt_result.get('factors', {})
    if isinstance(cpt_factors, list):
        factors_list = cpt_factors
    else:
        # Convert dict format to list format for consistency
        factors_list = []
        for name, values in cpt_factors.items():
            factors_list.append({
                'name': name,
                'possible_values': values if isinstance(values, list) else [values],
                'description': f"Factor: {name}"
            })
    
    for i, factor in enumerate(factors_list, 1):
        print(f"{i}. {factor['name']}")
        print(f"   Values: {factor.get('possible_values', ['high', 'low'])}")
    print()
    
    # Step 2: Analyze user situation for specific factor values
    print("🧠 STEP 2: Analyzing user situation for specific factor values...")
    
    factor_result = factor_generator.generate_factors(user_situation)
    
    print(f"✅ Factor analysis method: {factor_result['generation_method']}")
    print(f"📊 Factor values determined: {len(factor_result['factors'])}")
    
    # Calculate expected combinations from CPT factors
    total_combinations = 1
    for factor in factors_list:
        total_combinations *= len(factor.get('possible_values', ['high', 'low']))
    
    print(f"📈 Expected CPT rows: {total_combinations} (from {len(factors_list)} factors)")
    
    print(f"✅ CPT generation method: {cpt_result.get('metadata', {}).get('generation_method', 'unknown')}")
    print(f"🔢 Generated CPT rows: {len(cpt_result.get('cpt', []))}")
    print(f"🎭 Factors in CPT: {len(cpt_result.get('factors', {}))}")
    
    # Step 3: Display sample CPT entries
    print("\n📋 Sample CPT Entries:")
    cpt_entries = cpt_result.get('cpt', [])
    
    if cpt_entries:
        # Show first few entries
        for i, entry in enumerate(cpt_entries[:3], 1):
            print(f"\nEntry {i}:")
            # Show factor values
            factor_values = []
            for factor_name in cpt_result.get('factors', {}):
                if factor_name in entry:
                    factor_values.append(f"{factor_name}={entry[factor_name]}")
            print(f"  Factors: {', '.join(factor_values)}")
            
            # Show emotion probabilities
            emotions = entry.get('emotions', {})
            if emotions:
                print("  Emotion Probabilities:")
                for emotion, prob in emotions.items():
                    print(f"    {emotion}: {prob:.3f}")
        
        if len(cpt_entries) > 3:
            print(f"\n  ... and {len(cpt_entries) - 3} more entries")
    else:
        print("  ❌ No CPT entries generated")
    
    # Step 4: Validate completeness
    print("\n🔍 PIPELINE VALIDATION:")
    
    # Check if we have factors
    has_factors = len(factor_result['factors']) > 0
    print(f"  ✅ Factors generated: {has_factors} ({len(factor_result['factors'])} factors)")
    
    # Check if we have CPT
    has_cpt = len(cpt_result.get('cpt', [])) > 0
    print(f"  ✅ CPT generated: {has_cpt} ({len(cpt_result.get('cpt', []))} entries)")
    
    # Check if CPT covers all combinations
    expected_combinations = total_combinations
    actual_combinations = len(cpt_result.get('cpt', []))
    complete_coverage = actual_combinations == expected_combinations
    print(f"  ✅ Complete coverage: {complete_coverage} ({actual_combinations}/{expected_combinations})")
    
    # Check if emotion probabilities are valid
    valid_probabilities = True
    for entry in cpt_entries[:5]:  # Check first 5 entries
        emotions = entry.get('emotions', {})
        if emotions:
            total_prob = sum(emotions.values())
            if abs(total_prob - 1.0) > 0.01:  # Allow small floating point errors
                valid_probabilities = False
                break
    print(f"  ✅ Valid probabilities: {valid_probabilities}")
    
    # Overall success
    pipeline_success = has_factors and has_cpt and complete_coverage and valid_probabilities
    print(f"\n🎉 PIPELINE SUCCESS: {pipeline_success}")
    
    if pipeline_success:
        print("🚀 EmoBIRD pipeline is working end-to-end!")
        print("💡 Ready for emotion inference on user situations")
    else:
        print("⚠️  Pipeline needs debugging")
    
    return {
        'factors': factor_result,
        'cpt': cpt_result,
        'validation': {
            'has_factors': has_factors,
            'has_cpt': has_cpt,
            'complete_coverage': complete_coverage,
            'valid_probabilities': valid_probabilities,
            'pipeline_success': pipeline_success
        }
    }

def test_multiple_examples():
    """Test pipeline on multiple examples for robustness."""
    
    print("\n" + "="*80)
    print("🧪 ROBUSTNESS TEST: Multiple Examples")
    print("="*80)
    
    # Load examples
    with open('/mnt/shared/adarsh/EmoBIRD/examples/examples1.json', 'r') as f:
        examples_data = json.load(f)
    
    examples = examples_data['examples'][:3]  # Test first 3 for speed
    
    # Initialize components
    config = EmobirdConfig()
    vllm_wrapper = VLLMWrapper(config)
    factor_generator = FactorGenerator(config, vllm_wrapper)
    cpt_generator = CPTGenerator(config)
    cpt_generator.set_vllm(vllm_wrapper)
    
    results = []
    
    for i, user_situation in enumerate(examples, 1):
        print(f"\n📝 EXAMPLE {i}/{len(examples)}")
        print(f"Situation: {user_situation[:100]}...")
        
        # Generate CPT (with internal factor determination)
        scenario = {
            'id': f'dynamic_scenario_{i}',
            'description': f"Emotional scenario for: {user_situation[:50]}...",
            'tags': ['user_generated', 'dynamic']
        }
        
        cpt_result = cpt_generator.generate_cpt(scenario, user_situation)
        
        # Analyze user situation for specific factor values
        factor_result = factor_generator.generate_factors(user_situation)
        
        # Quick validation
        has_cpt = len(cpt_result.get('cpt', [])) > 0
        has_factors = len(factor_result['factors']) > 0
        
        success = has_cpt and has_factors
        print(f"✅ Result: {success} (cpt_entries={len(cpt_result.get('cpt', []))}, factor_values={len(factor_result['factors'])})")
        
        results.append(success)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n🎯 ROBUSTNESS: {success_rate:.1f}% success rate ({sum(results)}/{len(results)} examples)")
    
    if success_rate >= 80:
        print("🚀 Pipeline is robust and ready for production!")
    else:
        print("⚠️  Pipeline needs reliability improvements")

if __name__ == "__main__":
    # Test single example in detail
    test_complete_pipeline()
    
    # Test multiple examples for robustness
    test_multiple_examples()
