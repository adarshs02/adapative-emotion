#!/usr/bin/env python3
"""
Test script to compare scenario generation with different inputs:
1. Just user input → scenario
2. User input + abstract → scenario (current pipeline)  
3. Just abstract → scenario
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from scenario_generator import ScenarioGenerator
from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper

def load_test_situations():
    """Load test situations from examples.json"""
    try:
        with open('examples.json', 'r') as f:
            data = json.load(f)
            return data.get('examples', [])
    except FileNotFoundError:
        print("❌ examples.json not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing examples.json: {e}")
        return []

def test_scenario_comparison():
    """Test scenario generation with three different input approaches."""
    
    print("🧪 SCENARIO GENERATION COMPARISON TEST")
    print("=" * 60)
    
    try:
        # Initialize system
        print("🔧 Initializing Emobird system...")
        config = EmobirdConfig()
        vllm_wrapper = VLLMWrapper(config)
        scenario_generator = ScenarioGenerator(config, vllm_wrapper)
        
        # Load test situations
        test_situations = load_test_situations()
        if not test_situations:
            print("❌ No test situations found!")
            return
        
        print(f"📝 Loaded {len(test_situations)} test situations")
        print("🎯 Testing 3 approaches: user input only, user input + abstract, abstract only")
        print("=" * 60)
        
        # Test each situation with all three approaches
        for i, situation in enumerate(test_situations, 1):
            print(f"--- Test {i}/{len(test_situations)} ---")
            print(f"📝 Situation: '{situation[:60]}{'...' if len(situation) > 60 else ''}'")
            print("-" * 60)
            
            # First generate abstract for approaches 2 and 3
            print("📋 Generating abstract...")
            abstract = scenario_generator._generate_abstract(situation)
            print(f"📊 Abstract: '{abstract}'")
            print()
            
            # APPROACH 1: Just user input → scenario
            print("🔬 APPROACH 1: User Input Only")
            try:
                scenario_1 = scenario_generator._generate_scenario_from_input_only(situation)
                print(f"✅ Scenario: {scenario_1.get('description', 'N/A')[:100]}...")
            except Exception as e:
                print(f"❌ Failed: {e}")
            print()
            
            # APPROACH 2: User input + abstract → scenario (current pipeline)
            print("🔬 APPROACH 2: User Input + Abstract")  
            try:
                scenario_2 = scenario_generator._generate_scenario_with_abstract(situation, abstract)
                print(f"✅ Scenario: {scenario_2.get('description', 'N/A')[:100]}...")
            except Exception as e:
                print(f"❌ Failed: {e}")
            print()
            
            # APPROACH 3: Just abstract → scenario
            print("🔬 APPROACH 3: Abstract Only")
            try:
                scenario_3 = scenario_generator._generate_scenario_from_abstract_only(abstract)
                print(f"✅ Scenario: {scenario_3.get('description', 'N/A')[:100]}...")
            except Exception as e:
                print(f"❌ Failed: {e}")
            
            if i < len(test_situations):
                print("\n" + "=" * 60 + "\n")
        
        print("\n🎯 SCENARIO COMPARISON TEST COMPLETE")
        print("Check logs in logs/testing/ for detailed model interactions")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scenario_comparison()
