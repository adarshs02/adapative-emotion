#!/usr/bin/env python3
"""
Test script for isolating and testing ONLY the abstraction generation step.
This removes all other pipeline components to focus on debugging abstraction.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper
from scenario_generator import ScenarioGenerator
from utils import print_gpu_info

def test_abstraction_only():
    """Test only the abstraction generation component."""
    
    print("🔬 ABSTRACTION-ONLY TEST")
    print("=" * 50)
    
    # Load test situations from examples.json
    try:
        with open('examples.json', 'r') as f:
            examples_data = json.load(f)
        test_situations = examples_data['examples']
        print(f"📂 Loaded {len(test_situations)} test situations from examples.json")
    except Exception as e:
        print(f"⚠️ Failed to load examples.json: {e}")
        print("Using fallback test situations...")
        test_situations = [
            "I just got promoted at work after working really hard for months",
            "My best friend forgot my birthday and didn't even call me"
        ]
    
    try:
        # Initialize components
        print("🚀 Initializing system...")
        print_gpu_info()
        
        config = EmobirdConfig()
        vllm_wrapper = VLLMWrapper(config)
        scenario_generator = ScenarioGenerator(config)
        scenario_generator.set_vllm(vllm_wrapper)
        
        print("✅ System initialized successfully!")
        print(f"📝 Logging session started")
        
        # Test each situation
        for i, situation in enumerate(test_situations, 1):
            print(f"\n--- Test {i}/{len(test_situations)} ---")
            print(f"📝 Situation: '{situation}'")
            print("-" * 60)
            
            try:
                # ONLY test abstraction generation
                print("📋 Generating abstract...")
                abstract = scenario_generator._generate_abstract(situation)
                
                print("✅ Abstract generation completed!")
                print(f"📊 Abstract: '{abstract}'")
                print(f"📏 Length: {len(abstract)} characters")
                
                # Basic validation
                if abstract and len(abstract.strip()) > 0:
                    print("✅ Abstract validation: GENERATED")
                else:
                    print("❌ Abstract validation: FAILED (empty or None)")
                
            except Exception as e:
                print(f"❌ Abstract generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            if i < len(test_situations):
                print("\n" + "=" * 60 + "\n")
        
        print("\n🎯 ABSTRACTION TEST COMPLETE")
        print("Check logs in logs/testing/ for detailed model interactions")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_abstraction_only()
