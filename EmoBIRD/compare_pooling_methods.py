"""
Compare BIRD pooling vs Direct LLM assessment approaches.
"""

import json
from typing import Dict, Any
import sys
import atexit

from emobird_poc import Emobird as EmobirdWithPooling
from emobird_no_pooling import EmobirdNoPooling
from utils import print_gpu_info


def cleanup_resources():
    """Clean up distributed computing resources"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass


def compare_approaches(user_situation: str) -> None:
    """
    Compare BIRD pooling vs direct assessment on the same situation.
    """
    print("\n" + "="*80)
    print(f"SITUATION: {user_situation}")
    print("="*80)
    
    # Initialize both systems
    print("\n🔄 Initializing systems...")
    pooling_system = EmobirdWithPooling()
    pooling_system.verbose = False  # Reduce output noise
    
    direct_system = EmobirdNoPooling()
    direct_system.verbose = False
    
    # Run analysis with BIRD pooling
    print("\n1️⃣ WITH BIRD POOLING:")
    print("-" * 40)
    pooling_result = pooling_system.analyze_emotion(user_situation)
    
    if 'emotions' in pooling_result and pooling_result['emotions']:
        sorted_pooling = sorted(pooling_result['emotions'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_pooling[:5]:
            bar = '█' * int(prob * 30)
            print(f"  {emotion:12} {prob:.3f} {bar}")
    
    # Run analysis without pooling
    print("\n2️⃣ DIRECT LLM ASSESSMENT (No Pooling):")
    print("-" * 40)
    direct_result = direct_system.analyze_emotion(user_situation)
    
    if 'emotions' in direct_result and direct_result['emotions']:
        sorted_direct = sorted(direct_result['emotions'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_direct[:5]:
            bar = '░' * int(prob * 30)
            print(f"  {emotion:12} {prob:.3f} {bar}")
    
    # Compare top emotions
    print("\n📊 COMPARISON:")
    print("-" * 40)
    
    if 'top_emotions' in pooling_result and 'top_emotions' in direct_result:
        pooling_top = list(pooling_result['top_emotions'].keys())[:3]
        direct_top = list(direct_result['top_emotions'].keys())[:3]
        
        print(f"Top 3 (Pooling):   {', '.join(pooling_top)}")
        print(f"Top 3 (Direct):    {', '.join(direct_top)}")
        
        # Check agreement
        agreement = len(set(pooling_top) & set(direct_top))
        print(f"\nAgreement: {agreement}/3 emotions in common")
        
        # Show factors used
        if 'factors' in pooling_result:
            print(f"\nFactors identified:")
            for factor, value in pooling_result['factors'].items():
                print(f"  - {factor}: {value}")
    
    # Show explanation if available
    if 'explanation' in direct_result:
        print(f"\n💭 Direct Assessment Explanation:")
        print(f"  {direct_result['explanation']}")


def main():
    # Register cleanup
    atexit.register(cleanup_resources)
    
    print_gpu_info()
    
    # Test scenarios covering different emotional contexts
    test_scenarios = [
        {
            "name": "Success & Joy",
            "situation": "I just got accepted into my dream university after years of hard work and preparation."
        },
        {
            "name": "Betrayal & Anger",
            "situation": "My business partner secretly started a competing company using our confidential information."
        },
        {
            "name": "Loss & Sadness",
            "situation": "My childhood pet passed away yesterday after being sick for several weeks."
        },
        {
            "name": "Mixed Emotions",
            "situation": "I got a great job offer but it means moving away from my family and friends."
        },
        {
            "name": "Anxiety & Stress",
            "situation": "I have three major deadlines tomorrow and I haven't started any of them yet."
        }
    ]
    
    print("\n" + "="*80)
    print("COMPARING BIRD POOLING VS DIRECT LLM ASSESSMENT")
    print("="*80)
    
    mode = input("\n1. Run all test scenarios\n2. Enter custom situation\nChoice (1/2): ")
    
    if mode == "1":
        for scenario in test_scenarios:
            print(f"\n\n🎭 Scenario: {scenario['name']}")
            compare_approaches(scenario['situation'])
            input("\nPress Enter to continue...")
    else:
        user_situation = input("\nEnter your situation: ")
        compare_approaches(user_situation)
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("-" * 40)
    print("• BIRD Pooling: Amplifies consensus, can produce extreme probabilities")
    print("• Direct Assessment: More moderate, considers factors holistically")
    print("• Direct Assessment: Simpler pipeline with fewer transformations")
    print("• Both approaches identify similar top emotions in most cases")
    print("="*80)


if __name__ == "__main__":
    main()
