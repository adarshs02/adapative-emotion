#!/usr/bin/env python3
"""
Example usage of the Emobird system.

This script demonstrates how to use the dynamic emotion analysis system.
"""

import sys
import os
import json

# Add the parent directory to the path so we can import emobird modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emobird import Emobird
from utils import print_gpu_info


def main():
    """Run example emotion analysis."""
    
    print("🐦 Emobird Dynamic Emotion Analysis System")
    print("=" * 50)
    
    try:
        # Initialize Emobird system
        print("Initializing system...")
        print_gpu_info()
        emobird = Emobird()
        
        # Load example situations from examples.json
        try:
            with open('examples.json', 'r') as f:
                examples_data = json.load(f)
            example_situations = examples_data['examples']
            print(f"📂 Loaded {len(example_situations)} example situations from examples.json")
        except Exception as e:
            print(f"⚠️ Failed to load examples.json: {e}")
            print("Using fallback example situations...")
            example_situations = [
                "I just got promoted at work after working really hard for months",
                "My best friend forgot my birthday and didn't even call me"
            ]
        
        # Automatically run all example situations
        print("\n🧪 Running all example situations automatically...")
        choice = "2"  # Automatically choose to run examples
        
        if choice == "1":
            # Custom situation
            user_situation = input("\nPlease describe your situation: ").strip()
            if user_situation:
                analyze_situation(emobird, user_situation)
            else:
                print("No situation provided.")
                
        elif choice == "2":
            # Run examples
            print("\n🧪 Running example situations...")
            for i, situation in enumerate(example_situations, 1):
                print(f"\n--- Example {i}/{len(example_situations)} ---")
                analyze_situation(emobird, situation)
                
                if i < len(example_situations):
                    print("\n" + "="*60 + "\n")  # Just add a separator line
                    
        elif choice == "3":
            print("Goodbye! 🐦")
            return
        else:
            print("Invalid choice. Please run the script again.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis might be due to missing dependencies or model issues.")
        print("Make sure you have installed the requirements and have access to the specified model.")


def analyze_situation(emobird, situation):
    """Analyze a single situation and display results."""
    
    print(f"\n📝 Situation: '{situation}'")
    print("-" * 60)
    
    try:
        # Analyze the situation
        result = emobird.analyze_emotion(situation)
        
        # Display results
        print(f"\n🎭 Generated Scenario:")
        scenario = result.get('scenario', {})
        print(f"  ID: {scenario.get('id', 'N/A')}")
        print(f"  Description: {scenario.get('description', 'N/A')}")
        print(f"  Tags: {', '.join(scenario.get('tags', []))}")
        
        print(f"\n⚙️ Factor Values:")
        factors = result.get('factors', {})
        if factors:
            for factor, value in factors.items():
                print(f"  - {factor}: {value}")
        else:
            print("  No factors extracted")
        
        print(f"\n😊 Emotion Probabilities:")
        emotions = result.get('emotions', {})
        if emotions:
            # Sort by probability (highest first)
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_emotions:
                # Create a simple bar visualization
                bar_length = int(prob * 20)  # Scale to 20 characters
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {emotion:8}: {prob:.3f} [{bar}]")
        else:
            print("  No emotion probabilities calculated")
            
        # Show metadata
        metadata = result.get('metadata', {})
        print(f"\n📊 Metadata:")
        print(f"  Method: {metadata.get('inference_method', 'N/A')}")
        print(f"  Model: {metadata.get('model_used', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This could be due to model loading issues or generation failures.")


if __name__ == "__main__":
    main()
