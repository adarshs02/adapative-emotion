#!/usr/bin/env python3
"""
Test script to demonstrate the tag-based emotion scenario matching system.
This script tests tag generation and shows how tags improve matching interpretability.
"""

import json
import os
from tag_generator import generate_tags

def test_tag_generation():
    """Test tag generation for various scenarios and user inputs."""
    
    print("=" * 60)
    print("TESTING TAG GENERATION SYSTEM")
    print("=" * 60)
    
    # Test scenarios from the scenarios.json file
    test_scenarios = [
        "My roommate drank my milk again.",
        "I missed an important deadline at work.",
        "Someone keeps interrupting you while you are speaking.",
        "Your boss criticizes your work unfairly.",
        "Walking alone at night and hearing footsteps behind you.",
        "You forgot your best friend's birthday.",
        "Your friend gets promoted while you are overlooked."
    ]
    
    print("\n1. TESTING SCENARIO TAG GENERATION")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Scenario: {scenario}")
        try:
            tags = generate_tags(scenario)
            print(f"   Generated Tags: {tags}")
        except Exception as e:
            print(f"   Error generating tags: {e}")
    
    print("\n\n2. TESTING USER INPUT TAG GENERATION")
    print("-" * 40)
    
    # Test user inputs that should match the scenarios above
    test_user_inputs = [
        "I'm so frustrated because my flatmate keeps taking my food from the fridge",
        "I completely forgot about an important project deadline and now my boss is angry",
        "This person in the meeting won't let me finish my sentences and keeps cutting me off",
        "My manager gave me negative feedback that I don't think was fair or accurate",
        "I was walking home late and someone seemed to be following me, I felt scared",
        "I feel terrible because I completely forgot it was my best friend's birthday yesterday",
        "I'm feeling jealous because my colleague got the promotion I was hoping for"
    ]
    
    for i, user_input in enumerate(test_user_inputs, 1):
        print(f"\n{i}. User Input: {user_input}")
        try:
            tags = generate_tags(user_input)
            print(f"   Generated Tags: {tags}")
        except Exception as e:
            print(f"   Error generating tags: {e}")
    
    print("\n\n3. COMPARING TAGS FOR SIMILAR SITUATIONS")
    print("-" * 40)
    
    # Show how similar situations get similar tags
    similar_pairs = [
        ("My roommate drank my milk again.", 
         "I'm so frustrated because my flatmate keeps taking my food from the fridge"),
        ("Your boss criticizes your work unfairly.", 
         "My manager gave me negative feedback that I don't think was fair or accurate"),
        ("Walking alone at night and hearing footsteps behind you.", 
         "I was walking home late and someone seemed to be following me, I felt scared")
    ]
    
    for scenario, user_input in similar_pairs:
        print(f"\nScenario: {scenario}")
        scenario_tags = generate_tags(scenario)
        print(f"Scenario Tags: {scenario_tags}")
        
        print(f"User Input: {user_input}")
        user_tags = generate_tags(user_input)
        print(f"User Tags: {user_tags}")
        
        # Calculate tag overlap
        common_tags = set(scenario_tags) & set(user_tags)
        print(f"Common Tags: {list(common_tags)} ({len(common_tags)}/{max(len(scenario_tags), len(user_tags))} overlap)")
        print("-" * 60)

def show_tag_vocabulary():
    """Display the core tag vocabulary being used."""
    from tag_generator import CORE_TAG_VOCABULARY
    
    print("\n\n4. CORE TAG VOCABULARY")
    print("-" * 40)
    
    for category, tags in CORE_TAG_VOCABULARY.items():
        print(f"\n{category.upper()}:")
        for tag in tags:
            print(f"  - {tag}")

def main():
    """Main function to run tag system tests."""
    try:
        test_tag_generation()
        show_tag_vocabulary()
        
        print("\n" + "=" * 60)
        print("TAG GENERATION TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running tag system test: {e}")
        print("\nMake sure you have:")
        print("1. Installed all required dependencies")
        print("2. Set up the LLM model configuration correctly")
        print("3. Have sufficient GPU memory available")

if __name__ == "__main__":
    main()
