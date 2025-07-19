"""
Benchmark script to test joint embedding router performance.
"""

import time
from typing import Dict, List, Any
import json
from datetime import datetime
import os

from joint_embedding_router import get_joint_embedding_router


# Same test cases as before for comparison
ATOMIC_TEST_CASES = [
    {
        "query": "My friend borrowed my car and returned it with a huge dent",
        "expected_scenario": "property_damage_by_friend",
        "description": "Friend damages borrowed property"
    },
    {
        "query": "I found out my partner has been lying to me about where they go at night",
        "expected_scenario": "romantic_partner_deception", 
        "description": "Romantic partner deception discovery"
    },
    {
        "query": "My coworker took credit for the project I worked on all month",
        "expected_scenario": "work_credit_theft",
        "description": "Colleague steals work credit"
    },
    {
        "query": "Someone at the store was very rude to me for no reason",
        "expected_scenario": "stranger_rudeness",
        "description": "Unprovoked rudeness from stranger"
    },
    {
        "query": "I accidentally sent a private message to the wrong person",
        "expected_scenario": "accidental_message_error",
        "description": "Accidental message to wrong recipient"
    },
    {
        "query": "My neighbor's dog keeps barking all night and waking me up",
        "expected_scenario": "neighbor_noise_disturbance",
        "description": "Neighbor's pet causing noise issues"
    },
    {
        "query": "I was passed over for a promotion that I clearly deserved",
        "expected_scenario": "unfair_promotion_denial",
        "description": "Deserved promotion unfairly denied"
    },
    {
        "query": "My family member shared my personal secret with others",
        "expected_scenario": "family_privacy_breach",
        "description": "Family member violates privacy"
    },
    {
        "query": "The restaurant served me food that made me sick",
        "expected_scenario": "food_poisoning_incident",
        "description": "Restaurant serves contaminated food"
    },
    {
        "query": "I lost an important document right before a deadline",
        "expected_scenario": "important_document_loss",
        "description": "Critical document lost before deadline"
    },
    {
        "query": "My roommate ate my food without asking",
        "expected_scenario": "roommate_food_theft",
        "description": "Roommate takes food without permission"
    },
    {
        "query": "I was stuck in traffic and missed an important meeting",
        "expected_scenario": "traffic_delay_meeting",
        "description": "Traffic causes missed meeting"
    },
    {
        "query": "My phone died right when I needed to make an urgent call",
        "expected_scenario": "phone_battery_emergency",
        "description": "Phone dies during emergency"
    },
    {
        "query": "Someone cut in line in front of me at the store",
        "expected_scenario": "line_cutting_incident",
        "description": "Person cuts in line"
    },
    {
        "query": "My computer crashed and I lost hours of work",
        "expected_scenario": "computer_crash_work_loss",
        "description": "Computer crash causes work loss"
    },
    {
        "query": "I found out someone was spreading rumors about me",
        "expected_scenario": "rumor_spreading_incident",
        "description": "Someone spreads rumors"
    },
    {
        "query": "My flight was cancelled at the last minute",
        "expected_scenario": "flight_cancellation_lastminute",
        "description": "Last-minute flight cancellation"
    },
    {
        "query": "Someone took my parking spot that I was waiting for",
        "expected_scenario": "parking_spot_theft",
        "description": "Parking spot taken while waiting"
    },
    {
        "query": "Someone hacked into my email account and sent spam",
        "expected_scenario": "email_account_breach",
        "description": "Email account hacked"
    },
    {
        "query": "My boss assigned me three extra projects on top of my current workload",
        "expected_scenario": "unexpected_responsibility",
        "description": "Boss assigns excessive work"
    }
]


def benchmark_joint_embedding_system(test_cases: List[Dict]) -> Dict[str, Any]:
    """Benchmark the joint embedding system."""
    print("🔗 Benchmarking JOINT EMBEDDING system...")
    
    # Initialize joint embedding router
    joint_router = get_joint_embedding_router()
    
    results = []
    total_time = 0
    correct_matches = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"  Testing {i+1}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        # Get top-5 matches from joint embedding system
        top_matches = joint_router.route_top_k(test_case['query'], k=5)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Check if top match is correct
        top_match_id = top_matches[0]['scenario_id'] if top_matches else None
        is_correct = top_match_id == test_case['expected_scenario']
        if is_correct:
            correct_matches += 1
        
        # Get tags generated for this query
        input_tags = getattr(joint_router, '_last_generated_tags', [])
        
        # Convert numpy types to Python types for JSON serialization
        top_matches_serializable = []
        for match in top_matches:
            serializable_match = {
                'scenario_id': match['scenario_id'],
                'description': match['description'],
                'confidence': float(match['confidence']),
                'score': float(match['score']),
                'distance': float(match.get('distance', 0.0)),
                'input_tags': match.get('input_tags', []),
                'joint_representation': match.get('joint_representation', ''),
                'scenario': match.get('scenario', {})
            }
            top_matches_serializable.append(serializable_match)
        
        result = {
            "system_name": "JOINT-EMBEDDING",
            "query": test_case['query'],
            "query_time": float(query_time),
            "top_match_id": top_match_id,
            "top_match_score": float(top_matches[0]['confidence']) if top_matches else 0.0,
            "input_tags": input_tags,
            "top_5_matches": top_matches_serializable,
            "expected_scenario": test_case['expected_scenario'],
            "test_case_description": test_case['description'],
            "is_correct": bool(is_correct)
        }
        
        results.append(result)
        
        # Print immediate feedback
        status = "✅" if is_correct else "❌"
        print(f"    {status} Expected: {test_case['expected_scenario']}, Got: {top_match_id}")
        if input_tags:
            print(f"    🏷️  Generated tags: {input_tags}")
    
    # Calculate metrics
    accuracy = correct_matches / len(test_cases) if test_cases else 0
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    summary = {
        "system_name": "JOINT-EMBEDDING",
        "total_queries": len(test_cases),
        "correct_matches": correct_matches,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_query": avg_time,
        "results": results
    }
    
    print(f"  ✅ Joint embedding accuracy: {accuracy:.1%} ({correct_matches}/{len(test_cases)})")
    print(f"  ⏱️  Average time per query: {avg_time:.3f}s")
    
    return summary


def main():
    """Run joint embedding benchmark."""
    print("🔗 Testing Joint Embedding Performance")
    print("=" * 50)
    
    # Ensure logs directory exists
    os.makedirs("logs/test", exist_ok=True)
    
    # Run benchmark
    results = benchmark_joint_embedding_system(ATOMIC_TEST_CASES)
    
    # Save results
    output_file = f"logs/test/joint_embedding_benchmark_{int(time.time())}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "description": "Benchmark of joint embedding system (scenario description + tags)",
            "test_cases_count": len(ATOMIC_TEST_CASES),
            "joint_embedding_system": results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("🏆 JOINT EMBEDDING SYSTEM PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"Speed: {results['avg_time_per_query']:.3f}s per query")
    print(f"Total correct: {results['correct_matches']}/{results['total_queries']}")
    
    # Show some example joint representations
    print("\n📝 Example Joint Representations:")
    print("-" * 30)
    for i, result in enumerate(results['results'][:3]):
        if 'top_5_matches' in result and result['top_5_matches']:
            joint_repr = result['top_5_matches'][0].get('joint_representation', '')
            if joint_repr:
                print(f"{i+1}. Query: {result['query'][:60]}...")
                print(f"   Joint: {joint_repr[:120]}...")
                print()


if __name__ == "__main__":
    main()
