#!/usr/bin/env python3
"""
Benchmark script to compare tag-based and atomic text-based emotion scenario matching systems.
Tests both systems on atomic scenarios for fair comparison.
"""

import json
import os
import time
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Import both routing systems
from tag_router import get_tag_router
from atomic_text_router import get_atomic_text_router


# Test cases using atomic scenario patterns
ATOMIC_TEST_CASES = [
    {
        "query": "My roommate keeps drinking my milk without asking",
        "expected_scenario": "personal_property_taken",
        "description": "Someone uses personal property without permission"
    },
    {
        "query": "I missed an important deadline for my project submission",
        "expected_scenario": "critical_obligation_missed", 
        "description": "Failed to fulfill time-sensitive duty"
    },
    {
        "query": "The train was delayed for 2 hours and I'm late for my meeting",
        "expected_scenario": "unexpected_delay",
        "description": "Unforeseen delay disrupts schedule"
    },
    {
        "query": "I can't find my car keys anywhere and I need to leave now",
        "expected_scenario": "essential_item_unavailable",
        "description": "Cannot access something needed immediately"
    },
    {
        "query": "My neighbor is playing loud music at 2 AM",
        "expected_scenario": "environmental_disruption",
        "description": "External conditions prevent rest"
    },
    {
        "query": "The WiFi went down right before my important video call",
        "expected_scenario": "unexpected_resource_failure",
        "description": "Technology stops working when required"
    },
    {
        "query": "My friend canceled our dinner plans at the last minute",
        "expected_scenario": "commitment_revoked",
        "description": "Agreed plan canceled at short notice"
    },
    {
        "query": "I tripped and fell in front of everyone at the conference",
        "expected_scenario": "public_mistake_exposure",
        "description": "Error becomes visible to others"
    },
    {
        "query": "My car broke down and the repair will cost $2000",
        "expected_scenario": "unforeseen_financial_burden",
        "description": "Unplanned cost strains budget"
    },
    {
        "query": "I've been calling my doctor all day but no one answers",
        "expected_scenario": "communication_breakdown",
        "description": "Cannot reach someone when urgent"
    },
    {
        "query": "My visa application was rejected without explanation",
        "expected_scenario": "authority_denial",
        "description": "Formal request rejected by official body"
    },
    {
        "query": "I have two meetings scheduled at the same time",
        "expected_scenario": "schedule_overlap",
        "description": "Commitments clash forcing decision"
    },
    {
        "query": "The job I wanted was filled before I could apply",
        "expected_scenario": "lost_opportunity",
        "description": "Beneficial chance passes before action"
    },
    {
        "query": "I followed wrong directions and ended up completely lost",
        "expected_scenario": "misinformation_received",
        "description": "Acting on incorrect information causes harm"
    },
    {
        "query": "We ran out of food supplies during the camping trip",
        "expected_scenario": "resource_shortage",
        "description": "Necessary supplies run out sooner than expected"
    },
    {
        "query": "I sprained my ankle and can't go hiking this weekend",
        "expected_scenario": "physical_injury",
        "description": "Minor injury limits planned activities"
    },
    {
        "query": "My friends made plans without inviting me",
        "expected_scenario": "social_exclusion",
        "description": "Left out of group activity that matters"
    },
    {
        "query": "Someone drove recklessly and almost hit me",
        "expected_scenario": "hazardous_behavior_from_other",
        "description": "Another person's reckless action endangers you"
    }
]


def benchmark_tag_system(test_cases: List[Dict]) -> Dict[str, Any]:
    """Benchmark the tag-based system."""
    print("🏷️  Benchmarking TAG-BASED system...")
    
    # Initialize tag router
    tag_router = get_tag_router()
    
    results = []
    total_time = 0
    correct_matches = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"  Testing {i+1}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        # Get top-5 matches from tag system
        top_matches = tag_router.route_top_k(test_case['query'], k=5)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Check if top match is correct
        top_match_id = top_matches[0]['scenario_id'] if top_matches else None
        is_correct = top_match_id == test_case['expected_scenario']
        if is_correct:
            correct_matches += 1
        
        # Get tags generated for this query
        tags_generated = getattr(tag_router, '_last_generated_tags', [])
        
        result = {
            "system_name": "TAG-BASED",
            "query": test_case['query'],
            "query_time": query_time,
            "top_match_id": top_match_id,
            "top_match_score": top_matches[0]['confidence'] if top_matches else 0.0,
            "tags_generated": tags_generated,
            "top_5_matches": top_matches,
            "expected_scenario": test_case['expected_scenario'],
            "test_case_description": test_case['description'],
            "is_correct": is_correct
        }
        
        results.append(result)
    
    # Calculate metrics
    accuracy = correct_matches / len(test_cases) if test_cases else 0
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    summary = {
        "system_name": "TAG-BASED",
        "total_queries": len(test_cases),
        "correct_matches": correct_matches,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_query": avg_time,
        "results": results
    }
    
    print(f"  ✅ Tag-based accuracy: {accuracy:.1%} ({correct_matches}/{len(test_cases)})")
    print(f"  ⏱️  Average time per query: {avg_time:.3f}s")
    
    return summary


def benchmark_atomic_text_system(test_cases: List[Dict]) -> Dict[str, Any]:
    """Benchmark the atomic text-based system."""
    print("📝 Benchmarking ATOMIC TEXT-BASED system...")
    
    # Initialize atomic text router
    atomic_router = get_atomic_text_router()
    
    results = []
    total_time = 0
    correct_matches = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"  Testing {i+1}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        # Get top-5 matches from atomic text system
        top_matches = atomic_router.route_top_k(test_case['query'], k=5)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Check if top match is correct
        top_match_id = top_matches[0]['scenario_id'] if top_matches else None
        is_correct = top_match_id == test_case['expected_scenario']
        if is_correct:
            correct_matches += 1
        
        result = {
            "system_name": "ATOMIC-TEXT-BASED",
            "query": test_case['query'],
            "query_time": query_time,
            "top_match_id": top_match_id,
            "top_match_score": top_matches[0]['confidence'] if top_matches else 0.0,
            "top_5_matches": top_matches,
            "expected_scenario": test_case['expected_scenario'],
            "test_case_description": test_case['description'],
            "is_correct": is_correct
        }
        
        results.append(result)
    
    # Calculate metrics
    accuracy = correct_matches / len(test_cases) if test_cases else 0
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    summary = {
        "system_name": "ATOMIC-TEXT-BASED", 
        "total_queries": len(test_cases),
        "correct_matches": correct_matches,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_query": avg_time,
        "results": results
    }
    
    print(f"  ✅ Atomic text-based accuracy: {accuracy:.1%} ({correct_matches}/{len(test_cases)})")
    print(f"  ⏱️  Average time per query: {avg_time:.3f}s")
    
    return summary


def save_benchmark_results(tag_results: Dict, text_results: Dict, output_file: str = None):
    """Save benchmark results to JSON file in logs/test directory."""
    # Create logs/test directory if it doesn't exist
    logs_dir = "logs/test"
    os.makedirs(logs_dir, exist_ok=True)
    
    if output_file is None:
        timestamp = int(time.time())
        output_file = f"atomic_benchmark_results_{timestamp}.json"
    
    # Ensure output file is in logs/test directory
    if not output_file.startswith(logs_dir):
        output_file = os.path.join(logs_dir, os.path.basename(output_file))
    
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "test_cases_count": len(ATOMIC_TEST_CASES),
        "benchmark_description": "Comparison of tag-based vs atomic text-based emotion scenario matching systems",
        "test_cases": [
            {
                "query": case["query"],
                "expected_scenario": case["expected_scenario"],
                "description": case["description"]
            }
            for case in ATOMIC_TEST_CASES
        ],
        "tag_system": tag_results,
        "atomic_text_system": text_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"📊 Benchmark results saved to: {output_file}")
    return output_file


def print_comparison_summary(tag_results: Dict, text_results: Dict):
    """Print a comparison summary of both systems."""
    print("\n" + "="*80)
    print("🏆 ATOMIC SCENARIOS BENCHMARK COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"  Tag-based System:        {tag_results['accuracy']:.1%} ({tag_results['correct_matches']}/{tag_results['total_queries']})")
    print(f"  Atomic Text-based System: {text_results['accuracy']:.1%} ({text_results['correct_matches']}/{text_results['total_queries']})")
    
    accuracy_diff = tag_results['accuracy'] - text_results['accuracy']
    if accuracy_diff > 0:
        print(f"  🥇 Tag-based system is {accuracy_diff:.1%} more accurate")
    elif accuracy_diff < 0:
        print(f"  🥇 Atomic text-based system is {abs(accuracy_diff):.1%} more accurate")
    else:
        print(f"  🤝 Both systems have equal accuracy")
    
    print(f"\n⏱️  PERFORMANCE COMPARISON:")
    print(f"  Tag-based System:        {tag_results['avg_time_per_query']:.3f}s per query")
    print(f"  Atomic Text-based System: {text_results['avg_time_per_query']:.3f}s per query")
    
    time_diff = tag_results['avg_time_per_query'] - text_results['avg_time_per_query']
    if time_diff > 0:
        print(f"  🚀 Atomic text-based system is {time_diff:.3f}s faster per query")
    elif time_diff < 0:
        print(f"  🚀 Tag-based system is {abs(time_diff):.3f}s faster per query")
    else:
        print(f"  ⚖️  Both systems have equal performance")
    
    print(f"\n🎯 KEY INSIGHTS:")
    
    # Find cases where systems disagree
    disagreements = 0
    for i, (tag_result, text_result) in enumerate(zip(tag_results['results'], text_results['results'])):
        if tag_result['top_match_id'] != text_result['top_match_id']:
            disagreements += 1
    
    print(f"  • Systems disagreed on {disagreements}/{len(ATOMIC_TEST_CASES)} test cases")
    print(f"  • Tag system uses structured semantic tags for matching")
    print(f"  • Text system uses direct text embedding similarity")
    
    if tag_results['accuracy'] > text_results['accuracy']:
        print(f"  • Tag-based approach shows better semantic understanding")
    elif text_results['accuracy'] > tag_results['accuracy']:
        print(f"  • Text-based approach shows better direct similarity matching")
    
    print("="*80)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark tag-based vs atomic text-based systems")
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--tag-only', action='store_true', help='Only benchmark tag system')
    parser.add_argument('--text-only', action='store_true', help='Only benchmark text system')
    args = parser.parse_args()
    
    print("🚀 Starting Atomic Scenarios Benchmark...")
    print(f"📋 Testing {len(ATOMIC_TEST_CASES)} atomic scenario test cases")
    print()
    
    tag_results = None
    text_results = None
    
    try:
        if not args.text_only:
            tag_results = benchmark_tag_system(ATOMIC_TEST_CASES)
            print()
        
        if not args.tag_only:
            text_results = benchmark_atomic_text_system(ATOMIC_TEST_CASES)
            print()
        
        # Save results
        if tag_results and text_results:
            output_file = save_benchmark_results(tag_results, text_results, args.output)
            print_comparison_summary(tag_results, text_results)
        elif tag_results:
            print(f"Tag-based system accuracy: {tag_results['accuracy']:.1%}")
        elif text_results:
            print(f"Atomic text-based system accuracy: {text_results['accuracy']:.1%}")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()

    test_cases = [
    {
        "query": "My roommate keeps drinking my milk from the fridge without asking",
        "expected_scenario_id": "roommate_milk",
        "description": "Roommate property violation test"
    },
    {
        "query": "I completely forgot about an important project deadline at work",
        "expected_scenario_id": "missed_deadline", 
        "description": "Work deadline test"
    },
    {
        "query": "Someone in the meeting keeps cutting me off when I try to speak",
        "expected_scenario_id": "someone_interrupts",
        "description": "Interruption test"
    },
    {
        "query": "My manager gave me harsh criticism that I don't think was deserved",
        "expected_scenario_id": "boss_unfair",
        "description": "Unfair criticism test"
    },
    {
        "query": "I was walking alone at night and heard footsteps behind me",
        "expected_scenario_id": "walk_dark_alone",
        "description": "Fear/threat test"
    },
    {
        "query": "I completely forgot my best friend's birthday yesterday",
        "expected_scenario_id": "forgot_birthday",
        "description": "Guilt/forgetfulness test"
    },
    {
        "query": "My colleague got the promotion I was hoping to get",
        "expected_scenario_id": "friend_promotion",
        "description": "Envy/jealousy test"
    },
    {
        "query": "I can't find my car keys and I'm already late for my appointment",
        "expected_scenario_id": "lost_keys",
        "description": "Stress/urgency test"
    },
    {
        "query": "A customer started yelling at me even though I was trying to help",
        "expected_scenario_id": "customer_yells",
        "description": "Unfair treatment test"
    },
    {
        "query": "My friend promised to help me move but cancelled last minute",
        "expected_scenario_id": "friend_breaks_promise",
        "description": "Broken promise test"
    },
    # Additional edge cases
    {
        "query": "Something really weird happened that doesn't fit any normal situation",
        "expected_scenario_id": "unsure_scenario",
        "description": "Edge case - should default to unsure"
    },
    {
        "query": "I'm feeling emotional but can't quite describe why",
        "expected_scenario_id": "unsure_scenario", 
        "description": "Vague input test"
    }
]

    return test_cases
    
    def benchmark_system(self, system_name: str, route_func, route_top_k_func) -> List[BenchmarkResult]:
        """Benchmark a specific routing system."""
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {system_name.upper()} SYSTEM")
        print(f"{'='*60}")
        
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n{i:2d}. {test_case['description']}")
            print(f"    Query: {test_case['query'][:60]}...")
            
            # Time the query
            start_time = time.time()
            
            try:
                # Get top-k matches
                top_matches = route_top_k_func(test_case['query'], k=5)
                
                end_time = time.time()
                query_time = end_time - start_time
                
                if top_matches:
                    top_match = top_matches[0]
                    top_match_id = top_match.get('scenario_id', top_match.get('id', 'unknown'))
                    top_match_score = top_match.get('confidence', top_match.get('score', 0.0))
                    
                    # Generate tags if this is the tag system
                    tags_generated = None
                    if system_name == "TAG-BASED":
                        try:
                            tags_generated = generate_tags(test_case['query'])
                        except Exception as e:
                            print(f"    Warning: Could not generate tags: {e}")
                    
                    result = BenchmarkResult(
                        system_name=system_name,
                        query=test_case['query'],
                        query_time=query_time,
                        top_match_id=top_match_id,
                        top_match_score=top_match_score,
                        top_5_matches=top_matches,
                        expected_scenario=test_case['expected_scenario_id'],
                        test_case_description=test_case['description'],
                        is_correct=top_match_id == test_case['expected_scenario_id'],
                        tags_generated=tags_generated
                    )
                    
                    # Check accuracy
                    accuracy_symbol = "✓" if result.is_correct else "✗"
                    
                    print(f"    Result: {accuracy_symbol} {top_match_id} (score: {top_match_score:.3f}, time: {query_time:.3f}s)")
                    if tags_generated:
                        print(f"    Tags: {tags_generated}")
                    
                    results.append(result)
                else:
                    print(f"    Result: ✗ No matches found (time: {query_time:.3f}s)")
                    
            except Exception as e:
                end_time = time.time()
                query_time = end_time - start_time
                print(f"    Error: {e} (time: {query_time:.3f}s)")
        
        return results
    
    def calculate_accuracy_metrics(self, results: List[BenchmarkResult]) -> Dict:
        """Calculate accuracy metrics for benchmark results."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Top-1 accuracy
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        
        for i, result in enumerate(results):
            expected_id = result.expected_scenario
            
            # Check top-1 accuracy
            if result.top_match_id == expected_id:
                correct_top1 += 1
            
            # Check top-3 and top-5 accuracy
            top_5_ids = [match.get('scenario_id', match.get('id', '')) for match in result.top_5_matches]
            if expected_id in top_5_ids[:3]:
                correct_top3 += 1
            if expected_id in top_5_ids:
                correct_top5 += 1
        
        total = len(results)
        
        return {
            "top_1_accuracy": correct_top1 / total,
            "top_3_accuracy": correct_top3 / total, 
            "top_5_accuracy": correct_top5 / total,
            "correct_top1": correct_top1,
            "correct_top3": correct_top3,
            "correct_top5": correct_top5,
            "total_queries": total
        }
    
    def generate_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics for benchmark results."""
        if not results:
            return None
        
        query_times = [r.query_time for r in results]
        accuracy_metrics = self.calculate_accuracy_metrics(results)
        
        return BenchmarkSummary(
            system_name=results[0].system_name,
            avg_query_time=statistics.mean(query_times),
            median_query_time=statistics.median(query_times),
            min_query_time=min(query_times),
            max_query_time=max(query_times),
            total_queries=len(results),
            accuracy_metrics=accuracy_metrics
        )
    
    def print_summary(self, summary: BenchmarkSummary):
        """Print a formatted summary of benchmark results."""
        print(f"\n{'='*60}")
        print(f"SUMMARY: {summary.system_name} SYSTEM")
        print(f"{'='*60}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Average Query Time: {summary.avg_query_time:.3f}s")
        print(f"   Median Query Time:  {summary.median_query_time:.3f}s")
        print(f"   Min Query Time:     {summary.min_query_time:.3f}s")
        print(f"   Max Query Time:     {summary.max_query_time:.3f}s")
        
        print(f"\n🎯 ACCURACY METRICS:")
        acc = summary.accuracy_metrics
        print(f"   Top-1 Accuracy: {acc['top_1_accuracy']:.1%} ({acc['correct_top1']}/{acc['total_queries']})")
        print(f"   Top-3 Accuracy: {acc['top_3_accuracy']:.1%} ({acc['correct_top3']}/{acc['total_queries']})")
        print(f"   Top-5 Accuracy: {acc['top_5_accuracy']:.1%} ({acc['correct_top5']}/{acc['total_queries']})")
    
    def compare_systems(self, text_summary: BenchmarkSummary, tag_summary: BenchmarkSummary):
        """Compare two systems and show improvements."""
        print(f"\n{'='*60}")
        print("SYSTEM COMPARISON")
        print(f"{'='*60}")
        
        # Performance comparison
        speed_improvement = (text_summary.avg_query_time - tag_summary.avg_query_time) / text_summary.avg_query_time
        print(f"\n⚡ SPEED COMPARISON:")
        print(f"   Text-based avg: {text_summary.avg_query_time:.3f}s")
        print(f"   Tag-based avg:  {tag_summary.avg_query_time:.3f}s")
        if speed_improvement > 0:
            print(f"   🚀 Tag system is {speed_improvement:.1%} faster")
        else:
            print(f"   🐌 Tag system is {abs(speed_improvement):.1%} slower")
        
        # Accuracy comparison
        text_acc = text_summary.accuracy_metrics['top_1_accuracy']
        tag_acc = tag_summary.accuracy_metrics['top_1_accuracy']
        acc_improvement = tag_acc - text_acc
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"   Text-based top-1: {text_acc:.1%}")
        print(f"   Tag-based top-1:  {tag_acc:.1%}")
        if acc_improvement > 0:
            print(f"   📈 Tag system is {acc_improvement:.1%} more accurate")
        elif acc_improvement < 0:
            print(f"   📉 Tag system is {abs(acc_improvement):.1%} less accurate")
        else:
            print(f"   ➖ Same accuracy")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 STARTING COMPREHENSIVE TAG SYSTEM BENCHMARK")
        print(f"Testing {len(self.test_cases)} scenarios...")
        
        results = {}
        summaries = {}
        
        # Benchmark tag-based system
        if TAG_SYSTEM_AVAILABLE:
            try:
                tag_results = self.benchmark_system("TAG-BASED", tag_route, tag_route_top_k)
                results["tag"] = tag_results
                summaries["tag"] = self.generate_summary(tag_results)
                self.print_summary(summaries["tag"])
            except Exception as e:
                print(f"Error benchmarking tag system: {e}")
        
        # Benchmark text-based system (if available)
        if TEXT_SYSTEM_AVAILABLE:
            try:
                text_results = self.benchmark_system("TEXT-BASED", text_route, text_route_top_k)
                results["text"] = text_results
                summaries["text"] = self.generate_summary(text_results)
                self.print_summary(summaries["text"])
            except Exception as e:
                print(f"Error benchmarking text system: {e}")
        
        # Compare systems
        if "text" in summaries and "tag" in summaries:
            self.compare_systems(summaries["text"], summaries["tag"])
        
        # Save detailed results
        self._save_benchmark_results(results, summaries)
        
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE! 🎉")
        print(f"{'='*60}")
        
        return results, summaries
    
    def _save_benchmark_results(self, results: Dict, summaries: Dict):
        """Save benchmark results to JSON files."""
        timestamp = int(time.time())
        
        # Save detailed results
        results_dir = 'logs/test'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = {}
            for system, system_results in results.items():
                serializable_results[system] = [
                    {
                        "system_name": result.system_name,
                        "query": result.query,
                        "query_time": result.query_time,
                        "top_match_id": result.top_match_id,
                        "top_match_score": result.top_match_score,
                        "top_5_matches": result.top_5_matches,
                        "expected_scenario": result.expected_scenario,
                        "test_case_description": result.test_case_description,
                        "is_correct": result.is_correct,
                        "tags_generated": result.tags_generated
                    }
                    for result in system_results
                ]
            json.dump(serializable_results, f, indent=2)
        
        # Save summaries
        summary_file = os.path.join(results_dir, f"benchmark_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            serializable_summaries = {}
            for system, summary in summaries.items():
                if summary:
                    serializable_summaries[system] = {
                        "system_name": summary.system_name,
                        "avg_query_time": summary.avg_query_time,
                        "median_query_time": summary.median_query_time,
                        "min_query_time": summary.min_query_time,
                        "max_query_time": summary.max_query_time,
                        "total_queries": summary.total_queries,
                        "accuracy_metrics": summary.accuracy_metrics
                    }
            json.dump(serializable_summaries, f, indent=2)
        
        print(f"\n📁 Results saved to:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary:  {summary_file}")

def main():
    """Main function to run the benchmark."""
    benchmark = TagSystemBenchmark()
    
    # Check system availability
    if not TAG_SYSTEM_AVAILABLE:
        print("❌ Tag-based system not available. Please build the tag index first:")
        print("   python build_tag_index.py")
        return
    
    if not TEXT_SYSTEM_AVAILABLE:
        print("⚠️  Text-based system not available for comparison")
        print("   (This is expected if you only want to test the tag system)")
    
    # Run the benchmark
    try:
        results, summaries = benchmark.run_full_benchmark()
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
