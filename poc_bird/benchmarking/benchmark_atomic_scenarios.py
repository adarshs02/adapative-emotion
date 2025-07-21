#!/usr/bin/env python3
"""
Benchmark script to compare tag-based and atomic text-based emotion scenario matching systems.
Tests both systems on atomic scenarios for fair comparison.
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path to import project modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Change to project root directory so relative paths work correctly
os.chdir(project_root)

# Import all routing systems
from tag_router import get_tag_router
from atomic_text_router import get_atomic_text_router
from hybrid_router import get_hybrid_router

# Import fine-tuned model components
import torch
import torch.nn.functional as F
import numpy as np
import hnswlib
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def check_and_build_indexes():
    """Check if required indexes exist and build them if missing."""
    print("🔍 Checking baseline system indexes...")
    
    # Check tag system index
    tag_index_path = "scenario_tags.idx"
    tag_mapping_path = "scenario_tags_mapping.json"
    
    if not os.path.exists(tag_index_path) or not os.path.exists(tag_mapping_path):
        print(f"❌ Tag system index missing. Building tag index...")
        try:
            # Import and run tag index builder
            import subprocess
            result = subprocess.run(['python', 'build_tag_index.py'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Tag index built successfully")
            else:
                print(f"⚠️  Tag index build failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not build tag index: {e}")
    else:
        print(f"✅ Tag index found")
    
    # Check atomic text system index
    text_index_path = "scenario_atomic_text.idx"
    text_mapping_path = "scenario_atomic_text_mapping.json"
    
    if not os.path.exists(text_index_path) or not os.path.exists(text_mapping_path):
        print(f"❌ Text system index missing. Building text index...")
        try:
            # Import and run text index builder
            import subprocess
            result = subprocess.run(['python', 'build_atomic_text_index.py'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Text index built successfully")
            else:
                print(f"⚠️  Text index build failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not build text index: {e}")
    else:
        print(f"✅ Text index found")
    
    print("")


class FineTunedEmbeddingModel:
    """Wrapper for the fine-tuned Llama embedding model for benchmarking."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.1-8B"):
        self.model_path = model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading fine-tuned model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModel.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        print(f"✅ Fine-tuned model loaded successfully!")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Test cases using variations of actual atomic scenarios for better evaluation
ATOMIC_TEST_CASES = [
    {
        "query": "My coworker borrowed my laptop charger and hasn't returned it",
        "expected_scenario": "personal_property_taken",
        "description": "Someone uses or removes your personal property without permission"
    },
    {
        "query": "I forgot to submit my tax return before the deadline",
        "expected_scenario": "critical_obligation_missed", 
        "description": "You fail to fulfill a time-sensitive duty or commitment"
    },
    {
        "query": "The bus broke down and I'm going to be late for my interview",
        "expected_scenario": "unexpected_delay",
        "description": "An unforeseen delay forces you to wait and disrupts your schedule"
    },
    {
        "query": "I can't find my passport and my flight leaves in 3 hours",
        "expected_scenario": "essential_item_unavailable",
        "description": "You cannot access or locate something you need immediately"
    },
    {
        "query": "Construction noise outside is making it impossible to concentrate",
        "expected_scenario": "environmental_disruption",
        "description": "External conditions prevent focus or rest"
    },
    {
        "query": "My laptop crashed during an important presentation",
        "expected_scenario": "unexpected_resource_failure",
        "description": "A tool, service, or piece of technology stops working when required"
    },
    {
        "query": "My date texted to cancel our plans 30 minutes before we were supposed to meet",
        "expected_scenario": "commitment_revoked",
        "description": "An agreed-upon plan or promise is canceled at short notice"
    },
    {
        "query": "I accidentally sent a private email to the entire company",
        "expected_scenario": "public_mistake_exposure",
        "description": "Your error or shortcoming becomes visible to others"
    },
    {
        "query": "My medical insurance denied coverage for an expensive procedure",
        "expected_scenario": "unforeseen_financial_burden",
        "description": "An unplanned cost arises that strains your budget"
    },
    {
        "query": "I've been trying to reach my lawyer all week but they won't return my calls",
        "expected_scenario": "communication_breakdown",
        "description": "You cannot reach someone or get a response when it is urgent to do so"
    },
    {
        "query": "The university rejected my graduate school application without giving reasons",
        "expected_scenario": "authority_denial",
        "description": "A formal request or application is unexpectedly rejected by an official body"
    },
    {
        "query": "My doctor's appointment conflicts with my job interview",
        "expected_scenario": "schedule_overlap",
        "description": "Two or more commitments clash, forcing an immediate decision or cancellation"
    },
    {
        "query": "The concert tickets sold out while I was still deciding whether to buy them",
        "expected_scenario": "lost_opportunity",
        "description": "A beneficial chance or offer passes before you can act on it"
    },
    {
        "query": "I followed the GPS directions but they led me to the wrong address",
        "expected_scenario": "misinformation_received",
        "description": "You act on incorrect information that later causes inconvenience or harm"
    },
    {
        "query": "The printer ran out of toner right before I needed to print my presentation",
        "expected_scenario": "resource_shortage",
        "description": "Necessary supplies or materials run out sooner than anticipated"
    },
    {
        "query": "I twisted my wrist and now I can't type for my work deadline",
        "expected_scenario": "physical_injury",
        "description": "A sudden minor injury or ailment limits your ability to carry out planned activities"
    },
    {
        "query": "My team didn't include me in the planning meeting for our project",
        "expected_scenario": "social_exclusion",
        "description": "You are left out of a group activity or decision that matters to you"
    },
    {
        "query": "A cyclist ran a red light and nearly crashed into me on the crosswalk",
        "expected_scenario": "hazardous_behavior_from_other",
        "description": "Another person's reckless action endangers or inconveniences you"
    },
    {
        "query": "Someone hacked into my email account and sent spam to all my contacts",
        "expected_scenario": "privacy_violation",
        "description": "Your personal or sensitive data is accessed or shared without consent"
    },
    {
        "query": "My boss assigned me three extra projects on top of my regular workload",
        "expected_scenario": "unexpected_responsibility",
        "description": "You are suddenly assigned additional duties beyond your regular workload"
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


def benchmark_finetuned_system(test_cases: List[Dict], model_path: str) -> Dict[str, Any]:
    """Benchmark the fine-tuned embedding model system."""
    print("🔥 Benchmarking FINE-TUNED EMBEDDING system...")
    
    # Load scenarios and fine-tuned model
    with open('atomic-scenarios_with_tags.json', 'r') as f:
        data = json.load(f)
        scenarios = data['scenarios'] if 'scenarios' in data else data
    
    finetuned_model = FineTunedEmbeddingModel(model_path)
    
    # Build scenario index
    print("  Building scenario embeddings index...")
    scenario_texts = [s['description'] for s in scenarios]
    scenario_embeddings = finetuned_model.encode(scenario_texts)
    
    # Create HNSW index
    dim = scenario_embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(scenario_embeddings), ef_construction=200, M=16)
    index.add_items(scenario_embeddings, list(range(len(scenario_embeddings))))
    index.set_ef(100)
    
    print(f"  Built index with {len(scenarios)} scenarios")
    
    results = []
    total_time = 0
    correct_matches = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"  Testing {i+1}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        # Get query embedding
        query_embedding = finetuned_model.encode([test_case['query']])
        
        # Search for top-5 matches
        labels, distances = index.knn_query(query_embedding, k=5)
        
        # Build top matches
        top_matches = []
        for j, idx in enumerate(labels[0]):
            scenario_id = scenarios[idx]['id']
            scenario_desc = scenarios[idx]['description']
            confidence = 1 - distances[0][j]  # Convert distance to similarity
            
            top_matches.append({
                'scenario_id': scenario_id,
                'description': scenario_desc,
                'confidence': float(confidence),
                'distance': float(distances[0][j])
            })
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Check if top match is correct
        top_match_id = top_matches[0]['scenario_id'] if top_matches else None
        is_correct = top_match_id == test_case['expected_scenario']
        if is_correct:
            correct_matches += 1
        
        # Store detailed results
        result = {
            'query': test_case['query'],
            'expected_scenario': test_case['expected_scenario'],
            'predicted_scenario': top_match_id,
            'top_5_matches': top_matches,
            'correct': is_correct,
            'response_time': query_time
        }
        results.append(result)
        
        # Print result for this query
        status = "✅" if is_correct else "❌"
        print(f"    {status} Expected: {test_case['expected_scenario']}, Got: {top_match_id}")
        if top_matches:
            print(f"    🏆 Top confidence: {top_matches[0]['confidence']:.4f}")
    
    # Calculate overall stats
    accuracy = correct_matches / len(test_cases)
    avg_time_per_query = total_time / len(test_cases)
    avg_confidence = np.mean([r['top_5_matches'][0]['confidence'] for r in results if r['top_5_matches']])
    
    benchmark_results = {
        'system_name': 'Fine-tuned Embedding',
        'model_path': model_path,
        'accuracy': accuracy,
        'correct_matches': correct_matches,
        'total_queries': len(test_cases),
        'avg_time_per_query': avg_time_per_query,
        'total_time': total_time,
        'avg_confidence': float(avg_confidence),
        'detailed_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"  📏 Results: {correct_matches}/{len(test_cases)} correct ({accuracy:.1%} accuracy)")
    print(f"  ⏱️  Average time per query: {avg_time_per_query:.3f}s")
    print(f"  🏆 Average confidence: {avg_confidence:.4f}")
    
    return benchmark_results


def benchmark_hybrid_system(test_cases: List[Dict], fusion_method: str = "weighted_sum", 
                           tag_weight: float = 0.5, text_weight: float = 0.5) -> Dict[str, Any]:
    """Benchmark the hybrid tag+text system."""
    print(f"🔀 Benchmarking HYBRID system (fusion={fusion_method}, tag_weight={tag_weight:.1f}, text_weight={text_weight:.1f})...")
    
    # Initialize hybrid router
    hybrid_router = get_hybrid_router(tag_weight, text_weight)
    
    results = []
    total_time = 0
    correct_matches = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"  Testing {i+1}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        # Get top-5 matches from hybrid system
        top_matches = hybrid_router.route_top_k(test_case['query'], k=5, fusion_method=fusion_method)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Check if top match is correct
        top_match_id = top_matches[0]['scenario_id'] if top_matches else None
        is_correct = top_match_id == test_case['expected_scenario']
        if is_correct:
            correct_matches += 1
        
        # Get tags generated for this query
        tags_generated = getattr(hybrid_router, '_last_generated_tags', [])
        
        result = {
            "system_name": f"HYBRID-{fusion_method.upper()}",
            "query": test_case['query'],
            "query_time": query_time,
            "top_match_id": top_match_id,
            "top_match_score": top_matches[0]['confidence'] if top_matches else 0.0,
            "tags_generated": tags_generated,
            "top_5_matches": top_matches,
            "expected_scenario": test_case['expected_scenario'],
            "test_case_description": test_case['description'],
            "is_correct": is_correct,
            "fusion_method": fusion_method,
            "tag_weight": tag_weight,
            "text_weight": text_weight
        }
        
        results.append(result)
    
    # Calculate metrics
    accuracy = correct_matches / len(test_cases) if test_cases else 0
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    summary = {
        "system_name": f"HYBRID-{fusion_method.upper()}",
        "total_queries": len(test_cases),
        "correct_matches": correct_matches,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_query": avg_time,
        "fusion_method": fusion_method,
        "tag_weight": tag_weight,
        "text_weight": text_weight,
        "results": results
    }
    
    print(f"  ✅ Hybrid {fusion_method} accuracy: {accuracy:.1%} ({correct_matches}/{len(test_cases)})")
    print(f"  ⏱️  Average time per query: {avg_time:.3f}s")
    
    return summary


def save_benchmark_results(tag_results: Dict = None, text_results: Dict = None, 
                          hybrid_results: Dict = None, finetuned_results: Dict = None, output_file: str = None):
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
        "benchmark_description": "Comparison of tag-based, text-based, hybrid, and fine-tuned emotion scenario matching systems",
        "test_cases": [
            {
                "query": case["query"],
                "expected_scenario": case["expected_scenario"],
                "description": case["description"]
            }
            for case in ATOMIC_TEST_CASES
        ]
    }
    
    # Add results for systems that were tested
    if tag_results:
        combined_results["tag_system"] = tag_results
    if text_results:
        combined_results["atomic_text_system"] = text_results
    if hybrid_results:
        combined_results["hybrid_system"] = hybrid_results
    if finetuned_results:
        combined_results["finetuned_system"] = finetuned_results
    
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


def print_three_way_comparison_summary(tag_results: Dict, text_results: Dict, hybrid_results: Dict):
    """Print a three-way comparison summary of tag, text, and hybrid systems."""
    print("\n" + "="*90)
    print("🏆 THREE-WAY BENCHMARK COMPARISON SUMMARY")
    print("="*90)
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"  Tag-based System:     {tag_results['accuracy']:.1%} ({tag_results['correct_matches']}/{tag_results['total_queries']})")
    print(f"  Text-based System:    {text_results['accuracy']:.1%} ({text_results['correct_matches']}/{text_results['total_queries']})")
    print(f"  Hybrid System:        {hybrid_results['accuracy']:.1%} ({hybrid_results['correct_matches']}/{hybrid_results['total_queries']})")
    
    # Find the best system
    accuracies = [
        ("Tag-based", tag_results['accuracy']),
        ("Text-based", text_results['accuracy']),
        ("Hybrid", hybrid_results['accuracy'])
    ]
    best_system, best_accuracy = max(accuracies, key=lambda x: x[1])
    
    print(f"  🎆 Best accuracy: {best_system} ({best_accuracy:.1%})")
    
    print(f"\n⏱️  PERFORMANCE COMPARISON:")
    print(f"  Tag-based System:     {tag_results['avg_time_per_query']:.3f}s per query")
    print(f"  Text-based System:    {text_results['avg_time_per_query']:.3f}s per query")
    print(f"  Hybrid System:        {hybrid_results['avg_time_per_query']:.3f}s per query")
    
    # Find the fastest system
    times = [
        ("Tag-based", tag_results['avg_time_per_query']),
        ("Text-based", text_results['avg_time_per_query']),
        ("Hybrid", hybrid_results['avg_time_per_query'])
    ]
    fastest_system, fastest_time = min(times, key=lambda x: x[1])
    
    print(f"  🚀 Fastest system: {fastest_system} ({fastest_time:.3f}s per query)")
    
    print(f"\n🔀 HYBRID SYSTEM DETAILS:")
    print(f"  Fusion method: {hybrid_results['fusion_method']}")
    print(f"  Tag weight: {hybrid_results['tag_weight']:.1f}")
    print(f"  Text weight: {hybrid_results['text_weight']:.1f}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    
    # Analyze agreement between systems
    tag_matches = {r['query']: r['top_match_id'] for r in tag_results['results']}
    text_matches = {r['query']: r['top_match_id'] for r in text_results['results']}
    hybrid_matches = {r['query']: r['top_match_id'] for r in hybrid_results['results']}
    
    tag_text_agreement = sum(1 for q in tag_matches if tag_matches[q] == text_matches[q])
    tag_hybrid_agreement = sum(1 for q in tag_matches if tag_matches[q] == hybrid_matches[q])
    text_hybrid_agreement = sum(1 for q in text_matches if text_matches[q] == hybrid_matches[q])
    
    total_queries = len(tag_matches)
    
    print(f"  • Tag-Text agreement: {tag_text_agreement}/{total_queries} ({tag_text_agreement/total_queries:.1%})")
    print(f"  • Tag-Hybrid agreement: {tag_hybrid_agreement}/{total_queries} ({tag_hybrid_agreement/total_queries:.1%})")
    print(f"  • Text-Hybrid agreement: {text_hybrid_agreement}/{total_queries} ({text_hybrid_agreement/total_queries:.1%})")
    
    # Performance vs accuracy trade-off analysis
    if hybrid_results['accuracy'] > max(tag_results['accuracy'], text_results['accuracy']):
        print(f"  • 🎆 Hybrid approach achieves best accuracy by combining both methods")
    elif hybrid_results['accuracy'] >= min(tag_results['accuracy'], text_results['accuracy']):
        print(f"  • 🔄 Hybrid approach provides balanced performance between tag and text methods")
    else:
        print(f"  • ⚠️  Hybrid approach underperforms compared to individual methods")
    
    # Speed analysis
    if hybrid_results['avg_time_per_query'] > max(tag_results['avg_time_per_query'], text_results['avg_time_per_query']):
        print(f"  • 🐢 Hybrid approach is slower due to running both systems")
    else:
        print(f"  • ⚡ Hybrid approach maintains competitive speed")
    
    print("="*90)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark tag-based, text-based, hybrid, and fine-tuned systems")
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--tag-only', action='store_true', help='Only benchmark tag system')
    parser.add_argument('--text-only', action='store_true', help='Only benchmark text system')
    parser.add_argument('--hybrid-only', action='store_true', help='Only benchmark hybrid system')
    parser.add_argument('--finetuned-only', action='store_true', help='Only benchmark fine-tuned system')
    parser.add_argument('--finetuned-model', type=str, help='Path to fine-tuned model for benchmarking')
    parser.add_argument('--build-indexes', action='store_true', help='Build missing indexes before benchmarking')
    parser.add_argument('--skip-index-check', action='store_true', help='Skip checking/building indexes')
    parser.add_argument('--fusion-method', type=str, default='weighted_sum', 
                       choices=['weighted_sum', 'max', 'min', 'rank_fusion'],
                       help='Fusion method for hybrid system')
    parser.add_argument('--tag-weight', type=float, default=0.5, help='Weight for tag scores in hybrid')
    parser.add_argument('--text-weight', type=float, default=0.5, help='Weight for text scores in hybrid')
    args = parser.parse_args()
    
    print("🚀 Starting Atomic Scenarios Benchmark...")
    print(f"📋 Testing {len(ATOMIC_TEST_CASES)} atomic scenario test cases")
    if args.hybrid_only or (not args.tag_only and not args.text_only and not args.finetuned_only):
        print(f"🔀 Hybrid fusion method: {args.fusion_method} (tag_weight={args.tag_weight}, text_weight={args.text_weight})")
    if args.finetuned_model:
        print(f"🔥 Fine-tuned model: {args.finetuned_model}")
    print()
    
    # Check and build indexes if needed (unless explicitly skipped)
    if not args.skip_index_check:
        if args.build_indexes:
            print("🔧 Force building indexes...")
        check_and_build_indexes()
    
    tag_results = None
    text_results = None
    hybrid_results = None
    finetuned_results = None
    
    try:
        # Run fine-tuned system if specified
        if args.finetuned_model and (args.finetuned_only or not any([args.tag_only, args.text_only, args.hybrid_only])):
            finetuned_results = benchmark_finetuned_system(ATOMIC_TEST_CASES, args.finetuned_model)
            print()
        
        # Run individual systems unless hybrid-only or finetuned-only
        if not args.hybrid_only and not args.finetuned_only:
            if not args.text_only and not args.finetuned_only:
                tag_results = benchmark_tag_system(ATOMIC_TEST_CASES)
                print()
            
            if not args.tag_only and not args.finetuned_only:
                text_results = benchmark_atomic_text_system(ATOMIC_TEST_CASES)
                print()
        
        # Run hybrid system unless individual-only or finetuned-only
        if (args.hybrid_only or (not args.tag_only and not args.text_only and not args.finetuned_only)) and not args.finetuned_only:
            hybrid_results = benchmark_hybrid_system(
                ATOMIC_TEST_CASES, 
                fusion_method=args.fusion_method,
                tag_weight=args.tag_weight,
                text_weight=args.text_weight
            )
            print()
        
        # Save results
        output_file = save_benchmark_results(tag_results, text_results, hybrid_results, finetuned_results, args.output)
        
        # Print summary based on what was tested
        if finetuned_results:
            print(f"🔥 Fine-tuned system accuracy: {finetuned_results['accuracy']:.1%} (avg confidence: {finetuned_results['avg_confidence']:.3f})")
            if tag_results or text_results or hybrid_results:
                print("\n📈 COMPARISON WITH BASELINE SYSTEMS:")
                if tag_results:
                    if tag_results['accuracy'] > 0:
                        improvement = ((finetuned_results['accuracy'] - tag_results['accuracy']) / tag_results['accuracy']) * 100
                        print(f"  vs Tag Router: {improvement:+.1f}% improvement")
                    else:
                        print(f"  vs Tag Router: N/A (baseline 0% accuracy)")
                if text_results:
                    if text_results['accuracy'] > 0:
                        improvement = ((finetuned_results['accuracy'] - text_results['accuracy']) / text_results['accuracy']) * 100
                        print(f"  vs Text Router: {improvement:+.1f}% improvement")
                    else:
                        print(f"  vs Text Router: N/A (baseline 0% accuracy)")
                if hybrid_results:
                    if hybrid_results['accuracy'] > 0:
                        improvement = ((finetuned_results['accuracy'] - hybrid_results['accuracy']) / hybrid_results['accuracy']) * 100
                        print(f"  vs Hybrid Router: {improvement:+.1f}% improvement")
                    else:
                        print(f"  vs Hybrid Router: N/A (baseline 0% accuracy)")
        elif tag_results and text_results and hybrid_results:
            print_three_way_comparison_summary(tag_results, text_results, hybrid_results)
        elif tag_results and text_results:
            print_comparison_summary(tag_results, text_results)
        elif hybrid_results:
            print(f"Hybrid {args.fusion_method} system accuracy: {hybrid_results['accuracy']:.1%}")
        elif tag_results:
            print(f"Tag-based system accuracy: {tag_results['accuracy']:.1%}")
        elif text_results:
            print(f"Atomic text-based system accuracy: {text_results['accuracy']:.1%}")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
