"""
Test script for Factor and Emotion Generators

This script tests the new emotion-focused flow:
1. EmotionGenerator - extracts 2-4 crucial emotions from situations
2. FactorGenerator - generates psychological factors from situations
3. Integration testing of both generators working together
"""

import sys
import os
import json
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_generator import FactorGenerator
from emotion_generator import EmotionGenerator
from scenario_generator import ScenarioGenerator
from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper


class GeneratorTester:
    """Test harness for factor and emotion generators."""
    
    def __init__(self):
        """Initialize the test environment."""
        print("🧪 Initializing Generator Tester...")
        
        # Load configuration
        self.config = EmobirdConfig()
        
        # Initialize vLLM wrapper
        print("🚀 Loading vLLM...")
        self.vllm_wrapper = VLLMWrapper(self.config)
        
        # Initialize generators
        self.factor_generator = FactorGenerator(self.config)
        self.emotion_generator = EmotionGenerator(self.config)
        self.scenario_generator = ScenarioGenerator(self.config)
        
        # Set vLLM for generators
        self.factor_generator.set_vllm(self.vllm_wrapper)
        self.emotion_generator.set_vllm(self.vllm_wrapper)
        self.scenario_generator.set_vllm(self.vllm_wrapper)
        
        print("✅ Test environment initialized!")
    
    def test_emotion_generator(self, situations: List[str]) -> Dict[str, Any]:
        """Test the emotion generator with various situations."""
        print("\n" + "="*60)
        print("🎭 TESTING EMOTION GENERATOR")
        print("="*60)
        
        results = {}
        
        for i, situation in enumerate(situations, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Situation: {situation}")
            
            try:
                # Extract crucial emotions
                crucial_emotions = self.emotion_generator.extract_crucial_emotions(situation)
                
                # Validate emotions
                is_valid = self.emotion_generator.validate_emotions(crucial_emotions)
                
                results[f"case_{i}"] = {
                    'situation': situation,
                    'crucial_emotions': crucial_emotions,
                    'num_emotions': len(crucial_emotions),
                    'is_valid': is_valid,
                    'success': True
                }
                
                print(f"✅ Crucial Emotions: {crucial_emotions}")
                print(f"✅ Count: {len(crucial_emotions)} (valid: {is_valid})")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[f"case_{i}"] = {
                    'situation': situation,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def test_factor_generator(self, situations: List[str]) -> Dict[str, Any]:
        """Test the factor generator with various situations."""
        print("\n" + "="*60)
        print("⚙️ TESTING FACTOR GENERATOR")
        print("="*60)
        
        results = {}
        
        for i, situation in enumerate(situations, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Situation: {situation}")
            
            try:
                # Test if the new methods exist, fallback to old methods if needed
                if hasattr(self.factor_generator, 'generate_factors_from_situation'):
                    factors = self.factor_generator.generate_factors_from_situation(situation)
                else:
                    # Fallback to old method with empty abstract
                    print("⚠️ Using fallback method (generate_factors)")
                    factors = self.factor_generator.generate_factors(situation, "")
                
                if hasattr(self.factor_generator, 'extract_factor_values_direct'):
                    factor_values = self.factor_generator.extract_factor_values_direct(situation, factors)
                else:
                    # Fallback to old method
                    print("⚠️ Using fallback method (extract_factor_values)")
                    factor_values = self.factor_generator.extract_factor_values(situation, "", factors)
                
                results[f"case_{i}"] = {
                    'situation': situation,
                    'factors': factors,
                    'factor_values': factor_values,
                    'num_factors': len(factors) if factors else 0,
                    'success': True
                }
                
                print(f"✅ Generated {len(factors) if factors else 0} factors")
                print(f"✅ Factor values: {factor_values}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[f"case_{i}"] = {
                    'situation': situation,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def test_integration(self, situations: List[str]) -> Dict[str, Any]:
        """Test both generators working together."""
        print("\n" + "="*60)
        print("🔗 TESTING GENERATOR INTEGRATION")
        print("="*60)
        
        results = {}
        
        for i, situation in enumerate(situations, 1):
            print(f"\n--- Integration Test {i} ---")
            print(f"Situation: {situation}")
            
            try:
                # Step 1: Extract crucial emotions
                crucial_emotions = self.emotion_generator.extract_crucial_emotions(situation)
                
                # Step 2: Generate factors
                if hasattr(self.factor_generator, 'generate_factors_from_situation'):
                    factors = self.factor_generator.generate_factors_from_situation(situation)
                else:
                    factors = self.factor_generator.generate_factors(situation, "")
                
                # Step 3: Extract factor values
                if hasattr(self.factor_generator, 'extract_factor_values_direct'):
                    factor_values = self.factor_generator.extract_factor_values_direct(situation, factors)
                else:
                    factor_values = self.factor_generator.extract_factor_values(situation, "", factors)
                
                # Analyze the combination
                emotion_factor_mapping = self._analyze_emotion_factor_relationship(
                    crucial_emotions, factors, factor_values
                )
                
                results[f"integration_{i}"] = {
                    'situation': situation,
                    'crucial_emotions': crucial_emotions,
                    'factors': factors,
                    'factor_values': factor_values,
                    'emotion_factor_mapping': emotion_factor_mapping,
                    'success': True
                }
                
                print(f"✅ Emotions: {crucial_emotions}")
                print(f"✅ Factors: {len(factors) if factors else 0}")
                print(f"✅ Integration successful")
                
            except Exception as e:
                print(f"❌ Integration Error: {str(e)}")
                results[f"integration_{i}"] = {
                    'situation': situation,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def test_emotion_comparison_abstract_vs_full(self, situations: List[str]) -> Dict[str, Any]:
        """Test emotion extraction: abstracts vs full user prompts."""
        print("\n" + "="*60)
        print("🔄 TESTING EMOTION EXTRACTION: ABSTRACT VS FULL PROMPT")
        print("="*60)
        
        results = {}
        
        for i, situation in enumerate(situations, 1):
            print(f"\n--- Comparison Test {i} ---")
            print(f"Situation: {situation[:100]}...")
            
            try:
                # Step 1: Generate abstract from user situation
                print("📋 Generating abstract...")
                abstract = self.scenario_generator._generate_abstract(situation)
                print(f"   Abstract: {abstract[:100]}...")
                
                # Step 2: Extract emotions from full user situation
                print("🎭 Extracting emotions from FULL SITUATION...")
                emotions_from_full = self.emotion_generator.extract_crucial_emotions(situation)
                
                # Step 3: Extract emotions from abstract
                print("🎭 Extracting emotions from ABSTRACT...")
                emotions_from_abstract = self.emotion_generator.extract_crucial_emotions_from_abstract(abstract)
                
                # Step 4: Compare results
                comparison_analysis = self._analyze_emotion_differences(
                    emotions_from_full, emotions_from_abstract
                )
                
                results[f"comparison_{i}"] = {
                    'situation': situation,
                    'abstract': abstract,
                    'emotions_from_full': emotions_from_full,
                    'emotions_from_abstract': emotions_from_abstract,
                    'comparison': comparison_analysis,
                    'success': True
                }
                
                print(f"✅ Full Situation Emotions: {emotions_from_full}")
                print(f"✅ Abstract Emotions: {emotions_from_abstract}")
                print(f"✅ Match: {comparison_analysis['exact_match']} | Overlap: {comparison_analysis['overlap_count']}/{comparison_analysis['total_unique']}")
                
            except Exception as e:
                print(f"❌ Comparison Error: {str(e)}")
                results[f"comparison_{i}"] = {
                    'situation': situation,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _analyze_emotion_differences(self, emotions_full: List[str], 
                                   emotions_abstract: List[str]) -> Dict[str, Any]:
        """Analyze differences between emotion sets."""
        set_full = set(emotions_full)
        set_abstract = set(emotions_abstract)
        
        overlap = set_full.intersection(set_abstract)
        only_in_full = set_full - set_abstract
        only_in_abstract = set_abstract - set_full
        total_unique = len(set_full.union(set_abstract))
        
        return {
            'exact_match': set_full == set_abstract,
            'overlap_emotions': list(overlap),
            'overlap_count': len(overlap),
            'only_in_full': list(only_in_full),
            'only_in_abstract': list(only_in_abstract),
            'total_unique': total_unique,
            'similarity_score': len(overlap) / total_unique if total_unique > 0 else 0
        }
    
    def _analyze_emotion_factor_relationship(self, emotions: List[str], 
                                           factors: List[Dict], 
                                           factor_values: Dict[str, str]) -> Dict[str, Any]:
        """Analyze the relationship between emotions and factors."""
        return {
            'emotion_count': len(emotions),
            'factor_count': len(factors) if factors else 0,
            'factor_value_count': len(factor_values),
            'potential_combinations': len(emotions) * len(factor_values) if factor_values else 0,
            'dominant_emotion': emotions[0] if emotions else None,
            'sample_factors': [f.get('name', 'unknown') for f in factors[:3]] if factors else []
        }
    
    def run_comprehensive_test(self):
        """Run all tests with predefined test cases."""
        print("🚀 Starting Comprehensive Generator Testing")
        print("="*80)
        
        # Test situations covering various emotional scenarios
        test_situations = [
            "My friend Simon, an amateur painter, had been working on a portrait of his deceased pet for over a week. After he finally completed it, he stepped back and immediately noticed that he had painted the wrong color for his pet's fur. Just then, his wife saw it and exclaimed, That's exactly how I remember him! I was randomly reminded of this today when I was looking at a picture of my own dog.",
            "After months of voice-training, Lena finally nailed the high note in her late father’s favorite song and recorded it for a memorial video. That evening, she accidentally deleted the file. Distraught, she collapsed on the couch—only to realize that an old answering-machine tape, still sitting in the player, held a snippet of her dad humming the same melody (off-key, but joyfully). She played it on loop, crying and laughing as the house filled with their unfinished duet.",
            "For their tenth anniversary, June planned to give her partner a sealed envelope containing snippets of love letters she’d written but never dared to send during their early friendship. The night before, she accidentally left the envelope on the train. Heartbroken, she tried to rewrite them from memory, but the words felt forced. Two weeks later, a package arrived: a rail conductor had found the envelope and mailed it back, adding a short note—“Couldn’t help reading. Reminded me to phone my ex-wife.",
            "I am very very happy that my daughter is getting married"
        
        ]
        
        # Test each generator individually
        emotion_results = self.test_emotion_generator(test_situations)
        factor_results = self.test_factor_generator(test_situations)
        integration_results = self.test_integration(test_situations)
        
        # Test emotion comparison: abstract vs full prompt
        emotion_comparison_results = self.test_emotion_comparison_abstract_vs_full(test_situations)
        
        # Combine all results
        all_results = {
            'emotion_generator_tests': emotion_results,
            'factor_generator_tests': factor_results,
            'integration_tests': integration_results,
            'emotion_comparison_tests': emotion_comparison_results,
            'summary': self._generate_test_summary(emotion_results, factor_results, integration_results, emotion_comparison_results)
        }
        
        # Save results to file
        self._save_test_results(all_results)
        
        return all_results
    
    def _generate_test_summary(self, emotion_results: Dict, factor_results: Dict, 
                             integration_results: Dict, emotion_comparison_results: Dict = None) -> Dict[str, Any]:
        """Generate a summary of test results."""
        
        def count_successes(results):
            return sum(1 for r in results.values() if r.get('success', False))
        
        emotion_successes = count_successes(emotion_results)
        factor_successes = count_successes(factor_results)
        integration_successes = count_successes(integration_results)
        comparison_successes = count_successes(emotion_comparison_results) if emotion_comparison_results else 0
        
        total_tests = len(emotion_results) + len(factor_results) + len(integration_results)
        total_successes = emotion_successes + factor_successes + integration_successes
        
        if emotion_comparison_results:
            total_tests += len(emotion_comparison_results)
            total_successes += comparison_successes
        
        summary = {
            'total_tests': total_tests,
            'total_successes': total_successes,
            'success_rate': total_successes / total_tests if total_tests > 0 else 0,
            'emotion_generator': {
                'tests': len(emotion_results),
                'successes': emotion_successes,
                'success_rate': emotion_successes / len(emotion_results) if emotion_results else 0
            },
            'factor_generator': {
                'tests': len(factor_results),
                'successes': factor_successes,
                'success_rate': factor_successes / len(factor_results) if factor_results else 0
            },
            'integration': {
                'tests': len(integration_results),
                'successes': integration_successes,
                'success_rate': integration_successes / len(integration_results) if integration_results else 0
            }
        }
        
        if emotion_comparison_results:
            summary['emotion_comparison'] = {
                'tests': len(emotion_comparison_results),
                'successes': comparison_successes,
                'success_rate': comparison_successes / len(emotion_comparison_results) if emotion_comparison_results else 0
            }
            
        return summary
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to a JSON file."""
        output_file = "test_results_factor_emotion_generators.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Test results saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save results: {str(e)}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results."""
        summary = results['summary']
        
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Total Successes: {summary['total_successes']}")
        print(f"Overall Success Rate: {summary['success_rate']:.1%}")
        print()
        print(f"🎭 Emotion Generator: {summary['emotion_generator']['successes']}/{summary['emotion_generator']['tests']} ({summary['emotion_generator']['success_rate']:.1%})")
        print(f"⚙️ Factor Generator: {summary['factor_generator']['successes']}/{summary['factor_generator']['tests']} ({summary['factor_generator']['success_rate']:.1%})")
        print(f"🔗 Integration Tests: {summary['integration']['successes']}/{summary['integration']['tests']} ({summary['integration']['success_rate']:.1%})")
        print("="*80)


def main():
    """Main test execution function."""
    try:
        # Initialize tester
        tester = GeneratorTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Print summary
        tester.print_summary(results)
        
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
