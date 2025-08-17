"""
Emobird (No Pooling Version): Simplified Emotion Analysis System

This version removes BIRD pooling and instead directly asks the LLM
to assess emotion probabilities given the identified factor values.
"""

import json
import torch
from typing import Dict, List, Any, Tuple
import sys
import atexit

from scenario_generator import ScenarioGenerator
from factor_generator import FactorGenerator
from emotion_generator import EmotionGenerator
from direct_emotion_predictor import DirectEmotionPredictor
from output_generator import OutputGenerator
from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper
from utils import print_gpu_info


class EmobirdNoPooling:
    """
    Simplified Emobird engine without BIRD pooling.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        self.verbose = True
        
        print("🐦 Initializing Emobird (No Pooling Version)...")
        
        # Load vLLM for inference
        self._load_llm()
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(self.config)
        self.factor_generator = FactorGenerator(self.config)
        self.emotion_generator = EmotionGenerator(self.config)
        self.output_generator = OutputGenerator(self.config)
        
        # Set vLLM wrapper for generators
        self.scenario_generator.set_vllm(self.vllm_wrapper)
        self.factor_generator.set_vllm(self.vllm_wrapper)
        self.emotion_generator.set_vllm(self.vllm_wrapper)
        self.output_generator.set_vllm(self.vllm_wrapper)
        
        print("✅ Emobird system initialized successfully!")
    
    def _load_llm(self):
        """Load the vLLM wrapper for inference."""
        print(f"🚀 Loading vLLM: {self.config.llm_model_name}")
        self.vllm_wrapper = VLLMWrapper(self.config)
        print("✅ vLLM loaded successfully!")
    
    def analyze_emotion(self, user_situation: str) -> Dict[str, Any]:
        """
        Simplified emotion analysis pipeline without BIRD pooling.
        
        Flow:
        1. Generate abstract from user situation
        2. Extract crucial emotions from abstract
        3. Generate psychological factors from user input
        4. Extract factor values for this specific situation
        5. Directly assess emotion probabilities given factor values (NO POOLING)
        6. Generate conversational response
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing analysis results
        """
        if self.verbose:
            print(f"\n🔍 Analyzing situation: '{user_situation[:100]}...'")
        
        try:
            # Step 1: Generate abstract
            if self.verbose:
                print("📋 Generating abstract from situation...")
            abstract = self.scenario_generator._generate_abstract(user_situation)
            if self.verbose:
                print(f"   Generated abstract: {abstract[:100]}...")
            
            # Step 2: Extract crucial emotions from abstract
            if self.verbose:
                print("🎭 Extracting crucial emotions from abstract...")
            crucial_emotions = self.emotion_generator.extract_crucial_emotions_from_abstract(abstract)
            if self.verbose:
                print(f"   Found crucial emotions: {crucial_emotions}")
            
            # Step 3: Generate psychological factors
            if self.verbose:
                print("⚙️ Generating important factors...")
            factors = self.factor_generator.generate_factors_from_situation(user_situation)
            
            # Step 4: Extract factor values
            if self.verbose:
                print("🎯 Extracting factor values...")
            factor_values = self.factor_generator.extract_factor_values_direct(
                user_situation, factors
            )
            
            # Step 5: Direct emotion assessment (NO POOLING)
            if self.verbose:
                print("🎯 Directly assessing emotion probabilities from factors...")
            
            # Initialize direct predictor with factors and emotions
            direct_predictor = DirectEmotionPredictor(
                self.vllm_wrapper, 
                factors, 
                crucial_emotions
            )
            
            # Get emotion predictions directly
            result = direct_predictor.predict_with_explanation(
                user_situation, 
                factor_values
            )
            
            emotions = result['emotions']
            top_emotions = result['top_emotions']
            explanation = result['explanation']
            
            # Step 6: Generate conversational response
            if self.verbose:
                print("🗣️ Generating conversational response...")
            response = self.output_generator.generate_response(
                user_input=user_situation,
                top_emotions=top_emotions,
                context={
                    'factors': factor_values,
                    'abstract': abstract,
                    'crucial_emotions': crucial_emotions,
                    'note': 'EmoBIRD emotional analysis (no pooling)'
                }
            )
            
            # Return comprehensive result
            return {
                'abstract': abstract,
                'crucial_emotions': crucial_emotions,
                'factors': factor_values,
                'emotions': emotions,
                'top_emotions': top_emotions,
                'explanation': explanation,
                'response': response,
                'metadata': {
                    'method': 'direct_llm_assessment',
                    'pooling': 'none',
                    'abstract_length': len(abstract),
                    'num_crucial_emotions': len(crucial_emotions),
                    'num_factors': len(factor_values),
                    'processing_steps': [
                        'abstract_generation',
                        'emotion_extraction_from_abstract',
                        'factor_generation',
                        'factor_value_extraction',
                        'direct_emotion_assessment',
                        'response_generation'
                    ]
                }
            }
        
        except Exception as e:
            print(f"❌ Error in EmoBIRD pipeline: {e}")
            return {
                'abstract': "",
                'crucial_emotions': [],
                'factors': {},
                'emotions': {},
                'top_emotions': {},
                'explanation': "",
                'response': f"Error in EmoBIRD analysis: {str(e)}",
                'error': True,
                'error_message': str(e),
                'metadata': {
                    'method': 'direct_llm_assessment',
                    'pooling': 'none',
                    'abstract_length': 0,
                    'num_crucial_emotions': 0,
                    'num_factors': 0,
                    'processing_steps': ['error_occurred']
                }
            }
    
    def compare_with_pooling(self, user_situation: str) -> Dict[str, Any]:
        """
        Run analysis and compare with what BIRD pooling would have produced.
        Useful for evaluating the difference between approaches.
        """
        # Get direct assessment
        direct_result = self.analyze_emotion(user_situation)
        
        # For comparison, we could load the old pooling-based system
        # and run the same analysis, but for now just return the direct result
        # with a note about the comparison
        
        direct_result['comparison_note'] = (
            "This version uses direct LLM assessment without BIRD pooling. "
            "The LLM considers all factors holistically when assigning emotion probabilities."
        )
        
        return direct_result


def cleanup_resources():
    """Clean up distributed computing resources to prevent warnings"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass


def main():
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    print_gpu_info()
    
    # Initialize Emobird (No Pooling Version)
    emobird = EmobirdNoPooling()
    
    # Example situations to test
    test_situations = [
        "I just got promoted at work after months of hard work, and my team surprised me with a celebration.",
        "My best friend betrayed my trust by sharing my personal secrets with others.",
        "I'm feeling overwhelmed with deadlines and my manager keeps adding more tasks.",
    ]
    
    print("\n" + "="*70)
    print("TESTING EMOBIRD WITHOUT POOLING")
    print("="*70)
    
    # Interactive mode
    mode = input("\nChoose mode:\n1. Test with examples\n2. Enter custom situation\nChoice (1/2): ")
    
    if mode == "1":
        for i, situation in enumerate(test_situations, 1):
            print(f"\n{'='*70}")
            print(f"Example {i}: {situation[:50]}...")
            print('='*70)
            
            result = emobird.analyze_emotion(situation)
            
            print(f"\n📊 Results:")
            print(f"Crucial Emotions: {', '.join(result['crucial_emotions'])}")
            print(f"\nFactor Values:")
            for factor, value in result['factors'].items():
                print(f"  - {factor}: {value}")
            
            print(f"\n😊 Emotion Probabilities (Direct Assessment):")
            sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_emotions:
                bar = '█' * int(prob * 20)
                print(f"  {emotion:12} {prob:.3f} {bar}")
            
            print(f"\n💭 Explanation: {result['explanation']}")
            print(f"\n💬 Response: {result['response']}")
            
            input("\nPress Enter to continue...")
    
    else:
        # Custom situation
        user_situation = input("\nPlease describe your situation: ")
        
        result = emobird.analyze_emotion(user_situation)
        
        print(f"\n📊 Results:")
        print(f"Crucial Emotions: {', '.join(result['crucial_emotions'])}")
        print(f"\nFactor Values:")
        for factor, value in result['factors'].items():
            print(f"  - {factor}: {value}")
        
        print(f"\n😊 Emotion Probabilities (Direct Assessment):")
        sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_emotions:
            bar = '█' * int(prob * 20)
            print(f"  {emotion:12} {prob:.3f} {bar}")
        
        print(f"\n💭 Explanation: {result['explanation']}")
        print(f"\n💬 Response: {result['response']}")
    
    print(f"\n📝 Note: This version uses direct LLM assessment without BIRD pooling.")
    print(f"The LLM considers all factors together when determining emotion probabilities.")


if __name__ == "__main__":
    main()
