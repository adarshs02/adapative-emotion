"""
Emobird: Dynamic Emotion Analysis System

This system generates scenarios and CPTs dynamically at inference time
rather than using pre-stored scenarios and CPT files.
"""

import json
import torch
from typing import Dict, List, Any, Tuple
import sys
import atexit

from scenario_generator import ScenarioGenerator
from factor_generator import FactorGenerator
from emotion_generator import EmotionGenerator
from neutral_probability_extractor import NeutralProbabilityExtractor
from factor_entailment import FactorEntailment
from logistic_pooler import LogisticPooler
from emotion_predictor import EmotionPredictor
from output_generator import OutputGenerator
from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper
from utils import print_gpu_info


class Emobird:
    """
    Main Emobird inference engine that generates scenarios and CPTs dynamically.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        self.verbose = True  # Default to verbose mode
        
        print("🐦 Initializing Emobird system...")
        
        # Load vLLM for inference first
        self._load_llm()
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(self.config)
        self.factor_generator = FactorGenerator(self.config)
        self.emotion_generator = EmotionGenerator(self.config)
        self.neutral_prob_extractor = NeutralProbabilityExtractor(self.config)
        
        # Initialize runtime components (will be set up after CPT generation)
        self.factor_entailment = None
        self.logistic_pooler = LogisticPooler()
        self.emotion_predictor = None
        
        # Set vLLM wrapper for generators
        self.scenario_generator.set_vllm(self.vllm_wrapper)
        self.factor_generator.set_vllm(self.vllm_wrapper)
        self.emotion_generator.set_vllm(self.vllm_wrapper)
        self.neutral_prob_extractor.set_vllm(self.vllm_wrapper)
        
        print("✅ Emobird system initialized successfully!")
    
    def _load_llm(self):
        """Load the vLLM wrapper for inference."""
        print(f"🚀 Loading vLLM: {self.config.llm_model_name}")
        
        # Initialize vLLM wrapper
        self.vllm_wrapper = VLLMWrapper(self.config)
        
        print("✅ vLLM loaded successfully!")
    
    def analyze_emotion(self, user_situation: str) -> Dict[str, Any]:
        """
        Main inference method: analyze emotion for a given user situation.
        
        New flow:
        1. Extract crucial emotions (2-4) from user situation
        2. Generate psychological factors from user input
        3. Extract factor values for this specific situation
        4. Generate CPT using factors and crucial emotions
        5. Calculate final emotion probabilities
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing:
            - crucial_emotions: List of 2-4 key emotions identified
            - factors: Identified factors and their values
            - emotions: Final emotion probability distribution
            - metadata: Additional information about the inference
        """
        if self.verbose:
            print(f"\n🔍 Analyzing situation: '{user_situation[:100]}...'")
        
        try:
            # Step 1: Generate abstract from user situation
            if self.verbose:
                print("📋 Generating abstract from situation...")
            abstract = self.scenario_generator._generate_abstract(user_situation)
            if self.verbose:
                print(f"   Generated abstract: {abstract[:100]}...")  # Show first 100 chars
            
            # Step 2: Extract 2-4 crucial emotions from abstract
            if self.verbose:
                print("🎭 Extracting crucial emotions from abstract...")
            crucial_emotions = self.emotion_generator.extract_crucial_emotions_from_abstract(abstract)
            if self.verbose:
                print(f"   Found crucial emotions: {crucial_emotions}")
            
            # Step 3: Generate psychological factors from user input
            if self.verbose:
                print("⚙️ Generating important factors...")
            factors = self.factor_generator.generate_factors_from_situation(user_situation)
            
            # Step 4: Extract specific factor values for this situation
            if self.verbose:
                print("🎯 Extracting factor values...")
            factor_values = self.factor_generator.extract_factor_values_direct(
                user_situation, factors
            )
            
            # Step 5: Extract neutral probabilities for (factor, emotion) pairs
            if self.verbose:
                print("🎲 Extracting neutral probabilities...")
            neutral_probabilities = self.neutral_prob_extractor.extract_neutral_probabilities(
                factors, crucial_emotions
            )
            
            # Step 6: Build CPT from neutral probabilities
            if self.verbose:
                print("📊 Building CPT from neutral probabilities...")
            cpt_data = self.neutral_prob_extractor.build_cpt_from_probabilities(
                neutral_probabilities, factors
            )
            
            # Step 7: Initialize runtime components with CPT data
            if self.verbose:
                print("🔧 Setting up runtime components...")
            self.factor_entailment = FactorEntailment(self.vllm_wrapper, factors)
            # EmotionPredictor now loads CPT from cache automatically
            self.emotion_predictor = EmotionPredictor(
                self.factor_entailment, 
                self.logistic_pooler
            )
            
            # Step 8: Use BIRD pooling for final emotion probabilities  
            if self.verbose:
                print("🎯 Calculating emotions using BIRD pooling...")
            emotions = self.emotion_predictor.predict_all(user_situation)
            
            # Create top_emotions from sorted emotions with error handling
            top_emotions = {}
            try:
                if isinstance(emotions, dict) and emotions:
                    sorted_items = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_emotions = dict(sorted_items)
                else:
                    if self.verbose:
                        print("⚠️ Emotions is not a valid dictionary, using empty top_emotions")
                    top_emotions = {}
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error creating top_emotions: {e}")
                top_emotions = {}
            
            # Step 9: Generate conversational response
            if self.verbose:
                print("🗣️ Generating conversational response...")
            output_generator = OutputGenerator(self.config)
            output_generator.set_vllm(self.vllm_wrapper)
            response = output_generator.generate_response(
                user_input=user_situation,
                top_emotions=top_emotions,
                context={
                    'factors': factor_values,
                    'abstract': abstract,
                    'crucial_emotions': crucial_emotions,
                    'note': 'EmoBIRD emotional analysis'
                }
            )
            
            # Return comprehensive result
            return {
                'abstract': abstract,
                'crucial_emotions': crucial_emotions,
                'factors': factor_values,
                'neutral_probabilities': neutral_probabilities,
                'cpt_data': cpt_data,
                'emotions': emotions,
                'top_emotions': top_emotions,
                'response': response,
                'metadata': {
                    'abstract_length': len(abstract),
                    'num_crucial_emotions': len(crucial_emotions),
                    'num_factors': len(factor_values),
                    'processing_steps': [
                        'abstract_generation',
                        'emotion_extraction_from_abstract',
                        'factor_generation', 
                        'factor_value_extraction',
                        'neutral_probability_extraction',
                        'cpt_building',
                        'runtime_component_setup',
                        'bird_pooling_emotion_calculation',
                    ]
                }
            }
        
        except Exception as e:
            print(f"❌ Error in EmoBIRD pipeline: {e}")
            # Return structured error response that maintains expected dictionary format
            return {
                'abstract': "",
                'crucial_emotions': [],
                'factors': {},
                'neutral_probabilities': {},
                'cpt_data': {},
                'emotions': {},
                'top_emotions': {},
                'response': f"Error in EmoBIRD analysis: {str(e)}",
                'error': True,
                'error_message': str(e),
                'metadata': {
                    'abstract_length': 0,
                    'num_crucial_emotions': 0,
                    'num_factors': 0,
                    'processing_steps': ['error_occurred']
                }
            }
    
    def batch_analyze(self, situations: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple situations in batch."""
        results = []
        for i, situation in enumerate(situations):
            print(f"\n📦 Processing batch item {i+1}/{len(situations)}")
            result = self.analyze_emotion(situation)
            results.append(result)
        return results


def cleanup_resources():
    """Clean up distributed computing resources to prevent warnings"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass  # Ignore cleanup errors

def main():
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    print_gpu_info()

    # Initialize Emobird
    emobird = Emobird()
    
    # Example situation
    user_situation = input("\nPlease describe your situation: ")
    
    # Analyze emotion
    result = emobird.analyze_emotion(user_situation)
    
    # Display results
    print(f"\n🎭 Crucial Emotions Identified: {', '.join(result['crucial_emotions'])}")
    print(f"\n⚙️ Factor Values:")
    for factor, value in result['factors'].items():
        print(f"  - {factor}: {value}")
    
    print(f"\n😊 Final Emotion Probabilities:")
    sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_emotions:
        print(f"  - {emotion}: {prob:.3f}")
    
    print(f"\n📊 Processing Summary:")
    metadata = result['metadata']
    print(f"  - Crucial emotions found: {metadata['num_crucial_emotions']}")
    print(f"  - Psychological factors: {metadata['num_factors']}")
    print(f"  - Processing steps: {len(metadata['processing_steps'])}")


if __name__ == "__main__":
    main()
