"""
Emobird: Dynamic Emotion Analysis System (Direct Assessment Version)

This system generates scenarios and analyzes emotions dynamically at inference time.
Modified to use direct LLM assessment instead of BIRD pooling.
"""

import json
import torch
from typing import Dict, List, Any, Tuple
import sys
import atexit

from EmoBIRD.scenario_generator import ScenarioGenerator
from EmoBIRD.factor_generator import FactorGenerator
from EmoBIRD.emotion_generator import EmotionGenerator
from EmoBIRD.direct_emotion_predictor import DirectEmotionPredictor
from EmoBIRD.output_generator import OutputGenerator
from EmoBIRD.config import EmobirdConfig
from EmoBIRD.vllm_wrapper import VLLMWrapper
from EmoBIRD.utils import print_gpu_info
from EmoBIRD.schemas import UnifiedEmotionAnalysis


class Emobird:
    """
    Main Emobird inference engine that generates scenarios and analyzes emotions dynamically.
    Now uses direct LLM assessment instead of BIRD pooling.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        self.verbose = True  # Default to verbose mode
        
        print("🐦 Initializing Emobird system (Direct Assessment Version)...")
        
        # Load vLLM for inference first
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
        
        # Initialize vLLM wrapper
        self.vllm_wrapper = VLLMWrapper(self.config)
        
        print("✅ vLLM loaded successfully!")
    
    def analyze_emotion(self, user_situation: str) -> Dict[str, Any]:
        """
        Main inference method: analyze emotion for a given user situation.

        Modular flow (multi-stage):
        1. Generate abstract/summary from the user situation
        2. Generate psychological factors with possible values
        3. Extract factor values from the situation
        4. Extract 3-5 crucial emotions from the abstract
        5. Predict emotion probabilities using Likert ratings (strict JSON if possible)
        6. Generate conversational response from the predicted emotions
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing:
            - crucial_emotions: List of key emotions identified
            - factors: Identified factors and their values
            - emotions: Final emotion probability distribution
            - metadata: Additional information about the inference
        """
        if self.verbose:
            print(f"\n🔍 Analyzing situation: '{user_situation[:100]}...'")
        
        try:
            processing_steps: List[str] = []

            # 1) Abstract generation only
            if self.verbose:
                print("📋 Generating abstract...")
            abstract = self.scenario_generator._generate_abstract(user_situation)
            processing_steps.append('abstract_generation')

            # 2) Factor generation (definitions + initial values)
            if self.verbose:
                print("🧠 Generating factors...")
            factor_result = self.factor_generator.generate_factors(user_situation, abstract)
            factors = factor_result.get('factors', [])
            processing_steps.append('factor_generation')

            # Ensure we have at least 3 factors via fallback if needed
            if not factors or len(factors) < 3:
                print("⚠️ Insufficient factors, using fallback generation")
                fallback = self.factor_generator._create_fallback_factors(user_situation)  # fallback is internal
                factors = fallback.get('factors', [])

            # 3) Factor value extraction
            if self.verbose:
                print("🎯 Extracting factor values...")
            factor_values = self.factor_generator.analyze_situation(user_situation, factors)
            processing_steps.append('factor_value_extraction')

            # 4) Crucial emotion extraction from abstract
            if self.verbose:
                print("🎭 Extracting crucial emotions from abstract...")
            crucial_emotions = self.emotion_generator.extract_crucial_emotions_from_abstract(abstract)
            processing_steps.append('crucial_emotion_extraction')

            # Ensure we have a valid set of emotions (3-5). Apply simple fallback if needed.
            if not crucial_emotions or len(crucial_emotions) < 3:
                print("⚠️ Emotion extraction returned too few items, applying default fallback emotions")
                crucial_emotions = ["anxiety", "sadness", "frustration"]

            # 5) Likert-scale emotion prediction (strict JSON first, then fallback)
            if self.verbose:
                print("📈 Predicting emotions with Likert ratings...")
            predictor = DirectEmotionPredictor(self.vllm_wrapper, factors=factors, emotions=crucial_emotions)
            pred = predictor.predict_with_explanation(user_situation, factor_values)
            emotions_dict = pred.get('emotions', {})
            explanation = pred.get('explanation', '')
            processing_steps.append('likert_emotion_prediction')

            # 6) Conversational response
            top_emotions = dict(sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3])
            if self.verbose:
                print("🗣️ Generating conversational response...")
            response = self.output_generator.generate_response(
                user_input=user_situation,
                top_emotions=top_emotions,
                context={
                    'factors': factor_values,
                    'abstract': abstract,
                    'crucial_emotions': crucial_emotions,
                    'note': 'EmoBIRD emotional analysis (modular)'
                }
            )

            # Compose result
            return {
                'abstract': abstract,
                'crucial_emotions': crucial_emotions,
                'factors': factor_values,
                'emotions': emotions_dict,
                'top_emotions': top_emotions,
                'explanation': explanation,
                'response': response,
                'metadata': {
                    'method': 'modular_pipeline_v1',
                    'pooling': 'none',
                    'abstract_length': len(abstract),
                    'num_crucial_emotions': len(crucial_emotions),
                    'num_factors': len(factors),
                    'processing_steps': processing_steps + ['response_generation'],
                }
            }
        
        except Exception as e:
            print(f"❌ Error in EmoBIRD pipeline (modular): {e}")
            # Hard-gate fallback: use base LLM to respond without partial insights.
            try:
                fallback_prompt = (
                    "You are an empathetic, supportive assistant. "
                    "Provide a concise, compassionate response with 2-4 sentences. "
                    "Acknowledge feelings, offer understanding, and suggest one gentle next step. "
                    "Avoid diagnosing or giving medical/clinical advice.\n\n"
                    f"User situation: {user_situation}\n\n"
                    "Response:"
                )
                fallback_response = self.vllm_wrapper.generate(
                    fallback_prompt,
                    component="emobird",
                    interaction_type="fallback_base_llm_response",
                )
            except Exception as ef:
                print(f"⚠️ Fallback generation failed: {ef}")
                fallback_response = "I'm here for you. I'm sorry this is hard. If you'd like, share more about what's going on."

            # Return a consistent structure with empty analytics and the base LLM response
            return {
                'abstract': "",
                'crucial_emotions': [],
                'factors': {},
                'emotions': {},
                'top_emotions': {},
                'response': fallback_response if isinstance(fallback_response, str) else str(fallback_response),
                'metadata': {
                    'method': 'fallback_base_llm_only',
                    'pooling': 'none',
                    'abstract_length': 0,
                    'num_crucial_emotions': 0,
                    'num_factors': 0,
                    'processing_steps': ['modular_pipeline_failed', 'fallback_base_llm_response'],
                    'error_message': str(e),
                    'json_call_meta': getattr(self.vllm_wrapper, 'last_json_call_meta', None),
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
    
    print("\n🐦 EmoBIRD Interactive Session Started!")
    print("💡 Tip: Type 'exit' to quit the session\n")
    
    while True:
        try:
            # Get user input
            user_situation = input("\nPlease describe your situation (or 'exit' to quit): ")
            
            # Check for exit condition
            if user_situation.lower().strip() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for using EmoBIRD! Goodbye!")
                break
            
            # Skip empty inputs
            if not user_situation.strip():
                print("⚠️ Please enter a situation to analyze.")
                continue
            
            # Analyze emotion
            result = emobird.analyze_emotion(user_situation)
            
            # Display results
            print(f"\n🎭 Crucial Emotions Identified: {', '.join(result['crucial_emotions'])}")
            print(f"\n⚙️ Factor Values:")
            for factor, value in result['factors'].items():
                print(f"  - {factor}: {value}")
            
            print(f"\n😊 Final Emotion Probabilities (Direct Assessment):")
            sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_emotions:
                bar = '█' * int(prob * 20)
                print(f"  {emotion:12} {prob:.3f} {bar}")
            
            if 'explanation' in result:
                print(f"\n💭 Reasoning: {result['explanation']}")
            
            if 'response' in result and result['response']:
                print(f"\n🤖 Generated Response: {result['response']}")
            
            print(f"\n📊 Processing Summary:")
            metadata = result['metadata']
            print(f"  - Method: {metadata.get('method', 'unknown')}")
            print(f"  - Pooling: {metadata.get('pooling', 'unknown')}")
            print(f"  - Crucial emotions found: {metadata['num_crucial_emotions']}")
            print(f"  - Psychological factors: {metadata['num_factors']}")
            print(f"  - Processing steps: {len(metadata['processing_steps'])}")
            
            print("\n" + "="*60)  # Separator between analyses
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("🔄 You can try again with a different situation.")


if __name__ == "__main__":
    main()
