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
from schemas import UnifiedEmotionAnalysis


class Emobird:
    """
    Main Emobird inference engine that generates scenarios and CPTs dynamically.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        
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
        self.output_generator = OutputGenerator(self.config)
        
        # Set vLLM wrapper for generators
        self.scenario_generator.set_vllm(self.vllm_wrapper)
        self.factor_generator.set_vllm(self.vllm_wrapper)
        self.emotion_generator.set_vllm(self.vllm_wrapper)
        self.neutral_prob_extractor.set_vllm(self.vllm_wrapper)
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
        
        Hardened flow (unified stage):
        1. Single strict-JSON call returning unified object with factors, factor_values, emotions, and emotion_probs
        2. Generate conversational response from the unified output
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing:
            - crucial_emotions: List of 2-4 key emotions identified
            - factors: Identified factors and their values
            - emotions: Final emotion probability distribution
            - metadata: Additional information about the inference
        """
        print(f"\n🔍 Analyzing situation: '{user_situation[:100]}...'")

        # Unified strict-JSON stage with perspective locking
        subject = "user"
        prompt = self._build_unified_prompt(subject=subject, situation=user_situation)

        print("🧩 Running unified strict-JSON analysis...")
        parsed = self.vllm_wrapper.json_call(
            prompt=prompt,
            component="emobird",
            interaction_type="unified_emotion_analysis",
            max_retries=self.config.allow_format_only_retry,
            schema_model=UnifiedEmotionAnalysis,  # Pydantic validation
            temperature_override=self.config.temp_analysis,
            max_tokens_override=self.config.max_tokens_analysis,
        )

        # Construct model instance for convenience
        uea = UnifiedEmotionAnalysis.model_validate(parsed)

        # Perspective lock: Ensure subject matches
        if uea.subject.strip().lower() != subject:
            raise ValueError(f"Perspective lock failed: expected subject='{subject}' but got '{uea.subject}'")

        # Prepare response using existing output generator
        emotions_dict = dict(uea.emotion_probs)
        top_emotions = dict(sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3])

        print("💬 Generating conversational response...")
        response = self.output_generator.generate_response(
            user_input=user_situation,
            top_emotions=top_emotions,
            context={
                'factors': dict(uea.factor_values),
                'abstract': uea.scenario_summary,
                'crucial_emotions': list(uea.emotions),
            }
        )

        return {
            'abstract': uea.scenario_summary,
            'crucial_emotions': list(uea.emotions),
            'factors': dict(uea.factor_values),
            'neutral_probabilities': {},
            'cpt_data': {},
            'emotions': emotions_dict,
            'top_emotions': top_emotions,
            'response': response,
            'metadata': {
                'subject': uea.subject,
                'num_crucial_emotions': len(uea.emotions),
                'num_factors': len(uea.factor_values),
                'processing_steps': [
                    'unified_strict_json_analysis',
                    'conversational_response_generation',
                ],
                'version': uea.version,
            }
        }

    def _build_unified_prompt(self, subject: str, situation: str) -> str:
        """Construct the unified analysis prompt with strict JSON instructions."""
        required_keys = [
            'subject', 'situation', 'scenario_summary',
            'factors', 'factor_values', 'emotions', 'emotion_probs', 'version'
        ]
        instructions = (
            "You are EmoBIRD, an emotion analysis engine. "
            "Return STRICT JSON only. No commentary, no markdown. "
            "Preserve the subject exactly as provided."
        )
        # Minimal schema hint
        schema_hint = (
            '{'
            '"subject": "string", '
            '"situation": "string", '
            '"scenario_summary": "string", '
            '"factors": [{"name": "string", "value_type": "boolean|categorical|ordinal", "possible_values": [], "description": "string"}], '
            '"factor_values": {}, '
            '"emotions": ["string"], '
            '"emotion_probs": {"emotion": 0.0}, '
            '"version": "uea_v1"'
            '}'
        )
        prompt = (
            f"{instructions}\n"
            f"Subject: {subject}\n"
            f"Situation: {situation}\n"
            f"Requirements:\n"
            f"- Output JSON only matching keys: {', '.join(required_keys)}\n"
            f"- The 'emotion_probs' must sum to 1.0 and keys subset of 'emotions'\n"
            f"- Factor names must be consistent with 'factor_values' keys\n"
            f"- Do not include any text before or after JSON\n"
            f"Schema hint: {schema_hint}\n"
        )
        return prompt
    
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
    
    print(f"\n💬 AI Response:")
    print(f"{result['response']}")
    
    print(f"\n📊 Processing Summary:")
    metadata = result['metadata']
    print(f"  - Crucial emotions found: {metadata['num_crucial_emotions']}")
    print(f"  - Psychological factors: {metadata['num_factors']}")
    print(f"  - Processing steps: {len(metadata['processing_steps'])}")


if __name__ == "__main__":
    main()
