"""
Improved embedding strategies for better scenario matching without training.
Uses multiple techniques: better preprocessing, ensemble embeddings, and semantic enhancement.
"""

import json
import numpy as np
from typing import List, Dict, Any
import re
from vllm import LLM
import config


class ImprovedEmbeddingGenerator:
    """Enhanced embedding generation with multiple strategies."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            task='embed'  # Enable embedding mode
        )
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better embeddings."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Expand contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def create_enhanced_text(self, scenario: Dict) -> str:
        """Create enhanced text representation of scenario."""
        description = scenario.get('description', '')
        
        # Add context from tags if available
        enhanced_parts = [self.preprocess_text(description)]
        
        if 'tags' in scenario:
            # Convert tags to natural language
            tag_text = self._tags_to_text(scenario['tags'])
            enhanced_parts.append(tag_text)
        
        # Add emotional context keywords
        emotion_keywords = self._extract_emotion_keywords(description)
        if emotion_keywords:
            enhanced_parts.append(f"emotions: {', '.join(emotion_keywords)}")
        
        return ". ".join(enhanced_parts)
    
    def _tags_to_text(self, tags: List[str]) -> str:
        """Convert tags to natural language description."""
        if not tags:
            return ""
        
        # Group tags by type (if they follow a pattern)
        tag_groups = {}
        for tag in tags:
            if '_' in tag:
                prefix = tag.split('_')[0]
                if prefix not in tag_groups:
                    tag_groups[prefix] = []
                tag_groups[prefix].append(tag.replace('_', ' '))
            else:
                if 'general' not in tag_groups:
                    tag_groups['general'] = []
                tag_groups['general'].append(tag)
        
        # Convert to natural language
        descriptions = []
        for group, group_tags in tag_groups.items():
            if group == 'general':
                descriptions.append(f"involving {', '.join(group_tags)}")
            else:
                descriptions.append(f"{group}: {', '.join(group_tags)}")
        
        return "Context: " + "; ".join(descriptions)
    
    def _extract_emotion_keywords(self, text: str) -> List[str]:
        """Extract emotion-related keywords from text."""
        emotion_words = {
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'hurt', 'grief'],
            'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'panic'],
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'cheerful', 'joy'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'appalled'],
            'trust': ['trust', 'confident', 'secure', 'reliable', 'faith'],
            'anticipation': ['anticipate', 'expect', 'hope', 'eager', 'excited']
        }
        
        text_lower = text.lower()
        found_emotions = []
        
        for emotion, keywords in emotion_words.items():
            if any(keyword in text_lower for keyword in keywords):
                found_emotions.append(emotion)
        
        return found_emotions
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with improved preprocessing."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embedding_outputs = self.llm.encode(processed_texts)
        
        # Extract embeddings from vLLM output
        if isinstance(embedding_outputs, list):
            # Handle list of PoolingRequestOutput objects
            if len(embedding_outputs) > 0:
                first_item = embedding_outputs[0]
                
                if hasattr(first_item, 'embedding'):
                    embeddings = [output.embedding for output in embedding_outputs]
                elif hasattr(first_item, 'outputs'):
                    # PoolingRequestOutput.outputs.data structure (vLLM embedding format)
                    embeddings = [output.outputs.data for output in embedding_outputs]
                else:
                    # Assume it's already embeddings
                    embeddings = embedding_outputs
            else:
                embeddings = []
        elif hasattr(embedding_outputs, 'outputs'):
            # Handle single PoolingRequestOutput
            embeddings = [embedding_outputs.outputs.data]
        elif hasattr(embedding_outputs, 'embedding'):
            # Handle direct embedding
            embeddings = [embedding_outputs.embedding]
        else:
            # Fallback - assume it's already the embeddings
            embeddings = embedding_outputs
        
        return np.array(embeddings)
    
    def generate_ensemble_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate ensemble embeddings using multiple strategies."""
        # Strategy 1: Original text
        original_embeddings = self.generate_embeddings(texts)
        
        # Strategy 2: Enhanced text (with emotional context)
        enhanced_texts = []
        for text in texts:
            # Create a pseudo-scenario dict for enhancement
            scenario = {'description': text}
            enhanced_texts.append(self.create_enhanced_text(scenario))
        
        enhanced_embeddings = self.generate_embeddings(enhanced_texts)
        
        # Strategy 3: Question-formatted text (for better semantic matching)
        question_texts = [f"What emotion scenario involves: {text}" for text in texts]
        question_embeddings = self.generate_embeddings(question_texts)
        
        # Combine embeddings (weighted average)
        ensemble_embeddings = (
            0.5 * original_embeddings + 
            0.3 * enhanced_embeddings + 
            0.2 * question_embeddings
        )
        
        return ensemble_embeddings


def rebuild_improved_indices():
    """Rebuild indices using improved embedding strategies."""
    print("🔧 Rebuilding indices with improved embeddings...")
    
    # Load scenarios
    with open('atomic-scenarios.json', 'r') as f:
        data = json.load(f)
        # Handle both direct array and wrapped structure
        if isinstance(data, dict) and 'scenarios' in data:
            scenarios = data['scenarios']
        else:
            scenarios = data
    
    print(f"📚 Loaded {len(scenarios)} scenarios")
    
    # Initialize improved embedding generator
    embedder = ImprovedEmbeddingGenerator()
    
    # Generate improved text representations
    enhanced_texts = []
    for scenario in scenarios:
        enhanced_text = embedder.create_enhanced_text(scenario)
        enhanced_texts.append(enhanced_text)
    
    print("🚀 Generating improved embeddings...")
    
    # Generate ensemble embeddings
    improved_embeddings = embedder.generate_ensemble_embeddings(enhanced_texts)
    
    print(f"✅ Generated embeddings shape: {improved_embeddings.shape}")
    
    # Save improved embeddings
    np.save('improved_atomic_embeddings.npy', improved_embeddings)
    
    # Save enhanced texts for reference
    enhanced_data = []
    for i, scenario in enumerate(scenarios):
        enhanced_data.append({
            'scenario_id': scenario['id'],
            'original_description': scenario['description'],
            'enhanced_text': enhanced_texts[i],
            'embedding_index': i
        })
    
    with open('enhanced_atomic_scenarios.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print("💾 Saved improved embeddings and enhanced texts")
    print("📊 Ready to test improved performance!")
    
    return improved_embeddings, enhanced_data


if __name__ == "__main__":
    rebuild_improved_indices()
