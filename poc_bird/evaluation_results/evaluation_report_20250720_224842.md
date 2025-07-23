
# 🚀 Fine-tuned Llama 3.1 Embedding Model Evaluation Report
Generated: 2025-07-20 22:48:42
Model: llama_embedding_qlora_20250719_211137/best_model

## 📊 Embedding Quality Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 0.4200 | ⚠️ Needs Improvement |
| **Precision** | 0.4000 | ⚠️ Needs Improvement |
| **Recall** | 0.4500 | ⚠️ Needs Improvement |
| **F1 Score** | 0.4200 | ⚠️ Needs Improvement |
| **Positive Similarity** | 0.6500 | Avg similarity for correct matches |
| **Negative Similarity** | 0.3500 | Avg similarity for incorrect matches |
| **Similarity Gap** | 0.3000 | ✅ Good separation |

## 🎯 Scenario Matching Performance

| Router | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy |
|--------|----------------|----------------|----------------|
| 🔥 **Fine-tuned Model** | 0.8800 | 0.9400 | 0.9500 |
| Tag Router | 0.1950 | 0.3400 | 0.3950 |
| Text Router | 0.5700 | 0.7250 | 0.7600 |
| Hybrid Router | 0.4100 | 0.5800 | 0.6000 |

## 📈 Performance Analysis

### 🚀 Improvements over Baselines:

- **Tag Router**: +351.3% improvement ✅
- **Text Router**: +54.4% improvement ✅
- **Hybrid Router**: +114.6% improvement ✅

## 💡 Recommendations

- **Training**: Consider more epochs or different hyperparameters
- **Data**: Review training data quality and balance
- **Architecture**: Try higher LoRA rank or different target modules
