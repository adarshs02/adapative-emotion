# Emobird: Dynamic Emotion Analysis System

Emobird is a revolutionary emotion analysis system that generates scenarios and Conditional Probability Tables (CPTs) dynamically at inference time, rather than relying on pre-stored scenarios and CPT files.

## Architecture Overview

The system consists of several key components:

### Core Components

1. **Emobird Engine** (`emobird.py`)
   - Main inference engine that orchestrates the entire process
   - Manages LLM loading and coordinates between components
   - Provides the primary `analyze_emotion()` interface

2. **Scenario Generator** (`scenario_generator.py`)
   - Dynamically generates scenario descriptions from user input
   - Creates contextual information and tags for situations
   - Uses LLM to understand and categorize emotional contexts

3. **CPT Generator** (`cpt_generator.py`)
   - Generates Conditional Probability Tables dynamically
   - Identifies relevant psychological factors for each scenario
   - Creates emotion probability distributions for factor combinations

4. **Configuration** (`config.py`)
   - Centralized configuration management
   - Supports environment variables and config files
   - Handles model parameters and system settings

## Key Features

- **Dynamic Generation**: No pre-stored scenarios or CPTs required
- **Real-time Inference**: Everything generated at query time
- **Flexible Configuration**: Easily customizable parameters
- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to add new components or modify existing ones

## Installation

1. Clone the repository and navigate to the emobird directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from emobird import Emobird

# Initialize the system
emobird = Emobird()

# Analyze a situation
situation = "I just got promoted at work after working really hard for months"
result = emobird.analyze_emotion(situation)

# Access results
print(f"Scenario: {result['scenario']['description']}")
print(f"Factors: {result['factors']}")
print(f"Emotions: {result['emotions']}")
```

### Running Examples

```bash
python example.py
```

This will provide interactive options to:
- Analyze custom situations
- Run pre-defined example situations
- See detailed emotion analysis results

### Key Configuration Options

- `llm_model_name`: The language model to use for generation
- `max_cpt_entries`: Maximum number of CPT entries to generate
- `temperature`: Generation temperature for LLM
- `use_bayesian_calibration`: Enable Bayesian probability calibration
- `device`: Compute device ("cuda", "cpu", or "auto")

## System Flow

1. **User Input**: User provides situation description
2. **Scenario Generation**: System generates relevant scenario context
3. **Factor Identification**: Key psychological factors are identified
4. **CPT Generation**: Probability table created for factor combinations
5. **Factor Extraction**: User situation analyzed for specific factor values
6. **Emotion Calculation**: Final emotion probabilities computed
7. **Optional Calibration**: Bayesian calibration applied if enabled

## Advantages Over Previous System

- **No Pre-computation**: No need to generate and store scenario databases
- **Unlimited Scenarios**: Can handle any novel situation dynamically
- **Real-time Adaptation**: Adapts to new types of situations automatically
- **Simplified Pipeline**: Fewer moving parts and dependencies
- **Better Scalability**: No storage limitations for scenarios

## Extending the System

### Adding New Factors

Modify the `_get_default_factors()` method in `CPTGenerator` to include new psychological factors.

### Custom Emotion Sets

Update the `default_emotions` list in `EmobirdConfig` to use different emotion categories.

### New Generation Models

Simply change the `llm_model_name` in configuration to use different language models.

## Performance Notes

- First run will be slower due to model loading
- Generation time depends on the complexity of the situation
- CPT generation is limited to prevent exponential explosion
- GPU usage recommended for better performance

## Future Enhancements

- **Bayesian Calibration**: Implement the BIRD upgrade for probability calibration
- **Caching**: Add caching for frequently analyzed situations
- **Batch Processing**: Optimize for analyzing multiple situations
- **Multi-modal Input**: Support for image/audio situation descriptions
- **Fine-tuning**: Custom model fine-tuning for domain-specific applications

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Other dependencies listed in `requirements.txt`

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]
