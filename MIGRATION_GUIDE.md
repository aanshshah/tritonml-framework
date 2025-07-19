# Migration Guide: From Emotion Classifier to TritonML Framework

This guide shows how to migrate from the original emotion classifier implementation to the TritonML framework.

## Before (Original Implementation)

### Client Code
```python
# src/emotion_classifier/client.py
from src.emotion_classifier import EmotionClassifierClient

client = EmotionClassifierClient()
emotion = client.classify_emotion("I love this!")
```

### Deployment Process
```bash
# Manual steps required:
python scripts/setup_model.py
python scripts/quantize_model.py
docker run -p 8000:8000 -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:24.08-py3
```

## After (TritonML Framework)

### Client Code
```python
# Using TritonML
from tritonml import deploy

# One-line deployment
client = deploy("cardiffnlp/twitter-roberta-base-emotion")
emotion = client.predict("I love this!")
```

### Deployment Process
```bash
# Single command
tritonml deploy cardiffnlp/twitter-roberta-base-emotion
```

## Feature Comparison

| Feature | Original | TritonML |
|---------|----------|----------|
| Model Loading | Manual | `TritonModel.from_huggingface()` |
| Conversion | Separate script | `model.convert()` |
| Quantization | Separate script | `model.quantize()` |
| Deployment | Manual Docker | `model.deploy()` |
| Multiple Models | Not supported | Built-in registry |
| CLI Tools | None | Full CLI |
| Benchmarking | Custom tests | `model.benchmark()` |

## Step-by-Step Migration

### 1. Install TritonML

```bash
pip install -e .  # From the tritonml directory
```

### 2. Update Imports

Replace:
```python
from src.emotion_classifier import EmotionClassifierClient
from src.emotion_classifier.config import EMOTION_LABELS
```

With:
```python
from tritonml.tasks import EmotionClassifier
```

### 3. Update Model Loading

Replace:
```python
# Original
client = EmotionClassifierClient()
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
```

With:
```python
# TritonML
model = EmotionClassifier.from_pretrained()
client = model.deploy()
```

### 4. Update Inference Code

Replace:
```python
# Original
tokenized = tokenizer(text, ...)
inputs = prepare_inputs(tokenized)
response = client.infer(...)
emotion = process_response(response)
```

With:
```python
# TritonML
emotion = model.predict(text)
# Or get probabilities
probs = model.predict_proba(text)
```

### 5. Update Tests

Replace:
```python
# Original test
def test_emotion_classifier():
    client = EmotionClassifierClient()
    # Manual setup...
```

With:
```python
# TritonML test
def test_emotion_classifier():
    model = EmotionClassifier.from_pretrained()
    model.deploy()
    assert model.predict("I'm happy!") == "joy"
```

## Advanced Migration

### Custom Preprocessing

If you have custom preprocessing:

```python
class CustomEmotionClassifier(EmotionClassifier):
    def preprocess(self, inputs):
        # Your custom preprocessing
        processed = super().preprocess(inputs)
        # Additional processing
        return processed
```

### Custom Configuration

```python
from tritonml.core.config import TextClassificationConfig

config = TextClassificationConfig(
    model_name="emotion-classifier-v2",
    max_sequence_length=256,  # Increased from 128
    max_batch_size=64,        # Increased from 32
    labels=["anger", "joy", "optimism", "sadness", "neutral"]  # Added neutral
)

model = EmotionClassifier(config)
```

### Backward Compatibility

To maintain backward compatibility, create a wrapper:

```python
# backward_compat.py
from tritonml.tasks import EmotionClassifier

class EmotionClassifierClient:
    """Backward compatible client."""
    
    def __init__(self, server_url="localhost:8000"):
        self.model = EmotionClassifier.from_pretrained()
        self.model.deploy(server_url)
    
    def classify_emotion(self, text):
        return self.model.predict(text)
    
    def tokenize_text(self, text):
        return self.model.preprocess(text)

# Legacy functions
def process_response(response):
    # Delegate to model
    return response

def call_triton(text):
    model = EmotionClassifier.from_pretrained()
    model.deploy()
    return model.predict(text)
```

## Benefits After Migration

1. **Simplified API**: Less boilerplate code
2. **Automatic Optimization**: Built-in quantization and optimization
3. **Better Testing**: Easier to mock and test
4. **Multi-Model Support**: Deploy multiple models easily
5. **CLI Tools**: Command-line deployment and testing
6. **Docker Generation**: Automatic Docker package creation
7. **Extensibility**: Easy to add new model types

## Example: Complete Migration

Original workflow:
```python
# 1. Setup
client = EmotionClassifierClient()

# 2. Classify
emotion = client.classify_emotion("I'm excited!")

# 3. Batch processing
emotions = []
for text in texts:
    emotion = client.classify_emotion(text)
    emotions.append(emotion)
```

TritonML workflow:
```python
# 1. Setup
model = EmotionClassifier.from_pretrained()
model.deploy()

# 2. Classify
emotion = model.predict("I'm excited!")

# 3. Batch processing (automatic batching!)
emotions = model.predict(texts)

# 4. Additional features
probs = model.predict_proba(texts)
results = model.benchmark(texts[:100])
```

## Troubleshooting

### Import Errors
If you get import errors, ensure the framework is installed:
```bash
pip install -e . --force-reinstall
```

### Model Not Found
The framework looks for models in `./models` by default. Set:
```bash
export MODEL_REPOSITORY_PATH=/path/to/your/models
```

### Server Connection
Ensure Triton server is running:
```bash
docker-compose up  # Using framework's docker-compose
```

## Next Steps

1. Explore other model types (sentiment analysis, NER, etc.)
2. Try different quantization methods
3. Use the CLI for deployment automation
4. Contribute new model types to the framework!