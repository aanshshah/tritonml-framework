# TritonML Framework

A powerful framework for deploying machine learning models to NVIDIA Triton Inference Server with built-in optimizations, quantization, and easy-to-use APIs.

## Features

- 🚀 **Easy Deployment**: Deploy any HuggingFace model with a single command
- 🔧 **Automatic Optimization**: Built-in quantization and optimization for 4x model compression
- 🎯 **Task-Specific Models**: Pre-built support for text classification, image classification, and more
- 📦 **Model Conversion**: Automatic conversion to ONNX, TorchScript, or TensorRT
- 🔌 **Simple API**: Intuitive Python API and CLI tools
- 🐳 **Docker Ready**: Generate Docker deployment packages automatically
- 📊 **Benchmarking**: Built-in performance benchmarking tools

## Installation

```bash
pip install tritonml
```

Or install from source:

```bash
git clone https://github.com/aaanshshah/tritonml
cd tritonml
pip install -e .
```

## Quick Start

### 1. Deploy a Model in 3 Lines

```python
from tritonml import deploy

# Deploy any HuggingFace model
client = deploy("cardiffnlp/twitter-roberta-base-emotion")

# Make predictions
result = client.predict("I love this framework!")
print(result)  # Output: "joy"
```

### 2. Using the CLI

```bash
# Deploy a model
tritonml deploy cardiffnlp/twitter-roberta-base-emotion --server localhost:8000

# Make predictions
tritonml predict emotion-classifier "I'm so happy!" --server localhost:8000

# Benchmark performance
tritonml benchmark emotion-classifier --batch-sizes 1,8,16,32
```

## Core Concepts

### TritonModel

The base class for all deployable models:

```python
from tritonml import TritonModel

# Load a model from HuggingFace
model = TritonModel.from_huggingface(
    "bert-base-uncased",
    task="text-classification"  # Auto-detected if not specified
)

# Convert and optimize
model.convert()              # Convert to ONNX
model.quantize()            # Apply INT8 quantization
model.optimize()            # Apply graph optimizations

# Deploy
client = model.deploy(server_url="localhost:8000")

# Use the model
result = model.predict("Hello world!")
```

### Task-Specific Models

Pre-configured models for common tasks:

```python
from tritonml.tasks import TextClassificationModel, EmotionClassifier

# Generic text classification
model = TextClassificationModel.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    labels=["negative", "positive"]
)

# Specialized emotion classifier
emotion_model = EmotionClassifier.from_pretrained()
emotions = emotion_model.predict([
    "I'm furious!",
    "Best day ever!",
    "Things will improve",
    "Feeling down..."
])
```

### Model Conversion

Convert models to optimized formats:

```python
from tritonml.core.converter import get_converter

# Get appropriate converter
converter = get_converter("onnx", model, config)

# Convert with options
converter.convert(
    output_path="./models/my-model",
    opset_version=14,
    optimize_for_gpu=True
)

# Quantize for better performance
converter.quantize(
    method="dynamic",  # or "static" with calibration data
    per_channel=True
)
```

## Benchmarking

TritonML now supports benchmarking models with Hugging Face datasets:

### Basic Benchmarking

```python
from tritonml import TextClassificationModel, BenchmarkRunner, HuggingFaceDatasetLoader

# Load and deploy your model
model = TextClassificationModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.deploy()

# Create benchmark runner
runner = BenchmarkRunner(model)

# Load a dataset
dataset_loader = HuggingFaceDatasetLoader("imdb", split="test")

# Run benchmark
results = runner.benchmark_dataset(
    dataset_loader,
    batch_sizes=[1, 8, 16, 32],
    num_samples=1000
)

# Print summary
runner.print_summary()

# Save results
runner.save_results("benchmark_results.json")
```

### CLI Benchmarking

Use the CLI to benchmark deployed models with Hugging Face datasets:

```bash
# Benchmark with IMDB dataset
tritonml benchmark my-model --dataset imdb --num-samples 1000 --output results.json

# Custom batch sizes
tritonml benchmark my-model --dataset emotion --batch-sizes "1,4,8,16" --output results.csv
```

### Multiple Datasets

Benchmark across multiple datasets:

```python
# Define dataset configurations
datasets = [
    {"dataset_name": "imdb", "split": "test"},
    {"dataset_name": "rotten_tomatoes", "split": "test"},
    {"dataset_name": "emotion", "split": "test"}
]

# Run benchmarks
results = runner.benchmark_multiple_datasets(
    datasets,
    batch_sizes=[1, 8, 16],
    num_samples=500
)
```

### Available Datasets

Popular datasets for benchmarking:

**Text Classification:**
- `imdb` - Movie review sentiment
- `rotten_tomatoes` - Movie reviews
- `emotion` - Emotion classification
- `ag_news` - News categorization
- `tweet_eval` - Tweet sentiment

**Other Tasks:**
- See `HuggingFaceDatasetLoader.list_popular_datasets()` for more

## Advanced Usage

### Custom Models

Create custom model implementations:

```python
from tritonml.core.model import TritonModel
from tritonml.core.config import TritonConfig

class MyCustomModel(TritonModel):
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        # Load your model
        config = TritonConfig(
            model_name="my-model",
            input_shapes={"input": [512]},
            output_shapes={"output": [10]}
        )
        return cls(config)
    
    def preprocess(self, inputs):
        # Custom preprocessing
        return {"input": process_inputs(inputs)}
    
    def postprocess(self, outputs):
        # Custom postprocessing
        return outputs["output"].argmax()
```

### Deployment Configuration

Fine-tune deployment settings:

```python
from tritonml.core.config import TritonConfig

config = TritonConfig(
    model_name="my-model",
    max_batch_size=64,
    instance_group={"kind": "KIND_GPU", "count": 2},
    dynamic_batching={
        "preferred_batch_size": [8, 16, 32],
        "max_queue_delay_microseconds": 100
    }
)

model = MyCustomModel(config)
```

### Docker Deployment

Generate complete Docker deployment packages:

```python
from tritonml.deploy.docker import create_deployment_package

create_deployment_package(
    model_name="emotion-classifier",
    output_path="./deploy",
    include_client=True
)
```

This creates:
- `Dockerfile` - Custom Triton server image
- `docker-compose.yml` - Complete deployment configuration
- `client_example.py` - Example client code
- `README.md` - Deployment instructions

### Benchmarking

Benchmark model performance:

```python
# Built-in benchmarking
results = model.benchmark(
    test_inputs=["sample text"] * 100,
    batch_sizes=[1, 8, 16, 32, 64]
)

for batch_size, metrics in results.items():
    print(f"{batch_size}: {metrics['avg_latency_ms']:.2f}ms, "
          f"{metrics['throughput']:.2f} samples/sec")
```

## Architecture

TritonML follows a modular architecture:

```
tritonml/
├── core/               # Core framework components
│   ├── model.py       # Base TritonModel class
│   ├── client.py      # Enhanced Triton client
│   ├── converter.py   # Model conversion utilities
│   └── config.py      # Configuration management
├── tasks/              # Task-specific implementations
│   ├── text_classification.py
│   ├── image_classification.py
│   └── converters/    # Task-specific converters
├── utils/              # Utility functions
├── deploy/             # Deployment utilities
└── cli/                # Command-line interface
```

## Supported Models

### Text Models
- BERT, RoBERTa, DistilBERT, ALBERT
- GPT-2, GPT-Neo, T5 (coming soon)
- Any HuggingFace `AutoModelForSequenceClassification`

### Image Models (coming soon)
- Vision Transformer (ViT)
- ResNet, EfficientNet
- Any torchvision model

### Custom Models
- ONNX models
- TorchScript models
- TensorFlow SavedModel (coming soon)

## Performance

TritonML automatically applies optimizations:

- **Quantization**: 4x model size reduction with INT8
- **Graph Optimization**: ONNX runtime optimizations
- **Batching**: Dynamic batching for better throughput
- **Multi-Instance**: GPU/CPU instance scaling

Example results for emotion classification:
- Original model: 476MB
- Quantized model: 120MB (4x compression)
- Latency: 2-4x faster inference
- Accuracy: 93.8% maintained

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of:
- NVIDIA Triton Inference Server
- HuggingFace Transformers
- ONNX Runtime
- Microsoft Optimum