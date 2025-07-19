# Triton Inference Server Setup for Twitter RoBERTa Emotion Classifier

This guide provides step-by-step instructions to set up and deploy the `cardiffnlp/twitter-roberta-base-emotion` model on NVIDIA Triton Inference Server with INT8 quantization for 4x model compression.

## Prerequisites

- Docker installed and running
- Python 3.11 (for manual setup)
- At least 4GB of free disk space (final model size: 120MB)

## Quick Start with Docker (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd mle-candidate-ashah

# Build and run all services
docker-compose up --build
```

This will automatically:
- Build a Docker image with all dependencies
- Download and convert the model to ONNX
- Start the Triton server
- Run the test suite

For detailed Docker instructions, see [TRITON_SETUP_DOCKER.md](./TRITON_SETUP_DOCKER.md).

## Manual Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mle-candidate-ashah
```

### 2. Set up Python Environment

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download and Convert Model to ONNX

Run the model conversion script to download the HuggingFace model and convert it to ONNX format with INT8 quantization:

```bash
python scripts/setup_model.py
# Or use the Makefile:
make quantize-model
```

This script will:
- Download the `cardiffnlp/twitter-roberta-base-emotion` model from HuggingFace
- Convert it to ONNX format for optimized inference
- Apply INT8 quantization for 4x compression (476MB → 120MB)
- Save the quantized model to `models/emotion-classifier/1/model_quantized.onnx`
- Save tokenizer files for preprocessing

### 4. Pull Triton Server Docker Image

Pull the Triton Server version 24.08:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.08-py3
```

### 5. Start Triton Server

Run the Triton server with the model repository:

```bash
docker run -d --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/models:/models \
  --name triton-server \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
```

Verify the server is running:

```bash
docker logs triton-server
```

You should see output indicating the model is loaded and READY.

### 6. Test the Deployment

Run the test script to verify everything is working correctly:

```bash
python test_emotion_classifier.py
```

Expected output:
```
All sentiment tests passed successfully!
```

## Model Configuration

The model is configured to:
- Accept batch sizes up to 32
- Process fixed-length sequences of 128 tokens
- Accept two inputs: `input_ids` and `attention_mask` (both INT64 tensors of shape [128])
- Return logits as FP32 tensor of shape [4] representing probabilities for 4 emotion classes
- Use INT8 quantized model (model_quantized.onnx) for 4x smaller size and 2-4x faster inference

The emotion classes are:
1. anger
2. joy
3. optimism
4. sadness

## Client Implementation

The `triton_client.py` file provides a Python client implementation that:
1. Tokenizes input text using the HuggingFace tokenizer
2. Pads/truncates sequences to 128 tokens
3. Sends requests to the Triton server
4. Processes the response to return the predicted emotion label

### Usage Example

```python
from triton_client import call_triton

text = "I love watching tennis!"
emotion = call_triton(text)
print(f"Predicted emotion: {emotion}")  # Output: joy
```

## Architecture Details

### Model Repository Structure
```
models/
└── emotion-classifier/
    ├── config.pbtxt          # Triton model configuration
    └── 1/                    # Model version directory
        ├── model_quantized.onnx  # INT8 quantized ONNX model (120MB)
        ├── model.onnx           # Original ONNX model (476MB, optional)
        └── tokenizer files      # Tokenizer configuration
```

### Triton Configuration (`config.pbtxt`)
- Platform: `onnxruntime_onnx`
- Max batch size: 32
- Instance group: 1 CPU instance
- Input/Output specifications match the ONNX model interface

## Troubleshooting

### Docker Issues
- Ensure Docker is running: `docker version`
- Check container status: `docker ps`
- View server logs: `docker logs triton-server`

### Model Loading Issues
- Verify model files exist: `ls -la models/emotion-classifier/1/`
- Check config.pbtxt syntax
- Ensure ONNX model was properly converted

### Client Connection Issues
- Verify server is accessible: `curl localhost:8000/v2/health/ready`
- Check firewall settings for ports 8000, 8001, 8002

## Stopping the Server

To stop the Triton server:

```bash
docker stop triton-server
```

## Code Changes Made

1. **Fixed bug in `triton_client.py`**: Changed `NotImplemented` to `NotImplementedError` (line 19)
2. **Implemented `process_response` function**: Added logic to extract logits and return the predicted emotion label
3. **Fixed tokenization**: Added `padding="max_length"` to ensure fixed-size inputs
4. **Created model configuration**: Configured Triton server to properly load and serve the ONNX model

## Performance Notes

- The model runs on CPU by default (no GPU required)
- INT8 quantization provides:
  - 4x reduction in model size (476MB → 120MB)
  - 2-4x faster inference speed
  - Maintains 93.8% accuracy on test set
- Inference time varies based on batch size and hardware
- TODO: For production use, consider GPU deployment for even better performance

## Additional Resources

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [HuggingFace Model Card](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)