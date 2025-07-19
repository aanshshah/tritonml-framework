# Triton Inference Server Setup with Docker

This guide provides Docker-based setup instructions for deploying the `cardiffnlp/twitter-roberta-base-emotion` model on NVIDIA Triton Inference Server with automatic model compression.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB of free disk space (model will be compressed to 120MB)

## Quick Start (Docker Compose)

The easiest way to get started is using Docker Compose, which will handle all setup automatically:

```bash
# Clone the repository
git clone <repository-url>
cd mle-candidate-ashah

# Build and run all services
docker-compose up --build
```

This will:
1. Build a Docker image with all Python dependencies
2. Download and convert the model to ONNX format
3. Apply INT8 quantization for 4x compression (476MB → 120MB)
4. Start the Triton Inference Server
5. Run the test suite automatically

## Manual Docker Setup

If you prefer to run the services separately:

### 1. Build the Client Docker Image

```bash
docker build -t emotion-classifier-client .
```

This will:
- Install all Python dependencies
- Download the HuggingFace model
- Convert it to ONNX format with INT8 quantization
- Save the compressed model (120MB) in the `models/` directory

### 2. Run Triton Server

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

### 3. Run Tests

```bash
docker run --rm \
  --network host \
  -v $(pwd):/app \
  emotion-classifier-client \
  python test_emotion_classifier.py
```

## Docker Configuration Details

### Dockerfile

The Dockerfile:
- Uses Python 3.11 as the base image
- Installs all required dependencies from `requirements.txt`
- Runs the model conversion script during build
- Creates the necessary directory structure
- Copies all required files

### Docker Compose

The `docker-compose.yml` file:
- Defines two services: `triton-server` and `emotion-classifier-client`
- Sets up a custom network for service communication
- Includes health checks to ensure proper startup order
- Maps all necessary ports
- Mounts volumes for model files

## Environment Variables

The client supports the following environment variable:
- `TRITON_SERVER_URL`: URL of the Triton server (default: `localhost:8000`)

When using Docker Compose, this is automatically set to `triton-server:8000`.

## Accessing the Services

Once running, you can access:
- HTTP endpoint: `http://localhost:8000`
- gRPC endpoint: `http://localhost:8001`
- Metrics endpoint: `http://localhost:8002`

## Health Check

Check if the server is ready:
```bash
curl http://localhost:8000/v2/health/ready
```

## Stopping Services

### Docker Compose
```bash
docker-compose down
```

### Manual Docker
```bash
docker stop triton-server
```

## Rebuilding After Changes

If you make changes to the code:

```bash
# With Docker Compose
docker-compose up --build

# Manual Docker
docker build -t emotion-classifier-client .
```

## Production Considerations

For production deployment:

1. **Resource Limits**: Add resource constraints in docker-compose.yml:
   ```yaml
   services:
     triton-server:
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: '2'
   ```

2. **GPU Support**: For GPU inference, add:
   ```yaml
   services:
     triton-server:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

3. **Logging**: TODO: Configure logging drivers for better observability

4. **Persistence**: TODO: Use named volumes for model storage

## Troubleshooting

### Build Issues
- Ensure Docker has enough memory allocated (at least 4GB)
- Check internet connectivity for model downloads
- Verify Docker daemon is running

### Runtime Issues
- Check container logs: `docker-compose logs`
- Verify port availability: `netstat -an | grep 8000`
- Ensure model files were created: `ls -la models/emotion-classifier/1/`

### Network Issues
- Use `docker network ls` to verify the network exists
- Check firewall settings for Docker
- Try using `--network host` for debugging

## Architecture Benefits

Using Docker provides:
- **Reproducibility**: Consistent environment across different systems
- **Isolation**: Dependencies don't conflict with host system
- **Scalability**: Easy to deploy multiple instances
- **Portability**: Runs on any system with Docker

## Next Steps

1. Customize the model configuration in `models/emotion-classifier/config.pbtxt`
2. Add more models to the repository
3. Implement load balancing for multiple Triton instances
4. Add monitoring and logging solutions