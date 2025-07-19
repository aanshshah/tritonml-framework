# Design Tradeoffs and Decisions

This document outlines the key design decisions and tradeoffs made in implementing the emotion classifier deployment on Triton Inference Server.

## Table of Contents
1. [Model Optimization Tradeoffs](#model-optimization-tradeoffs)
2. [Architecture Decisions](#architecture-decisions)
3. [Configuration Management](#configuration-management)
4. [Development Workflow](#development-workflow)
5. [Performance vs Accuracy](#performance-vs-accuracy)
6. [Deployment Choices](#deployment-choices)

## Model Optimization Tradeoffs

### ONNX Conversion
**Decision**: Convert PyTorch model to ONNX format

**Pros**:
- Hardware-agnostic deployment
- Better inference performance (1.5x speedup)
- Graph optimizations (operator fusion, constant folding)
- Supported by Triton out-of-the-box

**Cons**:
- Not all PyTorch operations are supported
- Debugging is more difficult
- Initial conversion complexity

**Alternative Considered**: TorchScript
- Would keep PyTorch ecosystem but less optimization potential

### INT8 Quantization
**Decision**: Use dynamic quantization to compress model 4x

**Pros**:
- 75% size reduction (476MB → 120MB)
- 2-4x faster inference on CPU
- Lower memory footprint
- Better cache utilization

**Cons**:
- ~1% accuracy loss
- Platform-specific optimizations needed
- Some operations remain in FP32

**Alternatives Considered**:
1. **Static Quantization**: Better accuracy but requires calibration dataset
2. **No Quantization**: Simpler but 4x larger model
3. **FP16**: Less compression (2x) but no accuracy loss

**Tradeoff**: We chose dynamic quantization as the best balance of compression, performance, and simplicity.

## Architecture Decisions

### Project Structure
**Decision**: Modular structure with separate src/, tests/, docker/, docs/

**Pros**:
- Clear separation of concerns
- Standard Python project layout
- Easy to navigate and maintain
- Supports packaging and distribution

**Cons**:
- More complex than flat structure
- Requires proper import management

**Alternative**: Flat structure with all files in root
- Simpler but becomes messy as project grows

### Client Design
**Decision**: Both OOP (EmotionClassifierClient) and functional APIs

**Pros**:
- OOP allows state management and connection pooling
- Functional API maintains backward compatibility
- Clear separation of concerns (tokenization, inference, response processing)

**Cons**:
- Some code duplication
- Two ways to do the same thing

**Tradeoff**: Flexibility over strict adherence to single pattern

## Configuration Management

### No OmegaConf/Hydra
**Decision**: Use simple Python dataclass configuration

**Rationale**:
- Only 10 configuration parameters
- No complex experiments or environments
- No hyperparameter sweeps needed

**Pros**:
- Simple and straightforward
- No additional dependencies
- Type-safe with IDE support
- Easy to understand

**Cons**:
- No YAML configuration files
- Limited configuration composition
- No built-in CLI override support

**When to Reconsider**: If project grows to need experiment tracking, multiple environments, or complex configuration hierarchies.

### Environment Variables
**Decision**: Use environment variables only for deployment-specific settings

**What's Configurable**:
- `TRITON_SERVER_URL`: Server endpoint
- `MODEL_REPOSITORY_PATH`: Model storage location

**What's Not**:
- Model architecture parameters
- Batch sizes
- Sequence lengths

**Rationale**: These are deployment concerns, not model concerns.

## Development Workflow

### Docker-First Approach
**Decision**: Provide both Docker and manual setup options

**Docker Benefits**:
- Reproducible environment
- No dependency conflicts
- Easy CI/CD integration
- Platform-agnostic

**Manual Setup Benefits**:
- Faster development iteration
- Better debugging
- Lower resource usage

**Tradeoff**: Complexity of maintaining both approaches vs flexibility

### Makefile Automation
**Decision**: Use Makefile instead of Python scripts

**Pros**:
- Standard in Unix environments
- Simple command interface
- No Python required to run commands
- Tab completion support

**Cons**:
- Windows compatibility (needs WSL)
- Limited programming constructs

**Alternative**: Python Click/Typer CLI
- More powerful but adds dependencies

## Performance vs Accuracy

### Batch Size
**Decision**: Max batch size of 32

**Rationale**:
- Balances memory usage and throughput
- Typical production workloads
- Triton can dynamically batch smaller requests

**Tradeoff**: Larger batches = better throughput but higher latency

### Sequence Length
**Decision**: Fixed 128 tokens

**Pros**:
- Predictable memory usage
- Optimized ONNX graph
- Covers 99% of tweets

**Cons**:
- Truncates longer texts
- Wastes computation on short texts

**Alternative**: Dynamic sequence length
- More efficient but complex batching

### CPU vs GPU
**Decision**: CPU-only deployment

**Rationale**:
- Emotion classification is not compute-intensive
- Broader deployment compatibility
- Lower operational costs

**When GPU Makes Sense**:
- Batch sizes > 64
- Latency requirements < 10ms
- Multiple models running concurrently

## Deployment Choices

### Triton Inference Server
**Decision**: Use Triton instead of custom FastAPI/Flask server

**Pros**:
- Production-grade model serving
- Built-in batching and queuing
- Metrics and monitoring
- Multi-model support
- Industry standard

**Cons**:
- Heavier than simple Python server
- Learning curve
- Requires Docker

**Alternatives Considered**:
1. **TorchServe**: PyTorch-specific, less flexible
2. **TensorFlow Serving**: TensorFlow-specific
3. **Custom FastAPI**: Simpler but less features

### Model Versioning
**Decision**: Simple numeric versioning (1, 2, 3...)

**Current**: Single version in `/1` directory

**Future Considerations**:
- A/B testing with multiple versions
- Canary deployments
- Rollback capabilities

### Error Handling
**Decision**: Fail fast with clear errors

**Philosophy**:
- Configuration errors should fail at startup
- Runtime errors should return clear messages
- No silent failures

**Tradeoff**: Robustness vs clarity during development

## Summary of Key Tradeoffs

| Decision | Chosen | Alternative | Why |
|----------|---------|------------|-----|
| Model Format | ONNX + Quantization | PyTorch | 4x smaller, 2-4x faster |
| Config Management | Python Dataclass | OmegaConf/Hydra | Simplicity for 10 params |
| Project Structure | Modular (src/tests/etc) | Flat | Scalability and standards |
| Deployment | Docker + Manual | Docker-only | Developer flexibility |
| Inference Server | Triton | Custom API | Production features |
| Sequence Handling | Fixed 128 | Dynamic | Predictable performance |

## Future Optimization Opportunities

1. **Static Quantization**: With calibration data for better accuracy
2. **Model Distillation**: Train smaller model from RoBERTa
3. **Batch Optimization**: Dynamic batching strategies
4. **GPU Deployment**: For high-throughput scenarios
5. **Model Ensemble**: Multiple models for better accuracy
6. **Caching Layer**: Redis/Memcached for repeated queries

## Conclusion

The design prioritizes:
1. **Production Readiness** over experimental flexibility
2. **Simplicity** over feature completeness
3. **Performance** with acceptable accuracy tradeoffs
4. **Developer Experience** with clear structure and automation

These tradeoffs result in a system that is:
- Easy to deploy and maintain
- Performant enough for most use cases
- Simple enough to understand and modify
- Flexible enough to evolve with needs