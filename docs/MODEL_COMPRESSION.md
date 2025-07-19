# Model Compression Guide

This document explains how the emotion classifier model is compressed for efficient deployment.

## Compression Techniques Used

### 1. ONNX Conversion
The model is first converted from PyTorch to ONNX format, which provides:
- Graph optimizations (operator fusion, constant folding)
- Dead code elimination
- More efficient runtime execution
- Cross-platform compatibility

### 2. Dynamic Quantization
The model is then quantized to INT8 precision, which provides:
- **4x size reduction**: From 476MB to 120MB
- **2-4x faster inference**: Integer operations are faster than floating-point
- **Minimal accuracy loss**: Typically less than 1%

## How Quantization Works

Quantization reduces the precision of model weights from FP32 (32-bit floating-point) to INT8 (8-bit integer):

```
Original weight: 0.7264819 (32 bits)
Quantized weight: 93 (8 bits) → dequantized to ~0.7265
```

### Quantization Process

1. **Analyze weight distribution**: Find min/max values for each layer
2. **Calculate scale and zero-point**: Map FP32 range to INT8 range
3. **Quantize weights**: Convert FP32 weights to INT8
4. **Store quantization parameters**: Keep scale/zero-point for inference

### Dynamic vs Static Quantization

We use **dynamic quantization** which:
- Quantizes weights at conversion time
- Keeps activations in FP32 during inference
- No calibration dataset required
- Good balance of performance and accuracy

## Running Quantization

### Automatic (during setup)
The model is automatically quantized when you run:
```bash
make convert-model
```

### Manual Quantization
To re-quantize an existing model:
```bash
make quantize-model
```

### Custom Quantization
To convert without quantization:
```python
from src.emotion_classifier.converter import convert_model_to_onnx

# Convert without quantization (larger model)
convert_model_to_onnx(quantize=False)
```

## Performance Comparison

| Metric | Original PyTorch | ONNX | Quantized ONNX |
|--------|-----------------|------|----------------|
| Model Size | ~500MB | 476MB | 120MB |
| Inference Speed | 1x | 1.5x | 2-4x |
| Memory Usage | High | Medium | Low |
| Accuracy | 100% | 100% | >99% |

## Quantization Configuration

The quantization uses ARM64-optimized settings (in `converter.py`):
```python
qconfig = AutoQuantizationConfig.arm64(
    is_static=False,    # Dynamic quantization
    per_channel=True    # Per-channel quantization for better accuracy
)
```

### Platform-Specific Optimization

Different quantization configs for different platforms:
- `arm64`: For Apple Silicon and ARM servers
- `avx512_vnni`: For Intel CPUs with AVX-512
- `avx2`: For older Intel/AMD CPUs

## Verifying Quantization

After quantization, the model is tested to ensure it works correctly:
1. Load a test sentence
2. Run inference
3. Verify output shape and predictions
4. Compare with original model (if needed)

## Trade-offs

### Advantages
- **75% smaller model size**
- **2-4x faster inference**
- **Lower memory usage**
- **Better cache utilization**

### Disadvantages
- **Slight accuracy loss** (<1%)
- **Platform-specific optimizations needed**
- **Some operations may not support INT8**

## Advanced Compression Options

For even more compression:

1. **Static Quantization**: Requires calibration data but better accuracy
2. **QAT (Quantization Aware Training)**: Retrain with quantization
3. **Pruning**: Remove less important connections
4. **Knowledge Distillation**: Train a smaller model

## Troubleshooting

### Model Not Loading
- Ensure `config.pbtxt` specifies correct filename: `default_model_filename: "model_quantized.onnx"`

### Accuracy Issues
- Try per-channel quantization
- Use static quantization with calibration
- Keep certain layers in FP32

### Performance Issues
- Verify CPU supports required instructions (AVX, VNNI)
- Use platform-specific quantization config
- Check batch size optimization