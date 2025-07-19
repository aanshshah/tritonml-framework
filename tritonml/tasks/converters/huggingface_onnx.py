"""HuggingFace to ONNX converter implementation."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from ...core.converter import ONNXConverter

logger = logging.getLogger(__name__)


class HuggingFaceONNXConverter(ONNXConverter):
    """Converter for HuggingFace models to ONNX format."""

    def __init__(
            self,
            model: AutoModelForSequenceClassification,
            tokenizer: AutoTokenizer,
            config: Dict[str, Any]
    ):
        """Initialize with HuggingFace model and tokenizer."""
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.hf_model_name = config.get("model_name", "model")

    def convert(
        self,
        output_path: Union[str, Path],
        output_format: str = "onnx",
        opset_version: int = 14,
        optimize_for_gpu: bool = False,
        **kwargs
    ) -> Path:
        """Convert HuggingFace model to ONNX format."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Converting HuggingFace model to ONNX format at {output_path}"
        )

        # Convert to ONNX using Optimum
        provider = (
            "CUDAExecutionProvider" if optimize_for_gpu
            else "CPUExecutionProvider"
        )

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.hf_model_name,
            export=True,
            provider=provider
        )

        # Save the model
        ort_model.save_pretrained(str(output_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))

        # Log model size
        model_size = self.get_model_size(output_path)
        logger.info(f"ONNX model saved. Size: {model_size:.1f} MB")

        # Validate conversion
        if self.validate_conversion(self.model, output_path):
            logger.info("Model conversion validated successfully")
        else:
            raise RuntimeError("Model conversion validation failed")

        return output_path

    def quantize(
        self,
        method: str = "dynamic",
        output_path: Optional[Union[str, Path]] = None,
        per_channel: bool = True,
        **kwargs
    ) -> Path:
        """Quantize the ONNX model for better performance."""
        if output_path is None:
            raise ValueError("Output path must be specified for quantization")

        output_path = Path(output_path)

        logger.info(f"Quantizing model using {method} quantization")

        # Load the ONNX model
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(output_path),
            provider="CPUExecutionProvider"
        )

        # Configure quantization
        if method == "dynamic":
            qconfig = AutoQuantizationConfig.arm64(
                is_static=False,
                per_channel=per_channel
            )
        elif method == "static":
            # Static quantization would need calibration data
            raise NotImplementedError(
                "Static quantization requires calibration data"
            )
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # Create quantizer
        quantizer = ORTQuantizer.from_pretrained(ort_model)

        # Apply quantization
        quantizer.quantize(
            save_dir=str(output_path),
            quantization_config=qconfig,
        )

        # Log new model size
        onnx_path = output_path.parent / "model.onnx"
        original_size = (
            self.get_model_size(onnx_path) if onnx_path.exists() else 0
        )
        quantized_size = self.get_model_size(
            output_path / "model_quantized.onnx"
        )

        if original_size > 0:
            compression_ratio = original_size / quantized_size
            logger.info(
                f"Quantized model size: {quantized_size:.1f} MB "
                f"(compression ratio: {compression_ratio:.1f}x)"
            )
        else:
            logger.info(f"Quantized model size: {quantized_size:.1f} MB")

        return output_path

    def optimize(
        self,
        output_path: Union[str, Path],
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Apply additional optimizations to the ONNX model."""
        output_path = Path(output_path)

        if optimization_config is None:
            optimization_config = {
                "enable_all": True,
                "optimization_level": 99
            }

        logger.info(
            f"Optimizing ONNX model with config: {optimization_config}"
        )

        # Load and optimize using ONNX Runtime optimizations
        import onnxruntime as ort

        model_path = output_path / "model.onnx"
        if not model_path.exists():
            model_path = output_path / "model_quantized.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"No ONNX model found in {output_path}")

        # Create optimized session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.optimized_model_filepath = str(
            output_path / "model_optimized.onnx"
        )

        # Create session to trigger optimization
        _ = ort.InferenceSession(str(model_path), sess_options)

        logger.info("Model optimization completed")
        return output_path

    def validate_conversion(
            self, original_model: Any, converted_model_path: Path
    ) -> bool:
        """Validate the converted model produces similar outputs."""
        # Basic file existence check
        model_files = list(converted_model_path.glob("*.onnx"))
        if not model_files:
            logger.error("No ONNX model files found")
            return False

        try:
            # Load the converted model
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                str(converted_model_path),
                provider="CPUExecutionProvider"
            )

            # Test with sample input
            test_text = "This is a test input for validation."
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                max_length=self.config.get("max_length", 128),
                truncation=True,
                padding="max_length"
            )

            # Get outputs from both models
            with torch.no_grad():
                original_outputs = original_model(**inputs).logits
                onnx_outputs = ort_model(**inputs).logits

            # Compare outputs (allowing small numerical differences)
            import torch
            max_diff = torch.max(
                torch.abs(original_outputs - onnx_outputs)
            ).item()

            if max_diff < 1e-3:
                logger.info(f"Validation passed. Max difference: {max_diff}")
                return True
            else:
                logger.warning(
                    f"Validation failed. Max difference: {max_diff}"
                )
                return False

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
