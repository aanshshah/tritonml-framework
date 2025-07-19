"""Base model class for TritonML framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import logging

from .config import TritonConfig
from .client import TritonClient
from .converter import ModelConverter

logger = logging.getLogger(__name__)


class TritonModel(ABC):
    """Base class for all Triton-deployable models."""

    def __init__(self, config: TritonConfig):
        """Initialize the model with configuration."""
        self.config = config
        self._client: Optional[TritonClient] = None
        self._converter: Optional[ModelConverter] = None

    @classmethod
    @abstractmethod
    def from_pretrained(
            cls, model_name_or_path: str, **kwargs
    ) -> "TritonModel":
        """Load a model from a pretrained source."""
        pass

    @classmethod
    def from_huggingface(
            cls, model_id: str, task: Optional[str] = None, **kwargs
    ) -> "TritonModel":
        """Create a model from HuggingFace hub."""
        from ..tasks import get_task_model

        if task is None:
            # Auto-detect task from model
            task = cls._detect_task(model_id)

        model_class = get_task_model(task)
        return model_class.from_pretrained(model_id, **kwargs)

    @staticmethod
    def _detect_task(model_id: str) -> str:
        """Auto-detect the task type from model ID or config."""
        # This would be enhanced with actual model inspection
        # For now, use simple heuristics
        if "emotion" in model_id.lower() or "sentiment" in model_id.lower():
            return "text-classification"
        elif "bert" in model_id.lower() or "roberta" in model_id.lower():
            return "text-classification"
        elif "vit" in model_id.lower() or "resnet" in model_id.lower():
            return "image-classification"
        else:
            raise ValueError(
                f"Could not auto-detect task for model {model_id}"
            )

    @abstractmethod
    def preprocess(self, inputs: Any) -> Dict[str, Any]:
        """Preprocess inputs for model inference."""
        pass

    @abstractmethod
    def postprocess(self, outputs: Dict[str, Any]) -> Any:
        """Postprocess model outputs."""
        pass

    def predict(self, inputs: Any) -> Any:
        """Run inference on inputs."""
        if self._client is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        # Preprocess
        processed_inputs = self.preprocess(inputs)

        # Run inference
        outputs = self._client.infer(
            model_name=self.config.model_name,
            inputs=processed_inputs
        )

        # Postprocess
        return self.postprocess(outputs)

    def convert(self, output_format: str = "onnx", **kwargs) -> Path:
        """Convert the model to deployment format."""
        if self._converter is None:
            self._converter = self._get_converter()

        output_path = self.config.model_path
        self._converter.convert(
            output_path=output_path,
            output_format=output_format,
            **kwargs
        )

        # Save Triton configuration
        self.config.save_config()

        return output_path

    @abstractmethod
    def _get_converter(self) -> ModelConverter:
        """Get the appropriate converter for this model type."""
        pass

    def quantize(self, method: str = "dynamic", **kwargs) -> "TritonModel":
        """Quantize the model for better performance."""
        if self._converter is None:
            self._converter = self._get_converter()

        self._converter.quantize(
            method=method,
            output_path=self.config.model_path,
            **kwargs
        )

        return self

    def optimize(
            self, optimization_config: Optional[Dict[str, Any]] = None
    ) -> "TritonModel":
        """Apply optimizations to the model."""
        if optimization_config is None:
            optimization_config = {
                "enable_quantization": True,
                "enable_graph_optimization": True,
                "optimization_level": 99
            }

        if self._converter is None:
            self._converter = self._get_converter()

        self._converter.optimize(
            output_path=self.config.model_path,
            optimization_config=optimization_config
        )

        return self

    def deploy(self, server_url: Optional[str] = None) -> TritonClient:
        """Deploy the model to Triton server."""
        if server_url:
            self.config.triton_server_url = server_url

        # Ensure model is converted
        if not self.config.model_path.exists():
            logger.info("Model not found, converting...")
            self.convert()

        # Create client
        self._client = TritonClient(
            server_url=self.config.triton_server_url,
            model_name=self.config.model_name
        )

        # Verify deployment
        if not self._client.is_model_ready():
            raise RuntimeError(
                f"Model {self.config.model_name} is not ready on server"
            )

        logger.info(f"Model {self.config.model_name} deployed successfully")
        return self._client

    def benchmark(
            self, test_inputs: List[Any],
            batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Benchmark the model with different batch sizes."""
        if self._client is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        results = {}
        for batch_size in batch_sizes:
            # Create batch
            batch = test_inputs[:batch_size] * (
                batch_size // len(test_inputs) + 1
            )
            batch = batch[:batch_size]

            # Run benchmark
            import time
            start = time.time()
            for _ in range(10):  # Run 10 iterations
                self.predict(batch)
            elapsed = time.time() - start

            results[f"batch_{batch_size}"] = {
                "avg_latency_ms": (elapsed / 10) * 1000,
                "throughput": (batch_size * 10) / elapsed
            }

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save the model configuration and artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        import json
        config_dict = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "config": self.config.__dict__
        }

        with open(path / "model_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TritonModel":
        """Load a model from saved artifacts."""
        path = Path(path)

        # Load config
        import json
        with open(path / "model_config.json", "r") as f:
            json.load(f)  # config_dict would be used here

        # Reconstruct model
        # This would need task-specific loading logic
        raise NotImplementedError(
            "Model loading to be implemented by subclasses"
        )
