"""Generic Triton client for the TritonML framework."""

import numpy as np
import tritonclient.http as httpclient
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TritonClient:
    """Enhanced Triton client with additional functionality."""

    def __init__(
            self, server_url: str, model_name: str, model_version: str = "1"
    ):
        """Initialize the Triton client."""
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version

        # Initialize HTTP client
        self._client = httpclient.InferenceServerClient(url=server_url)

        # Cache model metadata
        self._model_metadata = None
        self._model_config = None

    @property
    def client(self) -> httpclient.InferenceServerClient:
        """Get the underlying Triton client."""
        return self._client

    def is_server_ready(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            return self._client.is_server_ready()
        except Exception as e:
            logger.error(f"Server health check failed: {e}")
            return False

    def is_model_ready(self) -> bool:
        """Check if the model is ready for inference."""
        try:
            return self._client.is_model_ready(
                model_name=self.model_name,
                model_version=self.model_version
            )
        except Exception as e:
            logger.error(f"Model ready check failed: {e}")
            return False

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata from the server."""
        if self._model_metadata is None:
            self._model_metadata = self._client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version
            )
        return self._model_metadata

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from the server."""
        if self._model_config is None:
            self._model_config = self._client.get_model_config(
                model_name=self.model_name,
                model_version=self.model_version
            )
        return self._model_config

    def prepare_inputs(
            self, inputs: Dict[str, np.ndarray]
    ) -> List[httpclient.InferInput]:
        """Prepare inputs for Triton inference."""
        triton_inputs = []

        for name, data in inputs.items():
            # Get data type from model metadata if available
            dtype = self._infer_triton_dtype(data.dtype)

            # Create Triton input
            triton_input = httpclient.InferInput(name, data.shape, dtype)
            triton_input.set_data_from_numpy(data)
            triton_inputs.append(triton_input)

        return triton_inputs

    def prepare_outputs(
            self, output_names: Optional[List[str]] = None
    ) -> List[httpclient.InferRequestedOutput]:
        """Prepare output requests for Triton inference."""
        if output_names is None:
            # Get all outputs from model metadata
            metadata = self.get_model_metadata()
            output_names = [
                output["name"] for output in metadata.get("outputs", [])
            ]

        return [httpclient.InferRequestedOutput(name) for name in output_names]

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Run inference on the model."""
        # Prepare inputs
        triton_inputs = self.prepare_inputs(inputs)

        # Prepare outputs
        triton_outputs = self.prepare_outputs(outputs)

        # Run inference
        response = self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=triton_inputs,
            outputs=triton_outputs,
            request_id=request_id,
            **kwargs
        )

        # Extract outputs
        result = {}
        for output in triton_outputs:
            output_name = output.name()
            result[output_name] = response.as_numpy(output_name)

        return result

    def infer_batch(
        self,
        batch_inputs: List[Dict[str, np.ndarray]],
        outputs: Optional[List[str]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Run batch inference on multiple inputs."""
        # Stack inputs into batches
        batched_inputs = {}
        for key in batch_inputs[0].keys():
            batched_inputs[key] = np.stack([inp[key] for inp in batch_inputs])

        # Run inference
        batch_outputs = self.infer(batched_inputs, outputs)

        # Split outputs back into individual results
        results = []
        batch_size = len(batch_inputs)
        for i in range(batch_size):
            result = {}
            for key, value in batch_outputs.items():
                result[key] = value[i]
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics for the model."""
        try:
            stats = self._client.get_inference_statistics(
                model_name=self.model_name,
                model_version=self.model_version
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def load_model(self) -> None:
        """Explicitly load the model on the server."""
        self._client.load_model(self.model_name)

    def unload_model(self) -> None:
        """Unload the model from the server."""
        self._client.unload_model(self.model_name)

    @staticmethod
    def _infer_triton_dtype(numpy_dtype: np.dtype) -> str:
        """Infer Triton data type from numpy dtype."""
        dtype_map = {
            np.float16: "FP16",
            np.float32: "FP32",
            np.float64: "FP64",
            np.int8: "INT8",
            np.int16: "INT16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.uint16: "UINT16",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.bool_: "BOOL",
            np.object_: "BYTES"
        }

        for np_type, triton_type in dtype_map.items():
            if numpy_dtype == np_type:
                return triton_type

        # Default to FP32
        return "FP32"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Could add cleanup logic here if needed
        pass
