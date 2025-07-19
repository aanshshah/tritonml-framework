"""Tests for TritonML core components."""

import pytest
# from pathlib import Path  # Unused import
import numpy as np
from unittest.mock import Mock, patch

from tritonml.core.config import TritonConfig, TextClassificationConfig
from tritonml.core.client import TritonClient
from tritonml.core.model import TritonModel


class TestTritonConfig:
    """Test configuration management."""

    def test_base_config_creation(self):
        """Test creating a base configuration."""
        config = TritonConfig(
            model_name="test-model",
            input_shapes={"input": [10]},
            output_shapes={"output": [5]}
        )

        assert config.model_name == "test-model"
        assert config.model_version == "1"
        assert config.max_batch_size == 32

    def test_config_to_pbtxt(self):
        """Test converting config to Triton pbtxt format."""
        config = TritonConfig(
            model_name="test-model",
            input_shapes={"input_ids": [128]},
            output_shapes={"logits": [4]}
        )

        pbtxt = config.to_pbtxt()

        assert 'name: "test-model"' in pbtxt
        assert "max_batch_size: 32" in pbtxt
        assert 'name: "input_ids"' in pbtxt
        assert "dims: [128]" in pbtxt

    def test_text_classification_config(self):
        """Test text classification specific config."""
        config = TextClassificationConfig(
            model_name="emotion-classifier",
            labels=["anger", "joy", "optimism", "sadness"],
            max_sequence_length=128
        )

        assert len(config.labels) == 4
        assert config.input_shapes["input_ids"] == [128]
        assert config.output_shapes["logits"] == [4]


class TestTritonClient:
    """Test Triton client functionality."""

    @patch("tritonclient.http.InferenceServerClient")
    def test_client_initialization(self, mock_client_class):
        """Test client initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        client = TritonClient(
            server_url="localhost:8000",
            model_name="test-model"
        )

        assert client.server_url == "localhost:8000"
        assert client.model_name == "test-model"
        mock_client_class.assert_called_with(url="localhost:8000")

    @patch("tritonclient.http.InferenceServerClient")
    def test_is_model_ready(self, mock_client_class):
        """Test checking if model is ready."""
        mock_client = Mock()
        mock_client.is_model_ready.return_value = True
        mock_client_class.return_value = mock_client

        client = TritonClient("localhost:8000", "test-model")
        assert client.is_model_ready() is True

        mock_client.is_model_ready.assert_called_with(
            model_name="test-model",
            model_version="1"
        )

    @patch("tritonclient.http.InferenceServerClient")
    def test_prepare_inputs(self, mock_client_class):
        """Test input preparation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        client = TritonClient("localhost:8000", "test-model")

        inputs = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64)
        }

        with patch("tritonclient.http.InferInput") as mock_input:
            triton_inputs = client.prepare_inputs(inputs)

            assert len(triton_inputs) == 2
            assert mock_input.call_count == 2

    def test_infer_triton_dtype(self):
        """Test numpy to Triton dtype conversion."""
        assert TritonClient._infer_triton_dtype(np.float32) == "FP32"
        assert TritonClient._infer_triton_dtype(np.int64) == "INT64"
        assert TritonClient._infer_triton_dtype(np.int8) == "INT8"


class TestTritonModel:
    """Test base TritonModel functionality."""

    def test_detect_task(self):
        """Test task auto-detection."""
        emotion_model = "cardiffnlp/twitter-roberta-base-emotion"
        assert TritonModel._detect_task(emotion_model) == "text-classification"
        bert_model = "bert-base-uncased"
        assert TritonModel._detect_task(bert_model) == "text-classification"
        vit_model = "google/vit-base-patch16-224"
        assert TritonModel._detect_task(vit_model) == "image-classification"

        with pytest.raises(ValueError):
            TritonModel._detect_task("unknown-model-xyz")

    @patch("tritonml.tasks.get_task_model")
    def test_from_huggingface(self, mock_get_task):
        """Test loading from HuggingFace."""
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_get_task.return_value = mock_model_class

        TritonModel.from_huggingface(
            "cardiffnlp/twitter-roberta-base-emotion",
            task="text-classification"
        )

        mock_get_task.assert_called_with("text-classification")
        mock_model_class.from_pretrained.assert_called_once()


class MockTritonModel(TritonModel):
    """Mock implementation for testing."""

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        config = TritonConfig(model_name="mock-model")
        return cls(config)

    def preprocess(self, inputs):
        return {"input": np.array([inputs])}

    def postprocess(self, outputs):
        return outputs.get("output", "mock-output")

    def _get_converter(self):
        return Mock()


class TestModelDeployment:
    """Test model deployment functionality."""

    @patch("tritonml.core.client.TritonClient")
    def test_deploy_model(self, mock_client_class):
        """Test deploying a model."""
        mock_client = Mock()
        mock_client.is_model_ready.return_value = True
        mock_client_class.return_value = mock_client

        model = MockTritonModel(TritonConfig(model_name="test"))

        with patch.object(model, "convert") as mock_convert:
            client = model.deploy(server_url="localhost:8000")

            assert client == mock_client
            mock_convert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
