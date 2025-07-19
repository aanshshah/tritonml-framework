"""Tests for text classification models."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from tritonml.tasks.text_classification import (
    TextClassificationModel,
    EmotionClassifier,
    SentimentClassifier
)
from tritonml.core.config import TextClassificationConfig


class TestTextClassificationModel:
    """Test text classification functionality."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_from_pretrained(self, mock_model, mock_tokenizer):
        """Test loading a pretrained model."""
        # Mock model config
        mock_model_instance = Mock()
        mock_model_instance.config.num_labels = 4
        mock_model_instance.config.id2label = {
            0: "anger", 1: "joy", 2: "optimism", 3: "sadness"
        }
        mock_model.return_value = mock_model_instance

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Load model
        model = TextClassificationModel.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )

        assert isinstance(model, TextClassificationModel)
        assert model.config.labels == ["anger", "joy", "optimism", "sadness"]
        assert model.config.model_name == "twitter-roberta-base-emotion"

    def test_preprocess_single_text(self):
        """Test preprocessing single text input."""
        config = TextClassificationConfig(
            model_name="test",
            labels=["pos", "neg"]
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 2023, 102]]),
            "attention_mask": np.array([[1, 1, 1]])
        }

        model = TextClassificationModel(config, tokenizer=mock_tokenizer)

        result = model.preprocess("This is a test")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dtype == np.int64

    def test_preprocess_batch(self):
        """Test preprocessing batch of texts."""
        config = TextClassificationConfig(
            model_name="test",
            labels=["pos", "neg"]
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 2023, 102], [101, 2000, 102]]),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]])
        }

        model = TextClassificationModel(config, tokenizer=mock_tokenizer)

        result = model.preprocess(["Text 1", "Text 2"])

        assert result["input_ids"].shape[0] == 2
        assert result["attention_mask"].shape[0] == 2

    def test_postprocess_single(self):
        """Test postprocessing single prediction."""
        config = TextClassificationConfig(
            model_name="test",
            labels=["negative", "positive"]
        )

        model = TextClassificationModel(config)

        outputs = {"logits": np.array([[-1.0, 2.0]])}
        result = model.postprocess(outputs)

        assert result == "positive"

    def test_postprocess_batch(self):
        """Test postprocessing batch predictions."""
        config = TextClassificationConfig(
            model_name="test",
            labels=["negative", "positive"]
        )

        model = TextClassificationModel(config)

        outputs = {"logits": np.array([[-1.0, 2.0], [3.0, -1.0]])}
        results = model.postprocess(outputs)

        assert results == ["positive", "negative"]

    def test_predict_proba(self):
        """Test getting prediction probabilities."""
        config = TextClassificationConfig(
            model_name="test",
            labels=["neg", "pos"]
        )

        model = TextClassificationModel(config)

        # Mock client
        mock_client = Mock()
        mock_client.infer.return_value = {"logits": np.array([[1.0, 2.0]])}
        model._client = mock_client

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 102]]),
            "attention_mask": np.array([[1, 1]])
        }
        model._tokenizer = mock_tokenizer

        probs = model.predict_proba("Test text")

        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestEmotionClassifier:
    """Test emotion classifier specialization."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_emotion_defaults(self, mock_model, mock_tokenizer):
        """Test emotion classifier defaults."""
        mock_model_instance = Mock()
        mock_model_instance.config.num_labels = 4
        mock_model.return_value = mock_model_instance

        mock_tokenizer.return_value = Mock()

        model = EmotionClassifier.from_pretrained()

        assert model.config.labels == ["anger", "joy", "optimism", "sadness"]
        assert model.config.model_name == "emotion-classifier"

        # Check it uses the correct HF model
        expected_model = "cardiffnlp/twitter-roberta-base-emotion"
        mock_model.assert_called_with(expected_model)


class TestSentimentClassifier:
    """Test sentiment classifier specialization."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_sentiment_defaults(self, mock_model, mock_tokenizer):
        """Test sentiment classifier defaults."""
        mock_model_instance = Mock()
        mock_model_instance.config.num_labels = 2
        mock_model.return_value = mock_model_instance

        mock_tokenizer.return_value = Mock()

        model = SentimentClassifier.from_pretrained()

        assert model.config.labels == ["negative", "positive"]
        assert model.config.model_name == "sentiment-classifier"


if __name__ == "__main__":
    pytest.main([__file__])
