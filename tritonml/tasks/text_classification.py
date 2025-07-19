"""Text classification model implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# import torch  # Used in converter module
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core.config import TextClassificationConfig
from ..core.converter import ModelConverter
from ..core.model import TritonModel
from .converters.huggingface_onnx import HuggingFaceONNXConverter


class TextClassificationModel(TritonModel):
    """Text classification model for Triton deployment."""

    def __init__(
        self,
        config: TextClassificationConfig,
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForSequenceClassification] = None,
    ):
        """Initialize text classification model."""
        super().__init__(config)
        self.config: TextClassificationConfig = config
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        max_sequence_length: int = 128,
        **kwargs: Any,
    ) -> "TextClassificationModel":
        """Load a text classification model from HuggingFace."""
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        # Infer labels if not provided
        if labels is None:
            if hasattr(model.config, "id2label"):
                labels = [
                    model.config.id2label[i] for i in range(model.config.num_labels)
                ]
            else:
                labels = [f"LABEL_{i}" for i in range(model.config.num_labels)]

        # Create configuration
        if model_name is None:
            model_name = model_name_or_path.split("/")[-1]

        config = TextClassificationConfig(
            model_name=model_name,
            max_sequence_length=max_sequence_length,
            labels=labels,
            tokenizer_name=model_name_or_path,
            **kwargs,
        )

        return cls(config=config, tokenizer=tokenizer, model=model)

    def preprocess(self, inputs: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """Preprocess text inputs for inference."""
        if isinstance(inputs, str):
            inputs = [inputs]

        # Tokenize inputs
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        encoded = self._tokenizer(  # type: ignore[operator]
            inputs,
            max_length=self.config.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

    def postprocess(self, outputs: Dict[str, np.ndarray]) -> Union[str, List[str]]:
        """Postprocess model outputs to get predictions."""
        logits = outputs["logits"]

        # Get predictions
        predictions = np.argmax(logits, axis=-1)

        # Convert to labels
        if len(predictions.shape) == 0:
            # Single prediction (scalar)
            return self.config.labels[int(predictions.item())]
        elif predictions.shape[0] == 1:
            # Single prediction in batch dimension
            return self.config.labels[int(predictions[0])]
        else:
            # Batch predictions
            return [self.config.labels[int(idx)] for idx in predictions]

    def predict_proba(self, inputs: Union[str, List[str]]) -> np.ndarray:
        """Get prediction probabilities."""
        if self._client is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        # Preprocess
        processed_inputs = self.preprocess(inputs)

        # Run inference
        outputs = self._client.infer(
            model_name=self.config.model_name, inputs=processed_inputs
        )

        # Apply softmax to get probabilities
        logits = outputs["logits"]
        probs = self._softmax(logits)

        return probs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # type: ignore[no-any-return]

    def _get_converter(self) -> ModelConverter:
        """Get the ONNX converter for HuggingFace models."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        return HuggingFaceONNXConverter(
            model=self._model,
            tokenizer=self._tokenizer,
            config={
                "task": "text-classification",
                "max_length": self.config.max_sequence_length,
                "model_name": self.config.tokenizer_name,
            },
        )

    def explain(self, text: str, method: str = "attention") -> Dict[str, Any]:
        """Get explanation for a prediction."""
        if method == "attention":
            # Get attention weights
            self.preprocess(text)  # processed would be used here

            # This would need special handling in the converter
            # to expose attention weights
            raise NotImplementedError("Attention-based explanation not yet implemented")
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def save_tokenizer(self, path: Union[str, Path]) -> None:
        """Save the tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        self._tokenizer.save_pretrained(str(path))  # type: ignore[attr-defined]


class EmotionClassifier(TextClassificationModel):
    """Specialized emotion classification model."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = ("cardiffnlp/twitter-roberta-base-emotion"),
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        max_sequence_length: int = 128,
        **kwargs: Any,
    ) -> "EmotionClassifier":
        """Load the emotion classification model."""
        # Set emotion-specific defaults
        if labels is None:
            labels = ["anger", "joy", "optimism", "sadness"]
        if model_name is None:
            model_name = "emotion-classifier"

        result = super().from_pretrained(model_name_or_path, model_name=model_name, labels=labels, max_sequence_length=max_sequence_length, **kwargs)
        return result  # type: ignore[return-value]


class SentimentClassifier(TextClassificationModel):
    """Specialized sentiment classification model."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = ("distilbert-base-uncased-finetuned-sst-2-english"),
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        max_sequence_length: int = 128,
        **kwargs: Any,
    ) -> "SentimentClassifier":
        """Load the sentiment classification model."""
        # Set sentiment-specific defaults
        if labels is None:
            labels = ["negative", "positive"]
        if model_name is None:
            model_name = "sentiment-classifier"

        result = super().from_pretrained(model_name_or_path, model_name=model_name, labels=labels, max_sequence_length=max_sequence_length, **kwargs)
        return result  # type: ignore[return-value]
