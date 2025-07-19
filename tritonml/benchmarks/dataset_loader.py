"""Hugging Face dataset loader for benchmarking."""

import logging
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class HuggingFaceDatasetLoader:
    """Load and prepare Hugging Face datasets for benchmarking."""

    def __init__(
        self, dataset_name: str, split: str = "test", config_name: Optional[str] = None
    ):
        """Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split to use (default: "test")
            config_name: Optional configuration name for datasets with
                multiple configs
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self._dataset = None
        self._preprocessor = None

    def load(self, max_samples: Optional[int] = None) -> Dataset:
        """Load the dataset from Hugging Face Hub.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset '{self.dataset_name}' (split: {self.split})")

        # Load dataset
        self._dataset = load_dataset(
            self.dataset_name, self.config_name, split=self.split
        )

        # Limit samples if requested
        if max_samples is not None and len(self._dataset) > max_samples:
            self._dataset = self._dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Loaded {len(self._dataset)} samples")
        return self._dataset

    def set_preprocessor(self, preprocessor):
        """Set a custom preprocessor function.

        Args:
            preprocessor: Function to preprocess dataset samples
        """
        self._preprocessor = preprocessor

    def get_samples(
        self,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Get samples from the dataset.

        Args:
            text_column: Name of the text column (auto-detected if None)
            label_column: Name of the label column (optional)
            batch_size: If specified, return samples in batches

        Returns:
            List of samples or batches
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Auto-detect text column if not specified
        if text_column is None:
            text_columns = [
                "text",
                "sentence",
                "sentence1",
                "premise",
                "question",
                "context",
                "input",
                "content",
                "review",
                "comment",
            ]
            for col in text_columns:
                if col in self._dataset.column_names:
                    text_column = col
                    break

            if text_column is None:
                raise ValueError(
                    f"Could not auto-detect text column. "
                    f"Available columns: {self._dataset.column_names}"
                )

        # Extract samples
        samples = []
        for i, sample in enumerate(self._dataset):
            if self._preprocessor:
                processed = self._preprocessor(sample)
                samples.append(processed)
            else:
                # Default: return text only or dict with text and label
                if label_column:
                    samples.append(
                        {"text": sample[text_column], "label": sample.get(label_column)}
                    )
                else:
                    samples.append(sample[text_column])

        # Batch if requested
        if batch_size:
            batched_samples = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                batched_samples.append(batch)
            return batched_samples

        return samples

    @staticmethod
    def list_popular_datasets() -> Dict[str, Dict[str, str]]:
        """List popular datasets for different tasks."""
        return {
            "text_classification": {
                "imdb": "Movie review sentiment classification",
                "ag_news": "News article classification (4 categories)",
                "emotion": "Emotion classification in text",
                "tweet_eval": "Tweet sentiment and emotion analysis",
                "financial_phrasebank": "Financial sentiment analysis",
                "rotten_tomatoes": "Movie review sentiment",
            },
            "named_entity_recognition": {
                "conll2003": "Named entity recognition benchmark",
                "wnut_17": "Novel and emerging entity recognition",
            },
            "question_answering": {
                "squad": "Stanford Question Answering Dataset",
                "squad_v2": "SQuAD 2.0 with unanswerable questions",
                "natural_questions": "Real Google search queries",
            },
            "text_generation": {
                "xsum": "Abstractive summarization",
                "cnn_dailymail": "News summarization",
                "reddit_tifu": "Reddit post summarization",
            },
            "image_classification": {
                "cifar10": "10-class image classification",
                "cifar100": "100-class image classification",
                "food101": "Food image classification",
                "oxford_flowers102": "Flower species classification",
            },
        }
