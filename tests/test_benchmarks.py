"""Tests for TritonML benchmarking functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from tritonml.benchmarks import HuggingFaceDatasetLoader, BenchmarkRunner
from tritonml.core.model import TritonModel
from tritonml.core.config import ModelConfig


class MockModel(TritonModel):
    """Mock model for testing."""
    
    def __init__(self):
        config = ModelConfig(model_name="test-model")
        super().__init__(config)
        self._client = Mock()
        
    def preprocess(self, inputs):
        return inputs
        
    def postprocess(self, outputs):
        return outputs
        
    def predict(self, inputs):
        # Simulate prediction with small delay
        import time
        time.sleep(0.001)  # 1ms delay
        return ["positive"] * len(inputs)


class TestHuggingFaceDatasetLoader:
    """Test the HuggingFace dataset loader."""
    
    @patch('tritonml.benchmarks.dataset_loader.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """Test loading a dataset."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.column_names = ["text", "label"]
        mock_load_dataset.return_value = mock_dataset
        
        # Load dataset
        loader = HuggingFaceDatasetLoader("imdb", split="test")
        dataset = loader.load(max_samples=50)
        
        # Verify
        mock_load_dataset.assert_called_once_with("imdb", None, split="test")
        assert len(dataset) == 50  # Should be limited
        
    @patch('tritonml.benchmarks.dataset_loader.load_dataset')
    def test_get_samples_auto_detect(self, mock_load_dataset):
        """Test auto-detection of text column."""
        # Mock dataset with samples
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.__len__.return_value = 3
        mock_dataset.__iter__.return_value = iter([
            {"text": "Sample 1", "label": 0},
            {"text": "Sample 2", "label": 1},
            {"text": "Sample 3", "label": 0}
        ])
        mock_load_dataset.return_value = mock_dataset
        
        # Load and get samples
        loader = HuggingFaceDatasetLoader("test_dataset")
        loader.load()
        samples = loader.get_samples()
        
        # Verify
        assert samples == ["Sample 1", "Sample 2", "Sample 3"]
        
    @patch('tritonml.benchmarks.dataset_loader.load_dataset')
    def test_get_samples_with_batching(self, mock_load_dataset):
        """Test getting samples in batches."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = 10
        mock_dataset.__iter__.return_value = iter([
            {"text": f"Sample {i}"} for i in range(10)
        ])
        mock_load_dataset.return_value = mock_dataset
        
        # Load and get batched samples
        loader = HuggingFaceDatasetLoader("test_dataset")
        loader.load()
        batches = loader.get_samples(batch_size=3)
        
        # Verify
        assert len(batches) == 4  # 3 full batches + 1 partial
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # Last batch has 1 sample
        
    def test_list_popular_datasets(self):
        """Test listing popular datasets."""
        datasets = HuggingFaceDatasetLoader.list_popular_datasets()
        
        # Verify structure
        assert "text_classification" in datasets
        assert "imdb" in datasets["text_classification"]
        assert isinstance(datasets["text_classification"]["imdb"], str)


class TestBenchmarkRunner:
    """Test the benchmark runner."""
    
    @patch('tritonml.benchmarks.dataset_loader.load_dataset')
    def test_benchmark_dataset(self, mock_load_dataset):
        """Test running benchmark on a dataset."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = 100
        mock_dataset.__iter__.return_value = iter([
            {"text": f"Sample text {i}"} for i in range(100)
        ])
        mock_load_dataset.return_value = mock_dataset
        
        # Create mock model and runner
        model = MockModel()
        runner = BenchmarkRunner(model, warmup_runs=1)
        
        # Create dataset loader
        loader = HuggingFaceDatasetLoader("test_dataset")
        
        # Run benchmark
        results = runner.benchmark_dataset(
            loader,
            batch_sizes=[1, 2],
            num_samples=10,
            runs_per_batch=2
        )
        
        # Verify results structure
        assert results["dataset"] == "test_dataset"
        assert results["num_samples"] == 10
        assert "batch_1" in results["batch_results"]
        assert "batch_2" in results["batch_results"]
        
        # Check batch results
        batch_1_results = results["batch_results"]["batch_1"]
        assert batch_1_results["batch_size"] == 1
        assert "latency_ms" in batch_1_results
        assert "mean" in batch_1_results["latency_ms"]
        assert "throughput" in batch_1_results
        
    def test_save_results_json(self, tmp_path):
        """Test saving results to JSON."""
        # Create runner with mock model
        model = MockModel()
        runner = BenchmarkRunner(model)
        
        # Add mock results
        runner.results = {
            "test_dataset": {
                "dataset": "test_dataset",
                "num_samples": 100,
                "batch_results": {
                    "batch_1": {
                        "batch_size": 1,
                        "latency_ms": {"mean": 10.5, "p95": 12.0},
                        "throughput": {"samples_per_second": 95.2}
                    }
                }
            }
        }
        
        # Save results
        output_file = tmp_path / "results.json"
        runner.save_results(output_file, format="json")
        
        # Verify file exists and content
        assert output_file.exists()
        import json
        with open(output_file) as f:
            loaded = json.load(f)
        assert "test_dataset" in loaded
        
    def test_save_results_csv(self, tmp_path):
        """Test saving results to CSV."""
        # Create runner with mock model
        model = MockModel()
        runner = BenchmarkRunner(model)
        
        # Add mock results
        runner.results = {
            "test_dataset": {
                "dataset": "test_dataset",
                "model_name": "test-model",
                "batch_results": {
                    "batch_1": {
                        "batch_size": 1,
                        "latency_ms": {"mean": 10.5, "p95": 12.0},
                        "throughput": {"samples_per_second": 95.2}
                    }
                }
            }
        }
        
        # Save results
        output_file = tmp_path / "results.csv"
        runner.save_results(output_file, format="csv")
        
        # Verify file exists and content
        assert output_file.exists()
        import csv
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["dataset"] == "test_dataset"