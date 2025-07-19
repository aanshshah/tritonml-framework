"""Benchmark runner for TritonML models with Hugging Face datasets."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.model import TritonModel
from .dataset_loader import HuggingFaceDatasetLoader

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmarks on TritonML models using various datasets."""

    def __init__(self, model: TritonModel, warmup_runs: int = 5):
        """Initialize the benchmark runner.

        Args:
            model: TritonML model to benchmark
            warmup_runs: Number of warmup runs before benchmarking
        """
        self.model = model
        self.warmup_runs = warmup_runs
        self.results: Dict[str, Any] = {}

    def benchmark_dataset(
        self,
        dataset_loader: HuggingFaceDatasetLoader,
        batch_sizes: List[int] = [1, 8, 16, 32],
        num_samples: Optional[int] = 1000,
        text_column: Optional[str] = None,
        runs_per_batch: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark model on a Hugging Face dataset.

        Args:
            dataset_loader: HuggingFaceDatasetLoader instance
            batch_sizes: List of batch sizes to test
            num_samples: Number of samples to use for benchmarking
            text_column: Name of the text column in dataset
            runs_per_batch: Number of runs per batch size

        Returns:
            Benchmark results dictionary
        """
        # Load dataset
        dataset_loader.load(max_samples=num_samples)
        samples = dataset_loader.get_samples(text_column=text_column)

        logger.info(f"Starting benchmark on dataset: {dataset_loader.dataset_name}")
        logger.info(f"Total samples: {len(samples)}")

        results: Dict[str, Any] = {
            "dataset": dataset_loader.dataset_name,
            "split": dataset_loader.split,
            "num_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model.config.model_name,
            "batch_results": {},
        }

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")

            # Prepare batches
            batches = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                if len(batch) == batch_size:  # Only full batches
                    batches.append(batch)

            if not batches:
                logger.warning(f"Not enough samples for batch size {batch_size}")
                continue

            # Warmup
            logger.info(f"  Running {self.warmup_runs} warmup iterations...")
            for _ in range(self.warmup_runs):
                _ = self.model.predict(batches[0])

            # Benchmark
            logger.info(f"  Running {runs_per_batch} benchmark iterations...")
            latencies = []

            for i in range(min(runs_per_batch, len(batches))):
                batch = batches[i % len(batches)]

                start_time = time.perf_counter()
                _ = self.model.predict(batch)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            latencies_array = np.array(latencies)

            batch_results: Dict[str, Any] = {
                "batch_size": batch_size,
                "num_batches": len(batches),
                "runs": runs_per_batch,
                "latency_ms": {
                    "mean": float(np.mean(latencies_array)),
                    "std": float(np.std(latencies_array)),
                    "min": float(np.min(latencies_array)),
                    "max": float(np.max(latencies_array)),
                    "p50": float(np.percentile(latencies_array, 50)),
                    "p90": float(np.percentile(latencies_array, 90)),
                    "p95": float(np.percentile(latencies_array, 95)),
                    "p99": float(np.percentile(latencies_array, 99)),
                },
                "throughput": {
                    "samples_per_second": float(
                        batch_size / (np.mean(latencies_array) / 1000)
                    ),
                    "batches_per_second": float(1000 / np.mean(latencies_array)),
                },
            }

            results["batch_results"][f"batch_{batch_size}"] = batch_results

            # Log results - access nested dicts step by step for mypy
            latency_dict = batch_results["latency_ms"]
            throughput_dict = batch_results["throughput"]

            mean_latency = latency_dict["mean"]  # type: ignore[index]
            p95_latency = latency_dict["p95"]  # type: ignore[index]
            samples_per_sec = throughput_dict[
                "samples_per_second"
            ]  # type: ignore[index]

            logger.info(f"  Mean latency: {mean_latency:.2f} ms")
            logger.info(f"  P95 latency: {p95_latency:.2f} ms")
            logger.info(f"  Throughput: {samples_per_sec:.2f} samples/sec")

        self.results[dataset_loader.dataset_name] = results
        return results

    def benchmark_multiple_datasets(
        self, dataset_configs: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Benchmark model on multiple datasets.

        Args:
            dataset_configs: List of dataset configurations
            **kwargs: Additional arguments passed to benchmark_dataset

        Returns:
            Combined benchmark results
        """
        all_results = {}

        for config in dataset_configs:
            dataset_name = config["dataset_name"]
            logger.info(f"\nBenchmarking dataset: {dataset_name}")

            # Create dataset loader
            loader = HuggingFaceDatasetLoader(
                dataset_name=dataset_name,
                split=config.get("split", "test"),
                config_name=config.get("config_name"),
            )

            # Set preprocessor if provided
            if "preprocessor" in config:
                loader.set_preprocessor(config["preprocessor"])

            # Run benchmark
            dataset_kwargs = kwargs.copy()
            dataset_kwargs.update(config.get("benchmark_params", {}))

            try:
                results = self.benchmark_dataset(loader, **dataset_kwargs)
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Failed to benchmark {dataset_name}: {e}")
                all_results[dataset_name] = {"error": str(e)}

        return all_results

    def save_results(self, output_path: Union[str, Path], format: str = "json") -> None:
        """Save benchmark results to file.

        Args:
            output_path: Path to save results
            format: Output format (json or csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2)
        elif format == "csv":
            # Convert to CSV format
            import csv

            rows = []
            for dataset_name, dataset_results in self.results.items():
                if "error" in dataset_results:
                    continue

                batch_results_dict = dataset_results.get("batch_results", {})
                for batch_key, batch_results in batch_results_dict.items():
                    row = {
                        "dataset": dataset_name,
                        "model": dataset_results["model_name"],
                        "batch_size": batch_results["batch_size"],
                        "mean_latency_ms": batch_results["latency_ms"]["mean"],
                        "p95_latency_ms": batch_results["latency_ms"]["p95"],
                        "throughput_samples_sec": (
                            batch_results["throughput"]["samples_per_second"]
                        ),
                    }
                    rows.append(row)

            if rows:
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results saved to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for dataset_name, results in self.results.items():
            print(f"\nDataset: {dataset_name}")

            if "error" in results:
                print(f"  Error: {results['error']}")
                continue

            print(f"  Model: {results['model_name']}")
            print(f"  Samples: {results['num_samples']}")
            print(
                f"\n  {'Batch Size':<12} {'Mean Latency':<15} "
                f"{'P95 Latency':<15} {'Throughput':<20}"
            )
            print("  " + "-" * 62)

            batch_results_dict = results.get("batch_results", {})
            for batch_key, batch_results in sorted(batch_results_dict.items()):
                batch_size = batch_results["batch_size"]
                mean_latency = batch_results["latency_ms"]["mean"]
                p95_latency = batch_results["latency_ms"]["p95"]
                throughput = batch_results["throughput"]["samples_per_second"]

                print(
                    f"  {batch_size:<12} {mean_latency:<15.2f} "
                    f"{p95_latency:<15.2f} {throughput:<20.2f}"
                )

        print("\n" + "=" * 80)
