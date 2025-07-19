"""Example of benchmarking TritonML models with Hugging Face datasets."""

from tritonml.tasks import TextClassificationModel
from tritonml.benchmarks import HuggingFaceDatasetLoader, BenchmarkRunner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def benchmark_sentiment_models():
    """Benchmark sentiment analysis models on multiple datasets."""
    
    print("TritonML Hugging Face Dataset Benchmarking Example")
    print("=" * 60)
    print()
    
    try:
        # Load a pre-trained sentiment model
        print("Loading sentiment analysis model...")
        model = TextClassificationModel.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            model_name="sentiment-benchmark"
        )
        
        # Deploy model (requires running Triton server)
        print("Deploying model to Triton server...")
        model.deploy()
        
        # Create benchmark runner
        runner = BenchmarkRunner(model, warmup_runs=3)
        
        # Define datasets to benchmark
        dataset_configs = [
            {
                "dataset_name": "imdb",
                "split": "test",
                "benchmark_params": {
                    "num_samples": 500,
                    "batch_sizes": [1, 8, 16, 32],
                    "text_column": "text"
                }
            },
            {
                "dataset_name": "rotten_tomatoes", 
                "split": "test",
                "benchmark_params": {
                    "num_samples": 500,
                    "batch_sizes": [1, 8, 16, 32],
                    "text_column": "text"
                }
            },
            {
                "dataset_name": "emotion",
                "split": "test",
                "benchmark_params": {
                    "num_samples": 500,
                    "batch_sizes": [1, 8, 16, 32],
                    "text_column": "text"
                }
            }
        ]
        
        # Run benchmarks
        print("\nRunning benchmarks on multiple datasets...")
        results = runner.benchmark_multiple_datasets(dataset_configs)
        
        # Print summary
        runner.print_summary()
        
        # Save results
        runner.save_results("benchmark_results.json", format="json")
        runner.save_results("benchmark_results.csv", format="csv")
        
        print("\nBenchmark results saved to:")
        print("  - benchmark_results.json")
        print("  - benchmark_results.csv")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Triton server is running")
        print("2. Required datasets are accessible")
        print("3. Model deployment succeeded")


def benchmark_custom_dataset():
    """Example of benchmarking with a custom dataset and preprocessor."""
    
    print("\nCustom Dataset Benchmarking Example")
    print("=" * 60)
    print()
    
    try:
        # Load model
        model = TextClassificationModel.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            model_name="sentiment-custom"
        )
        model.deploy()
        
        # Create dataset loader with custom preprocessor
        loader = HuggingFaceDatasetLoader("ag_news", split="test")
        
        # Define custom preprocessor to extract and format text
        def preprocess_ag_news(sample):
            # AG News has 'text' column, but we might want to preprocess it
            text = sample["text"]
            # Truncate to first 512 characters for faster processing
            return text[:512]
        
        loader.set_preprocessor(preprocess_ag_news)
        
        # Create runner and benchmark
        runner = BenchmarkRunner(model)
        results = runner.benchmark_dataset(
            loader,
            batch_sizes=[1, 4, 8, 16],
            num_samples=200,
            text_column="text"
        )
        
        # Print results
        print(f"\nResults for {loader.dataset_name}:")
        print(f"Model: {results['model_name']}")
        print(f"Samples: {results['num_samples']}")
        
        for batch_key, batch_results in results["batch_results"].items():
            print(f"\n{batch_key}:")
            print(f"  Mean latency: {batch_results['latency_ms']['mean']:.2f} ms")
            print(f"  Throughput: {batch_results['throughput']['samples_per_second']:.2f} samples/sec")
            
    except Exception as e:
        print(f"\nError in custom benchmark: {e}")


def list_available_datasets():
    """List popular datasets for benchmarking."""
    print("\nPopular Datasets for Benchmarking")
    print("=" * 60)
    
    datasets = HuggingFaceDatasetLoader.list_popular_datasets()
    
    for task, task_datasets in datasets.items():
        print(f"\n{task.replace('_', ' ').title()}:")
        for dataset_name, description in task_datasets.items():
            print(f"  - {dataset_name}: {description}")


if __name__ == "__main__":
    # List available datasets
    list_available_datasets()
    
    # Run sentiment model benchmarks
    benchmark_sentiment_models()
    
    # Run custom dataset benchmark
    benchmark_custom_dataset()