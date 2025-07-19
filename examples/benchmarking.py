"""Example of benchmarking models with TritonML."""

from tritonml.tasks import TextClassificationModel
import time
import numpy as np


def generate_test_data(num_samples: int = 100):
    """Generate test data for benchmarking."""
    templates = [
        "This is a {} review about the product.",
        "I {} this service very much.",
        "The experience was {} overall.",
        "Would {} recommend to others.",
        "Quality is {} what I expected."
    ]
    
    sentiments = ["positive", "negative", "great", "terrible", "amazing", "awful"]
    
    texts = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        sentiment = sentiments[i % len(sentiments)]
        texts.append(template.format(sentiment))
    
    return texts


def benchmark_latency(model, texts, batch_sizes=[1, 8, 16, 32]):
    """Benchmark model latency with different batch sizes."""
    
    print("Benchmarking Model Performance")
    print("=" * 60)
    print(f"Total samples: {len(texts)}")
    print()
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Prepare batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if len(batch) == batch_size:  # Only full batches
                batches.append(batch)
        
        # Warmup
        print("  Warming up...", end="", flush=True)
        for _ in range(5):
            _ = model.predict(batches[0])
        print(" done")
        
        # Benchmark
        print("  Running benchmark...", end="", flush=True)
        latencies = []
        
        for batch in batches[:10]:  # Test 10 batches
            start = time.time()
            _ = model.predict(batch)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        
        print(" done")
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = (batch_size / avg_latency) * 1000  # samples/sec
        
        results[batch_size] = {
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "throughput": throughput
        }
        
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print()
    
    return results


def main():
    """Run benchmarking example."""
    
    print("TritonML Benchmarking Example")
    print("-" * 60)
    print()
    
    # Note: This example shows the benchmarking API
    # In practice, you would have a running Triton server
    
    try:
        # Load model
        print("Loading model...")
        model = TextClassificationModel.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            model_name="sentiment-benchmark"
        )
        
        # Deploy model (requires running Triton server)
        print("Deploying model...")
        model.deploy()
        
        # Generate test data
        print("Generating test data...")
        test_texts = generate_test_data(1000)
        
        # Run benchmarks
        results = benchmark_latency(model, test_texts)
        
        # Summary
        print("\nBenchmark Summary")
        print("=" * 60)
        print(f"{'Batch Size':<12} {'Avg Latency':<15} {'P95 Latency':<15} {'Throughput':<15}")
        print("-" * 60)
        
        for batch_size, metrics in results.items():
            print(f"{batch_size:<12} "
                  f"{metrics['avg_latency_ms']:<15.2f} "
                  f"{metrics['p95_latency_ms']:<15.2f} "
                  f"{metrics['throughput']:<15.2f}")
        
    except Exception as e:
        print(f"\nNote: This example requires a running Triton server.")
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("1. Start Triton server: docker run -p 8000:8000 nvcr.io/nvidia/tritonserver:24.08-py3")
        print("2. Run this script again")


if __name__ == "__main__":
    main()