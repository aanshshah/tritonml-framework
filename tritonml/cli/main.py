"""Main CLI entry point for TritonML."""

import click
import logging
# from pathlib import Path  # Unused import
from typing import Optional

from ..utils import quick_deploy
# from ..tasks import list_tasks  # Imported later when needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """TritonML - Deploy ML models to Triton Inference Server."""
    pass


@cli.command(name="deploy")
@click.argument("model_name")
@click.option("--task", "-t", help="Model task type", default=None)
@click.option("--server", "-s", default="localhost:8000",
              help="Triton server URL")
@click.option("--name", "-n", help="Deployment name", default=None)
@click.option("--quantize/--no-quantize", default=True, help="Quantize model")
@click.option("--optimize/--no-optimize", default=True, help="Optimize model")
def deploy_command(model_name: str, task: Optional[str], server: str,
                   name: Optional[str], quantize: bool, optimize: bool):
    """Deploy a model to Triton server."""
    try:
        if task is None:
            # Try to auto-detect
            from ..core.model import TritonModel
            task = TritonModel._detect_task(model_name)
            click.echo(f"Auto-detected task: {task}")

        deployment_args = {
            "model_name": model_name,
            "task": task,
            "server_url": server,
            "quantize": quantize,
            "optimize": optimize
        }
        if name:
            deployment_args["model_name"] = name
        deployment = quick_deploy(**deployment_args)

        click.echo("\n✅ Model deployed successfully!")
        click.echo(f"Model: {deployment['model_name']}")
        click.echo(f"Server: {deployment['server_url']}")
        click.echo(f"Path: {deployment['model_path']}")

    except Exception as e:
        click.echo(f"❌ Deployment failed: {e}", err=True)
        raise click.Exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output path for converted model")
@click.option("--format", "-f", default="onnx",
              help="Output format (onnx, torchscript, tensorrt)")
@click.option("--quantize/--no-quantize", default=False, help="Quantize model")
def convert(model_path: str, output: Optional[str], format: str,
            quantize: bool):
    """Convert a model to deployment format."""
    click.echo(f"Converting model from {model_path} to {format} format...")

    # Implementation would go here
    click.echo("Model conversion not yet implemented in CLI")


@cli.command()
@click.argument("model_name")
@click.argument("text")
@click.option("--server", "-s", default="localhost:8000",
              help="Triton server URL")
def predict(model_name: str, text: str, server: str):
    """Run inference on deployed model."""
    try:
        from ..core.client import TritonClient

        client = TritonClient(server_url=server, model_name=model_name)

        # This assumes text classification for now
        from ..tasks.text_classification import TextClassificationModel

        # Create a dummy model just for preprocessing
        model = TextClassificationModel.from_pretrained(
            "bert-base-uncased", model_name=model_name
        )

        # Preprocess
        inputs = model.preprocess(text)

        # Infer
        outputs = client.infer(inputs)

        # Postprocess
        result = model.postprocess(outputs)

        click.echo(f"\nPrediction: {result}")

    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}", err=True)
        raise click.Exit(1)


@cli.command()
def list_tasks():
    """List available model tasks."""
    from ..tasks import list_tasks as get_tasks

    tasks = get_tasks()
    click.echo("\nAvailable tasks:")
    for task in tasks:
        click.echo(f"  - {task}")


@cli.command()
@click.argument("model_name")
@click.option("--batch-sizes", "-b", default="1,8,16,32",
              help="Batch sizes to test")
@click.option("--server", "-s", default="localhost:8000",
              help="Triton server URL")
@click.option("--dataset", "-d",
              help="Hugging Face dataset name for benchmarking")
@click.option("--num-samples", "-n", default=1000,
              help="Number of samples to use")
@click.option("--output", "-o", help="Output file for results (JSON or CSV)")
def benchmark(model_name: str, batch_sizes: str, server: str, dataset: str,
              num_samples: int, output: str):
    """Benchmark a deployed model."""
    try:
        from ..core.client import TritonClient

        # Parse batch sizes
        sizes = [int(s.strip()) for s in batch_sizes.split(",")]

        if dataset:
            # Use Hugging Face dataset benchmarking
            click.echo(
                f"\nBenchmarking model {model_name} with dataset {dataset}..."
            )

            from ..benchmarks import HuggingFaceDatasetLoader, BenchmarkRunner
            from ..tasks import TextClassificationModel

            # Create model instance connected to deployed model
            model = TextClassificationModel(model_name=model_name)
            model._client = TritonClient(
                server_url=server, model_name=model_name
            )

            # Create dataset loader
            loader = HuggingFaceDatasetLoader(dataset, split="test")

            # Run benchmark
            runner = BenchmarkRunner(model)
            runner.benchmark_dataset(
                loader,
                batch_sizes=sizes,
                num_samples=num_samples
            )

            # Print summary
            runner.print_summary()

            # Save results if requested
            if output:
                format = "csv" if output.endswith(".csv") else "json"
                runner.save_results(output, format=format)
                click.echo(f"\nResults saved to {output}")

        else:
            # Original simple benchmark
            client = TritonClient(server_url=server, model_name=model_name)

            click.echo(f"\nBenchmarking model {model_name}...")
            click.echo(f"Batch sizes: {sizes}")

            # Run simple latency test
            import time
            import numpy as np

            for size in sizes:
                # Create dummy inputs
                # (would need proper inputs in real implementation)
                inputs = {
                    "input_ids": np.ones((size, 128), dtype=np.int64),
                    "attention_mask": np.ones((size, 128), dtype=np.int64)
                }

                # Warmup
                for _ in range(5):
                    client.infer(inputs)

                # Benchmark
                start = time.time()
                iterations = 20
                for _ in range(iterations):
                    client.infer(inputs)
                elapsed = time.time() - start

                avg_latency = (elapsed / iterations) * 1000  # ms
                throughput = (size * iterations) / elapsed

                click.echo(f"\nBatch size {size}:")
                click.echo(f"  Average latency: {avg_latency:.2f} ms")
                click.echo(f"  Throughput: {throughput:.2f} samples/sec")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        raise click.Exit(1)


@cli.group()
def model():
    """Model management commands."""
    pass


@model.command("list")
@click.option("--server", "-s", default="localhost:8000",
              help="Triton server URL")
def list_models(server: str):
    """List models on Triton server."""
    try:
        import tritonclient.http as httpclient

        client = httpclient.InferenceServerClient(url=server)

        # Get model repository index
        models = client.get_model_repository_index()

        click.echo(f"\nModels on {server}:")
        for model in models:
            name = model.get("name", "unknown")
            version = model.get("version", "unknown")
            state = model.get("state", "unknown")
            click.echo(f"  - {name} (v{version}) [{state}]")

    except Exception as e:
        click.echo(f"❌ Failed to list models: {e}", err=True)
        raise click.Exit(1)


if __name__ == "__main__":
    cli()
