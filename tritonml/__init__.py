"""TritonML - A framework for deploying ML models to Triton Server."""

__version__ = "0.1.0"

from .benchmarks import BenchmarkRunner, HuggingFaceDatasetLoader
from .core.client import TritonClient
from .core.converter import ModelConverter
from .core.model import TritonModel
from .tasks import ImageClassificationModel, TextClassificationModel
from .utils.deploy import deploy

__all__ = [
    "TritonModel",
    "TritonClient",
    "ModelConverter",
    "TextClassificationModel",
    "ImageClassificationModel",
    "deploy",
    "HuggingFaceDatasetLoader",
    "BenchmarkRunner",
]
