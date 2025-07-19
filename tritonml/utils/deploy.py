"""Deployment utilities for TritonML models."""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import logging

from ..core.model import TritonModel
from ..core.client import TritonClient

logger = logging.getLogger(__name__)


def deploy(
    model: Union[TritonModel, str],
    server_url: str = "localhost:8000",
    quantize: bool = True,
    optimize: bool = True,
    **kwargs
) -> TritonClient:
    """Deploy a model to Triton server with optional optimizations."""
    
    # If string, load the model
    if isinstance(model, str):
        # Auto-detect and load model
        from ..core.model import TritonModel as BaseModel
        model = BaseModel.from_huggingface(model, **kwargs)
    
    logger.info(f"Deploying model {model.config.model_name} to {server_url}")
    
    # Convert model if needed
    if not model.config.model_path.exists():
        logger.info("Converting model to ONNX format...")
        model.convert()
    
    # Apply optimizations
    if quantize:
        logger.info("Quantizing model for better performance...")
        model.quantize()
    
    if optimize:
        logger.info("Optimizing model...")
        model.optimize()
    
    # Deploy to server
    client = model.deploy(server_url=server_url)
    
    logger.info("Model deployed successfully!")
    return client


def quick_deploy(
    model_name: str,
    task: str = "text-classification",
    server_url: str = "localhost:8000",
    **kwargs
) -> Dict[str, Any]:
    """Quick deployment with minimal configuration."""
    
    from ..tasks import get_task_model
    
    # Get appropriate model class
    model_class = get_task_model(task)
    
    # Load model
    model = model_class.from_pretrained(model_name, **kwargs)
    
    # Deploy with optimizations
    client = deploy(model, server_url=server_url, quantize=True, optimize=True)
    
    # Return deployment info
    return {
        "model_name": model.config.model_name,
        "server_url": server_url,
        "model_path": str(model.config.model_path),
        "client": client,
        "model": model
    }


def deploy_from_config(config_path: Union[str, Path]) -> TritonClient:
    """Deploy a model from a configuration file."""
    import json
    
    config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract deployment settings
    model_name = config.get("model_name")
    task = config.get("task", "text-classification")
    server_url = config.get("server_url", "localhost:8000")
    model_config = config.get("model_config", {})
    
    return quick_deploy(
        model_name=model_name,
        task=task,
        server_url=server_url,
        **model_config
    )["client"]