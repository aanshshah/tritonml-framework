"""Model registry for task-specific implementations."""

from typing import Dict, Type, Optional
from ..core.model import TritonModel

# Global registry of task models
_TASK_REGISTRY: Dict[str, Type[TritonModel]] = {}


def register_task_model(task_name: str, model_class: Type[TritonModel]) -> None:
    """Register a model class for a specific task."""
    _TASK_REGISTRY[task_name.lower()] = model_class


def get_task_model(task_name: str) -> Type[TritonModel]:
    """Get the model class for a specific task."""
    task_name = task_name.lower()
    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[task_name]


def list_tasks() -> list[str]:
    """List all registered tasks."""
    return list(_TASK_REGISTRY.keys())