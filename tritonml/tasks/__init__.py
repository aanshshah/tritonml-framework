"""Task-specific model implementations."""

from .text_classification import TextClassificationModel
from .image_classification import ImageClassificationModel
from .registry import get_task_model, register_task_model

# Register default task models
register_task_model("text-classification", TextClassificationModel)
register_task_model("image-classification", ImageClassificationModel)

__all__ = [
    "TextClassificationModel",
    "ImageClassificationModel", 
    "get_task_model",
    "register_task_model"
]