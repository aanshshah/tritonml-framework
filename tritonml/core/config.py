"""Configuration management for TritonML models."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TritonConfig:
    """Base configuration for Triton models."""

    # Model identification
    model_name: str
    model_version: str = "1"

    # Server configuration
    triton_server_url: str = field(
        default_factory=lambda: os.environ.get("TRITON_SERVER_URL", "localhost:8000")
    )

    # Model settings
    max_batch_size: int = 32
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)

    # Paths
    model_repository_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MODEL_REPOSITORY_PATH", "./models")
        )
    )

    # Performance settings
    instance_group: Dict[str, Any] = field(
        default_factory=lambda: {"kind": "KIND_CPU", "count": 1}
    )
    dynamic_batching: Dict[str, Any] = field(default_factory=dict)

    # Model-specific configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_path(self) -> Path:
        """Get the full model path in the repository."""
        return self.model_repository_path / self.model_name / self.model_version

    @property
    def config_path(self) -> Path:
        """Get the path to the Triton config.pbtxt file."""
        return self.model_repository_path / self.model_name / "config.pbtxt"

    def to_pbtxt(self) -> str:
        """Convert configuration to Triton's config.pbtxt format."""
        config_lines = [
            f'name: "{self.model_name}"',
            f"max_batch_size: {self.max_batch_size}",
        ]

        # Add input configuration
        for input_name, shape in self.input_shapes.items():
            config_lines.append("input [")
            config_lines.append("  {")
            config_lines.append(f'    name: "{input_name}"')
            dtype = self.model_config.get(f"{input_name}_dtype", "TYPE_INT64")
            config_lines.append(f"    data_type: {dtype}")
            config_lines.append(f"    dims: {shape}")
            config_lines.append("  }")
            config_lines.append("]")

        # Add output configuration
        for output_name, shape in self.output_shapes.items():
            config_lines.append("output [")
            config_lines.append("  {")
            config_lines.append(f'    name: "{output_name}"')
            dtype = self.model_config.get(f"{output_name}_dtype", "TYPE_FP32")
            config_lines.append(f"    data_type: {dtype}")
            config_lines.append(f"    dims: {shape}")
            config_lines.append("  }")
            config_lines.append("]")

        # Add instance group configuration
        if self.instance_group:
            config_lines.append("instance_group [")
            config_lines.append("  {")
            for key, value in self.instance_group.items():
                if isinstance(value, str):
                    config_lines.append(f"    {key}: {value}")
                else:
                    config_lines.append(f"    {key}: {value}")
            config_lines.append("  }")
            config_lines.append("]")

        # Add dynamic batching if configured
        if self.dynamic_batching:
            config_lines.append("dynamic_batching {")
            for key, value in self.dynamic_batching.items():
                config_lines.append(f"  {key}: {value}")
            config_lines.append("}")

        return "\n".join(config_lines)

    def save_config(self) -> None:
        """Save the configuration to config.pbtxt."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(self.to_pbtxt())


@dataclass
class TextClassificationConfig(TritonConfig):
    """Configuration specific to text classification models."""

    # Text-specific settings
    max_sequence_length: int = 128
    labels: List[str] = field(default_factory=list)
    tokenizer_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default shapes for text classification."""
        if not self.input_shapes:
            self.input_shapes = {
                "input_ids": [self.max_sequence_length],
                "attention_mask": [self.max_sequence_length],
            }
        if not self.output_shapes:
            self.output_shapes = {"logits": [len(self.labels)] if self.labels else [-1]}
