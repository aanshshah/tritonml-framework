"""Docker deployment utilities for TritonML."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def generate_dockerfile(
    base_image: str = "nvcr.io/nvidia/tritonserver:24.08-py3",
    model_repository: str = "/models",
    additional_packages: Optional[list] = None,
) -> str:
    """Generate a Dockerfile for Triton deployment."""

    dockerfile = f"""FROM {base_image}

# Install additional Python packages if needed
"""

    if additional_packages:
        packages = " ".join(additional_packages)
        dockerfile += f"RUN pip install {packages}\n"

    dockerfile += (
        f"\n# Set working directory\n"
        f"WORKDIR /workspace\n"
        f"\n# Copy model repository\n"
        f"COPY ./models {model_repository}\n"
        f"\n# Expose Triton ports\n"
        f"EXPOSE 8000 8001 8002\n"
        f"\n# Set model repository path\n"
        f"ENV MODEL_REPOSITORY_PATH={model_repository}\n"
        f"\n# Start Triton server\n"
        f'CMD ["tritonserver", "--model-repository=/models", '
        f'"--strict-model-config=false"]\n'
    )

    return dockerfile


def generate_docker_compose(
    service_name: str = "triton",
    model_repository: str = "./models",
    ports: Dict[str, int] = None,
    environment: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Generate docker-compose configuration."""

    if ports is None:
        ports = {"8000": 8000, "8001": 8001, "8002": 8002}  # HTTP  # GRPC  # Metrics

    if environment is None:
        environment = {}

    compose_config = {
        "version": "3.8",
        "services": {
            service_name: {
                "image": "nvcr.io/nvidia/tritonserver:24.08-py3",
                "ports": [f"{host}:{container}" for host, container in ports.items()],
                "volumes": [f"{model_repository}:/models"],
                "environment": environment,
                "command": [
                    "tritonserver",
                    "--model-repository=/models",
                    "--strict-model-config=false",
                ],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": "all",
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                },
            }
        },
    }

    return compose_config


def save_docker_files(
    output_path: Path,
    dockerfile_content: Optional[str] = None,
    compose_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save Docker deployment files."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if dockerfile_content:
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        print(f"Created {dockerfile_path}")

    if compose_config:
        compose_path = output_path / "docker-compose.yml"
        with open(compose_path, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        print(f"Created {compose_path}")


def create_deployment_package(
    model_name: str, output_path: Path, include_client: bool = True
) -> None:
    """Create a complete deployment package with Docker files."""

    output_path = Path(output_path)

    # Generate Dockerfile
    dockerfile = generate_dockerfile()

    # Generate docker-compose
    compose = generate_docker_compose(
        service_name=f"triton-{model_name}", environment={"MODEL_NAME": model_name}
    )

    # Save files
    save_docker_files(output_path, dockerfile, compose)

    # Create client example if requested
    if include_client:
        client_example = f"""#!/usr/bin/env python3
\"\"\"Client example for {model_name} model.\"\"\"

from tritonml import TritonClient

# Initialize client
client = TritonClient(
    server_url="localhost:8000",
    model_name="{model_name}"
)

# Check model is ready
if client.is_model_ready():
    print("Model is ready!")

    # Example inference
    # Adjust inputs based on your model
    inputs = {{
        "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]]
    }}

    outputs = client.infer(inputs)
    print(f"Outputs: {{outputs}}")
else:
    print("Model is not ready")
"""

        client_path = output_path / "client_example.py"
        client_path.write_text(client_example)
        client_path.chmod(0o755)
        print(f"Created {client_path}")

    # Create README
    readme = f"""# {model_name} Triton Deployment

## Quick Start

1. Start Triton server:
   ```bash
   docker-compose up
   ```

2. Test the deployment:
   ```bash
   python client_example.py
   ```

## Files

- `Dockerfile` - Custom Triton server image
- `docker-compose.yml` - Docker Compose configuration
- `client_example.py` - Example client code
- `models/` - Model repository (place your models here)

## Ports

- 8000: HTTP API
- 8001: GRPC API
- 8002: Metrics

## Environment Variables

- `MODEL_NAME`: Name of the model ({model_name})
- `MODEL_REPOSITORY_PATH`: Path to model repository (/models)
"""

    readme_path = output_path / "README.md"
    readme_path.write_text(readme)
    print(f"Created {readme_path}")

    print(f"\nDeployment package created at {output_path}")
    print("Run 'docker-compose up' to start the server")
