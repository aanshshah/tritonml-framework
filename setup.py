"""Setup script for TritonML framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tritonml",
    version="0.1.0",
    author="TritonML Contributors",
    description="A framework for deploying ML models to Triton Server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aanshshah/tritonml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "tritonclient[http]>=2.36.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "optimum>=1.13.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tritonml=tritonml.cli.main:cli",
        ],
    },
)
