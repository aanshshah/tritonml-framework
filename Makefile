.PHONY: help install install-dev test lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  install      Install TritonML in production mode"
	@echo "  install-dev  Install TritonML in development mode"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"
	@echo "  docs         Serve documentation locally"

install:
	pip install -r requirements.txt
	pip install .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest

test-cov:
	pytest --cov=tritonml --cov-report=html --cov-report=term

lint:
	black --check tritonml tests
	isort --check-only tritonml tests
	flake8 tritonml tests
	mypy tritonml

format:
	black tritonml tests
	isort tritonml tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && python -m http.server 8000

publish-test: build
	python -m twine upload --repository testpypi dist/*

publish: build
	python -m twine upload dist/*