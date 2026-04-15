# Contributing to Tinker

Good day! Thank you for your interest in contributing to Tinker.

We welcome contributions! Please follow these guidelines to help us maintain the project quality.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/thinking-machines-lab/tinker.git
cd tinker

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run tests
pytest
```

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Run `ruff check .` to lint
- Run `ruff format .` to format
- Type hints are required via mypy

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Run the test suite
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Pull Request Guidelines

- PRs should pass all tests
- Include a clear description of changes
- Link any related issues

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=tinker
```

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.