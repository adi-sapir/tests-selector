# Tests Selector

A Python project for selecting and managing tests.

## Installation

### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/tests-selector.git
cd tests-selector

# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the package in development mode with all dependencies
uv pip install -e ".[dev]"
```

### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/yourusername/tests-selector.git
cd tests-selector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the package in development mode
pip install -e ".[dev]"
```

## Development

This project uses several development tools:

- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

To run the development tools:

```bash
# Run tests
pytest

# Format code
black .

# Run linter
flake8

# Run type checker
mypy .
```

### Using uv for Development

```bash
# Update dependencies
uv pip sync requirements.txt

# Install a new package
uv pip install package-name

# Update requirements.txt after adding new packages
uv pip freeze > requirements.txt
```

## Project Structure

```
tests-selector/
├── src/                    # Source code
│   └── tests_selector/     # Main package
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
├── setup.py               # Package configuration
└── README.md              # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 