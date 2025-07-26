# Deep-Seek-temporal-reasoner

[![Python Package CI/CD](https://github.com/9hostguard/Deep-Seek-temporal-reasoner/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/9hostguard/Deep-Seek-temporal-reasoner/actions/workflows/python-package-conda.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project demonstrates 4D augmentation for Large Language Models (LLMs) using Deep Seek models, with a focus on temporal reasoning and decomposition.

## Features

- **Temporal Decomposition**: Break down prompts into past, present, and future components
- **Deep Seek Integration**: Compatible with Deep Seek API for advanced reasoning
- **Modern Python**: Built with Python 3.9+ features and best practices
- **Comprehensive Testing**: Full test suite with pytest
- **CI/CD Pipeline**: Automated testing and dependency management
- **Type Safety**: Type hints and mypy checking

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/9hostguard/Deep-Seek-temporal-reasoner.git
cd Deep-Seek-temporal-reasoner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

## Usage

### Basic Temporal Decomposition

```python
from core.decomposition import decompose

# Decompose a prompt into temporal components
prompt = "I was working yesterday, am working today, and will work tomorrow."
result = decompose(prompt)

print(result)
# Output: {
#     'past': 'I was working yesterday',
#     'present': 'am working today',
#     'future': 'will work tomorrow'
# }
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_decomposition.py
```

### Development

```bash
# Install development dependencies
pip install black isort mypy flake8

# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Lint code
flake8 .
```

## API Reference

### `decompose(prompt: str) -> Dict[str, str]`

Decomposes a given prompt into temporal components.

**Parameters:**
- `prompt` (str): The input text to analyze for temporal components

**Returns:**
- `Dict[str, str]`: Dictionary containing 'past', 'present', and 'future' keys with corresponding text segments

**Example:**
```python
result = decompose("I learned Python last year and use it daily now.")
# result['past'] contains past-related content
# result['present'] contains present-related content  
# result['future'] contains future-related content
```

## Configuration

The application can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | Your Deep Seek API key | `your-key-here` |
| `DEEPSEEK_BASE_URL` | Deep Seek API base URL | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | Model to use | `deepseek-chat` |
| `DEBUG` | Enable debug mode | `false` |
| `MAX_PROMPT_LENGTH` | Maximum prompt length | `4096` |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v1.0.0 (Latest)
- ✅ **Core Implementation**: Added temporal decomposition functionality
- ✅ **Modern Python**: Updated to Python 3.9+ with type hints
- ✅ **Dependencies**: Upgraded all dependencies to latest stable versions
- ✅ **Testing**: Comprehensive test suite with pytest
- ✅ **CI/CD**: GitHub Actions workflow with automated testing
- ✅ **Code Quality**: Black formatting, isort, mypy type checking
- ✅ **Documentation**: Enhanced README and API documentation
- ✅ **Environment**: Proper environment variable management
- ✅ **Automation**: Automated dependency updates via GitHub Actions

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the existing documentation
- Review the test cases for usage examples
