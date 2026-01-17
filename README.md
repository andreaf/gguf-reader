# GGUF Reader - Kiro experiment

A Python library for parsing GGUF (GPT-Generated Unified Format) files used for storing large language models generated via Kiro as experiment

## Overview

GGUF is a binary file format designed for storing large language models with their metadata and tensor data. This library provides a clean interface to read and parse GGUF files without requiring external dependencies beyond Python's standard library.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from gguf_reader import GGUFReader

# Use context manager for automatic resource management
with GGUFReader('model.gguf') as reader:
    # Access metadata
    metadata = reader.get_metadata()
    architecture = reader.get_metadata_value('general.architecture')
    
    # List and access tensors
    tensors = reader.list_tensors()
    tensor_info = reader.get_tensor_info('token_embd.weight')
    tensor_data = reader.get_tensor_data('token_embd.weight')
    
    # Get file information
    version = reader.get_version()
    tensor_count = reader.get_tensor_count()
```

### Example Script

A comprehensive example script is provided that demonstrates all features:

```bash
# Create a demo GGUF file
python create_demo_gguf.py

# Run the example script
python example_usage.py demo_model.gguf
```

The example script (`example_usage.py`) demonstrates:
- Context manager usage for automatic resource management
- Accessing and displaying metadata
- Listing and inspecting tensors
- Reading tensor data with caching
- Error handling for various failure scenarios

## Development

### Running Tests

Run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest test_gguf_reader.py
```

### Project Structure

- `gguf_reader.py` - Main library module
- `test_gguf_reader.py` - Test suite
- `test_tensor_edge_cases.py` - Additional tensor type tests
- `example_usage.py` - Comprehensive usage examples
- `create_demo_gguf.py` - Script to create a demo GGUF file
- `requirements.txt` - Python dependencies
- `pytest.ini` - Pytest configuration

## Features

- ✅ Type definitions for GGUF value types and GGML tensor types
- ✅ Custom exception classes for error handling
- ✅ Type size mappings for all tensor types (standard and quantized)
- ✅ Context manager support for resource management
- ✅ Header parsing with validation
- ✅ Metadata parsing (all primitive types and arrays)
- ✅ Tensor information parsing
- ✅ Tensor data reading with caching
- ✅ Comprehensive error handling
- ✅ Example scripts and documentation

## Testing

The project uses a dual testing approach:

- **Unit Tests**: Test specific examples and edge cases
- **Property-Based Tests**: Test universal properties across randomly generated inputs using Hypothesis

## License

This project is part of the GGUF Reader specification implementation.
