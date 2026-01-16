#!/usr/bin/env python3
"""
Example Usage of GGUF Reader

This script demonstrates how to use the GGUF Reader library to parse and
access data from GGUF (GPT-Generated Unified Format) files.

The examples cover:
- Context manager usage for automatic resource management
- Accessing metadata (model configuration and parameters)
- Listing and retrieving tensor information
- Reading tensor data
- Error handling for common issues
"""

import sys
from gguf_reader import (
    GGUFReader,
    GGUFFileError,
    GGUFInvalidMagicError,
    GGUFVersionError,
    GGUFParseError,
    GGUFTruncatedError,
    GGUFInvalidTypeError
)


def example_basic_usage(filepath: str):
    """
    Example 1: Basic usage with context manager
    
    The recommended way to use GGUFReader is with a context manager (with statement).
    This ensures the file is properly opened and closed, even if an error occurs.
    """
    print("=" * 70)
    print("Example 1: Basic Usage with Context Manager")
    print("=" * 70)
    
    try:
        # Open the GGUF file using context manager
        with GGUFReader(filepath) as reader:
            # Get basic file information
            version = reader.get_version()
            tensor_count = reader.get_tensor_count()
            
            print(f"✓ Successfully opened: {filepath}")
            print(f"  GGUF Version: {version}")
            print(f"  Number of tensors: {tensor_count}")
            print()
            
    except GGUFFileError as e:
        print(f"✗ Error reading GGUF file: {e}")
        print()


def example_metadata_access(filepath: str):
    """
    Example 2: Accessing metadata
    
    GGUF files contain metadata as key-value pairs that describe the model
    architecture, parameters, tokenizer information, and more.
    """
    print("=" * 70)
    print("Example 2: Accessing Metadata")
    print("=" * 70)
    
    try:
        with GGUFReader(filepath) as reader:
            # Get all metadata as a dictionary
            metadata = reader.get_metadata()
            
            print(f"Total metadata entries: {len(metadata)}")
            print()
            
            # Display some common metadata keys (if they exist)
            common_keys = [
                'general.architecture',
                'general.name',
                'general.alignment',
                'llama.context_length',
                'llama.embedding_length',
                'llama.block_count',
                'llama.attention.head_count',
                'llama.feed_forward_length'
            ]
            
            print("Common metadata values:")
            for key in common_keys:
                try:
                    value = reader.get_metadata_value(key)
                    print(f"  {key}: {value}")
                except KeyError:
                    # Key doesn't exist in this file
                    pass
            
            print()
            
            # Display first 10 metadata keys (to show what's available)
            print("First 10 metadata keys:")
            for i, key in enumerate(sorted(metadata.keys())[:10]):
                value = metadata[key]
                # Truncate long values for display
                if isinstance(value, list) and len(value) > 3:
                    value_str = f"[{value[0]}, {value[1]}, ... ({len(value)} items)]"
                elif isinstance(value, str) and len(value) > 50:
                    value_str = value[:47] + "..."
                else:
                    value_str = str(value)
                print(f"  {i+1}. {key}: {value_str}")
            
            print()
            
    except GGUFFileError as e:
        print(f"✗ Error reading metadata: {e}")
        print()


def example_tensor_listing(filepath: str):
    """
    Example 3: Listing and inspecting tensors
    
    GGUF files contain tensors (multi-dimensional arrays) that store model weights
    and parameters. You can list all tensors and inspect their properties.
    """
    print("=" * 70)
    print("Example 3: Listing and Inspecting Tensors")
    print("=" * 70)
    
    try:
        with GGUFReader(filepath) as reader:
            # Get list of all tensor names
            tensor_names = reader.list_tensors()
            
            print(f"Total tensors: {len(tensor_names)}")
            print()
            
            # Display first 5 tensors with their information
            print("First 5 tensors:")
            for i, name in enumerate(tensor_names[:5]):
                # Get detailed information about this tensor
                info = reader.get_tensor_info(name)
                
                # Format dimensions as a readable string
                dims_str = " × ".join(str(d) for d in info['dims'])
                
                # Get type name (for display)
                type_code = info['type']
                
                print(f"  {i+1}. {name}")
                print(f"     Shape: [{dims_str}]")
                print(f"     Type: {type_code}")
                print(f"     Dimensions: {info['n_dims']}")
                print(f"     Offset: {info['offset']} bytes")
                print()
            
            # Show summary of remaining tensors
            if len(tensor_names) > 5:
                print(f"  ... and {len(tensor_names) - 5} more tensors")
                print()
            
    except GGUFFileError as e:
        print(f"✗ Error listing tensors: {e}")
        print()


def example_tensor_data_reading(filepath: str):
    """
    Example 4: Reading tensor data
    
    You can read the raw binary data for any tensor. The data is returned as bytes
    and is cached after the first read for efficiency.
    """
    print("=" * 70)
    print("Example 4: Reading Tensor Data")
    print("=" * 70)
    
    try:
        with GGUFReader(filepath) as reader:
            # Get the first tensor name
            tensor_names = reader.list_tensors()
            if not tensor_names:
                print("No tensors found in file")
                print()
                return
            
            # Read data from the first tensor
            first_tensor = tensor_names[0]
            print(f"Reading data from tensor: {first_tensor}")
            
            # Get tensor info
            info = reader.get_tensor_info(first_tensor)
            dims_str = " × ".join(str(d) for d in info['dims'])
            
            # Read the raw tensor data
            data = reader.get_tensor_data(first_tensor)
            
            print(f"  Shape: [{dims_str}]")
            print(f"  Data size: {len(data)} bytes")
            print(f"  First 32 bytes (hex): {data[:32].hex()}")
            print()
            
            # Reading the same tensor again uses cached data (no file I/O)
            data_again = reader.get_tensor_data(first_tensor)
            print(f"✓ Data is cached (second read returns same data)")
            print(f"  Data matches: {data == data_again}")
            print()
            
    except GGUFFileError as e:
        print(f"✗ Error reading tensor data: {e}")
        print()


def example_error_handling():
    """
    Example 5: Error handling
    
    The GGUF Reader provides specific exception types for different error conditions,
    making it easy to handle errors appropriately.
    """
    print("=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)
    
    # Example 5a: File not found
    print("5a. Handling file not found:")
    try:
        with GGUFReader('nonexistent_file.gguf') as reader:
            pass
    except GGUFFileError as e:
        print(f"  ✓ Caught error: {type(e).__name__}")
        print(f"    Message: {e}")
    print()
    
    # Example 5b: Invalid magic number (not a GGUF file)
    print("5b. Handling invalid file format:")
    try:
        # Create a temporary file with invalid magic number
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.gguf', delete=False) as f:
            temp_path = f.name
            # Write invalid magic number
            f.write(b'\x00\x00\x00\x00')
            f.write(b'\x00' * 20)  # Some padding
        
        try:
            with GGUFReader(temp_path) as reader:
                pass
        except GGUFInvalidMagicError as e:
            print(f"  ✓ Caught error: {type(e).__name__}")
            print(f"    Message: {str(e)[:80]}...")
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_path)
    except Exception as e:
        print(f"  Note: Could not create temp file for demo: {e}")
    print()
    
    # Example 5c: Accessing non-existent metadata key
    print("5c. Handling missing metadata key:")
    try:
        # We'll use a valid file if available, otherwise skip
        import os
        test_files = ['model.gguf', 'test.gguf']
        test_file = None
        for f in test_files:
            if os.path.exists(f):
                test_file = f
                break
        
        if test_file:
            with GGUFReader(test_file) as reader:
                # Try to access a key that doesn't exist
                value = reader.get_metadata_value('nonexistent.key')
        else:
            # Simulate the error for demonstration
            raise KeyError("Metadata key 'nonexistent.key' not found")
            
    except KeyError as e:
        print(f"  ✓ Caught error: {type(e).__name__}")
        print(f"    Message: {e}")
    print()
    
    # Example 5d: Accessing non-existent tensor
    print("5d. Handling missing tensor:")
    try:
        import os
        test_files = ['model.gguf', 'test.gguf']
        test_file = None
        for f in test_files:
            if os.path.exists(f):
                test_file = f
                break
        
        if test_file:
            with GGUFReader(test_file) as reader:
                # Try to access a tensor that doesn't exist
                info = reader.get_tensor_info('nonexistent.tensor')
        else:
            # Simulate the error for demonstration
            raise KeyError("Tensor 'nonexistent.tensor' not found")
            
    except KeyError as e:
        print(f"  ✓ Caught error: {type(e).__name__}")
        print(f"    Message: {e}")
    print()


def main():
    """
    Main function to run all examples.
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "GGUF Reader - Example Usage" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Check if a file path was provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Using GGUF file: {filepath}")
        print()
        
        # Run examples that require a file
        example_basic_usage(filepath)
        example_metadata_access(filepath)
        example_tensor_listing(filepath)
        example_tensor_data_reading(filepath)
    else:
        print("Note: No GGUF file provided as command-line argument.")
        print("      Some examples will be skipped.")
        print()
        print("Usage: python example_usage.py <path_to_gguf_file>")
        print()
    
    # Run error handling examples (don't require a valid file)
    example_error_handling()
    
    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
