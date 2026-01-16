# Design Document: GGUF Reader

## Overview

The GGUF Reader is a Python library that parses GGUF (GPT-Generated Unified Format) binary files used for storing large language models. The reader provides a clean interface to access file metadata, tensor information, and raw tensor data without requiring external dependencies beyond Python's standard library.

The design follows a sequential parsing approach that mirrors the GGUF file structure:
1. Read and validate the file header
2. Parse metadata key-value pairs
3. Parse tensor information records
4. Calculate aligned offsets for tensor data
5. Read tensor data on demand

The implementation uses Python's `struct` module for binary parsing and provides both direct access methods and context manager support for resource management.

## Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────┐
│                     GGUFReader                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              File Parser                          │  │
│  │  - Read header                                    │  │
│  │  - Parse metadata                                 │  │
│  │  - Parse tensor info                              │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │           Data Structures                         │  │
│  │  - Header info                                    │  │
│  │  - Metadata dictionary                            │  │
│  │  - Tensor info list                               │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Access Interface                         │  │
│  │  - Get metadata                                   │  │
│  │  - Get tensor info                                │  │
│  │  - Read tensor data                               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**File Parser**: Handles low-level binary reading and parsing of GGUF structures
**Data Structures**: Stores parsed information in Python-native formats
**Access Interface**: Provides high-level methods for accessing parsed data

## Components and Interfaces

### GGUFReader Class

The main class that encapsulates all functionality.

```python
class GGUFReader:
    def __init__(self, filepath: str):
        """Initialize reader with path to GGUF file."""
        
    def __enter__(self):
        """Context manager entry - opens and parses file."""
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes file."""
        
    def get_metadata(self) -> dict:
        """Return all metadata as dictionary."""
        
    def get_metadata_value(self, key: str):
        """Return specific metadata value by key."""
        
    def list_tensors(self) -> list[str]:
        """Return list of all tensor names."""
        
    def get_tensor_info(self, name: str) -> dict:
        """Return tensor metadata (shape, type, offset)."""
        
    def get_tensor_data(self, name: str) -> bytes:
        """Return raw tensor data as bytes."""
        
    def get_tensor_count(self) -> int:
        """Return total number of tensors."""
        
    def get_version(self) -> int:
        """Return GGUF format version."""
```

### Internal Parsing Methods

```python
def _read_header(self):
    """Read and validate GGUF header (magic, version, counts)."""
    
def _read_string(self) -> str:
    """Read length-prefixed UTF-8 string."""
    
def _read_metadata(self):
    """Parse all metadata key-value pairs."""
    
def _read_value(self, value_type: int):
    """Read a single metadata value based on type."""
    
def _read_array(self):
    """Read array metadata value."""
    
def _read_tensor_info(self):
    """Parse all tensor information records."""
    
def _calculate_tensor_offset(self, tensor_info: dict) -> int:
    """Calculate aligned file offset for tensor data."""
    
def _calculate_tensor_size(self, tensor_info: dict) -> int:
    """Calculate size in bytes of tensor data."""
```

### Type Enumerations

```python
class GGUFValueType:
    """Metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGMLType:
    """Tensor data types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18
```

## Data Models

### Header Structure

```python
{
    'magic': int,        # 4-byte magic number
    'version': int,      # uint32 version
    'tensor_count': int, # uint64 number of tensors
    'metadata_kv_count': int  # uint64 number of metadata pairs
}
```

### Metadata Structure

Stored as a flat dictionary where keys are strings and values can be:
- Primitive types (int, float, bool, str)
- Arrays of primitive types (list)

```python
{
    'general.architecture': 'llama',
    'general.name': 'LLaMA v2',
    'llama.context_length': 2048,
    'llama.embedding_length': 4096,
    'llama.block_count': 32,
    'tokenizer.ggml.tokens': ['<s>', '</s>', ...],
    # ... more metadata
}
```

### Tensor Info Structure

```python
{
    'name': str,           # Tensor name (e.g., 'token_embd.weight')
    'n_dims': int,         # Number of dimensions
    'dims': list[int],     # Dimension sizes [d0, d1, d2, d3]
    'type': int,           # GGMLType enum value
    'offset': int          # Offset from tensor data section start
}
```

### Type Size Mapping

```python
TYPE_SIZES = {
    # Standard types
    GGMLType.F32: (4, 1),      # (bytes per element, block size)
    GGMLType.F16: (2, 1),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (2, 1),
    GGMLType.I32: (4, 1),
    
    # Quantized types
    GGMLType.Q4_0: (18, 32),   # 18 bytes per 32-element block
    GGMLType.Q4_1: (20, 32),
    GGMLType.Q5_0: (22, 32),
    GGMLType.Q5_1: (24, 32),
    GGMLType.Q8_0: (34, 32),
    GGMLType.Q8_1: (40, 32),
    GGMLType.Q2_K: (82, 256),
    GGMLType.Q3_K: (110, 256),
    GGMLType.Q4_K: (144, 256),
    GGMLType.Q5_K: (176, 256),
    GGMLType.Q6_K: (210, 256),
    GGMLType.Q8_K: (292, 256),
}
```

## Parsing Algorithm

### Sequential Parsing Flow

```
1. Open file in binary read mode
2. Read header (magic + version + counts)
3. Validate magic number
4. For each metadata key-value pair:
   a. Read key string
   b. Read value type
   c. Read value (recursive for arrays)
   d. Store in metadata dict
5. For each tensor:
   a. Read tensor name
   b. Read dimension count
   c. Read dimension sizes
   d. Read data type
   e. Read offset
   f. Store tensor info
6. Calculate alignment padding
7. Calculate base offset for tensor data
8. Keep file open for on-demand tensor data reads
```

### Alignment Calculation

```python
# After parsing all tensor info
alignment = metadata.get('general.alignment', 32)
current_position = file.tell()
padding = (alignment - (current_position % alignment)) % alignment
tensor_data_base = current_position + padding
```

### Tensor Data Size Calculation

```python
type_size, block_size = TYPE_SIZES[tensor_type]
elements_in_first_dim = dims[0]
blocks = elements_in_first_dim // block_size
size = type_size * blocks

# Multiply by remaining dimensions
for dim in dims[1:]:
    size *= dim
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Magic Number Validation

*For any* byte sequence, when passed to the reader as a file, if the first 4 bytes match the GGUF magic number then parsing should proceed, otherwise an error should be raised.

**Validates: Requirements 1.1, 1.5**

### Property 2: Header Field Round-Trip

*For any* valid GGUF header values (version, tensor_count, metadata_kv_count), when written to a file and parsed by the reader, the parsed values should match the original values.

**Validates: Requirements 1.2, 1.3, 1.4**

### Property 3: Metadata Round-Trip

*For any* valid metadata dictionary containing primitive types and arrays, when encoded in GGUF format and parsed by the reader, the parsed metadata should equal the original metadata.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8**

### Property 4: Invalid Metadata Type Rejection

*For any* metadata value with an invalid type code (not in the valid GGUF type enumeration), the reader should raise an error when attempting to parse it.

**Validates: Requirements 2.7**

### Property 5: Tensor Info Round-Trip

*For any* valid tensor information (name, dimensions, type, offset), when encoded in GGUF format and parsed by the reader, the parsed tensor info should match the original values.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

### Property 6: Alignment Calculation

*For any* file position and alignment value, the calculated aligned position should be the smallest value >= file position that is divisible by the alignment value.

**Validates: Requirements 4.1, 4.3**

### Property 7: Default Alignment

*For any* GGUF file without a "general.alignment" metadata key, the reader should use 32 bytes as the alignment value.

**Validates: Requirements 4.2**

### Property 8: Tensor Offset Calculation

*For any* tensor with a given offset value, the absolute file position for that tensor's data should equal the aligned base position plus the tensor's offset.

**Validates: Requirements 4.4**

### Property 9: Tensor Size Calculation

*For any* tensor with dimensions [d0, d1, d2, d3] and type with (type_size, block_size), the calculated data size should equal: type_size * (d0 / block_size) * d1 * d2 * d3.

**Validates: Requirements 5.3**

### Property 10: Tensor Data Round-Trip

*For any* tensor with valid type and dimensions, when tensor data is written to a GGUF file and read back by the reader, the read bytes should match the original bytes.

**Validates: Requirements 5.1, 5.2, 5.4, 5.5**

### Property 11: Metadata Retrieval

*For any* parsed GGUF file with metadata, retrieving all metadata should return a dictionary containing all key-value pairs, and retrieving by specific key should return the correct value.

**Validates: Requirements 6.1, 6.2**

### Property 12: Tensor Retrieval

*For any* parsed GGUF file with tensors, listing tensor names should return all tensor names, and retrieving tensor info or data by name should return the correct information.

**Validates: Requirements 6.3, 6.4, 6.5**

### Property 13: Count Retrieval

*For any* parsed GGUF file, the reported tensor count and version should match the values in the file header.

**Validates: Requirements 6.6, 6.7**

### Property 14: Truncated File Detection

*For any* file that is shorter than the expected size based on header information, the reader should raise an error indicating truncation or unexpected EOF.

**Validates: Requirements 7.2, 7.5**

### Property 15: Invalid Version Rejection

*For any* GGUF file with an unsupported version number, the reader should raise an error containing the version number.

**Validates: Requirements 7.3**

### Property 16: Invalid Type Rejection

*For any* GGUF file containing an invalid type code in metadata or tensor info, the reader should raise an error indicating the invalid type.

**Validates: Requirements 7.4**

### Property 17: Invalid String Length Rejection

*For any* string with a length value that would read past the end of file or is unreasonably large, the reader should raise an error.

**Validates: Requirements 7.6**

## Error Handling

The reader implements a fail-fast approach with descriptive error messages:

### Error Types

1. **GGUFFileError**: Base exception for all GGUF-related errors
2. **GGUFInvalidMagicError**: Magic number doesn't match GGUF format
3. **GGUFVersionError**: Unsupported GGUF version
4. **GGUFParseError**: Generic parsing error with context
5. **GGUFTruncatedError**: File ends unexpectedly
6. **GGUFInvalidTypeError**: Invalid type code encountered

### Error Context

All errors include:
- File path
- Current file position when error occurred
- Specific value that caused the error (when applicable)
- Expected vs actual values (for validation errors)

### Example Error Messages

```
GGUFInvalidMagicError: Invalid GGUF magic number in file 'model.gguf' at position 0: 
  expected 0x46554747, got 0x12345678

GGUFVersionError: Unsupported GGUF version in file 'model.gguf': 
  version 5 (only versions 1-3 supported)

GGUFTruncatedError: Unexpected end of file 'model.gguf' at position 1024: 
  expected to read 8 bytes for tensor count, only 3 bytes available

GGUFInvalidTypeError: Invalid metadata type in file 'model.gguf' at position 256: 
  type code 99 is not a valid GGUF type
```

## Testing Strategy

The GGUF Reader will be tested using a dual approach combining property-based testing and unit testing.

### Property-Based Testing

Property-based tests will validate universal correctness properties across randomly generated inputs. Each property test will:
- Run a minimum of 100 iterations with randomized inputs
- Generate valid GGUF file structures with random data
- Verify the property holds for all generated inputs
- Tag tests with format: **Feature: gguf-reader, Property N: [property description]**

We will use the `hypothesis` library for property-based testing in Python, which provides:
- Strategies for generating random data (integers, strings, bytes, lists)
- Automatic shrinking of failing test cases to minimal examples
- Stateful testing for complex scenarios
- Integration with pytest

**Key Property Tests:**
- Round-trip properties for all data structures (metadata, tensor info, tensor data)
- Alignment calculation correctness
- Size calculation correctness
- Error detection for invalid inputs

### Unit Testing

Unit tests will focus on:
- Specific examples of valid GGUF files
- Edge cases (empty metadata, single tensor, maximum dimensions)
- Error conditions (invalid magic, truncated files, invalid types)
- Context manager behavior
- Integration between components

**Test Coverage Areas:**
- Header parsing with various version numbers
- All metadata types (primitives and arrays)
- All tensor types (standard and quantized)
- Alignment edge cases (already aligned, needs padding)
- File resource management (context manager)
- Error messages contain expected information

### Test Data Generation

For property-based tests, we will create generators for:
- Random valid GGUF headers
- Random metadata dictionaries with mixed types
- Random tensor information with valid dimensions
- Random tensor data matching type specifications
- Invalid inputs for error testing (bad magic, bad types, truncated data)

### Integration Testing

Integration tests will use real GGUF files (if available) or realistic synthetic files to verify:
- Complete file parsing from start to finish
- Correct interaction between all components
- Performance with large files
- Memory efficiency (lazy loading of tensor data)
