# Implementation Plan: GGUF Reader

## Overview

This implementation plan breaks down the GGUF Reader into discrete coding tasks. The approach follows a bottom-up strategy: first implementing low-level parsing utilities, then building up to complete file parsing, and finally adding the high-level access interface. Each task includes property-based tests to validate correctness.

## Tasks

- [x] 1. Set up project structure and core type definitions
  - Create `gguf_reader.py` module
  - Define `GGUFValueType` enumeration with all metadata types (UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64)
  - Define `GGMLType` enumeration with all tensor types (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, I8, I16, I32)
  - Define custom exception classes (GGUFFileError, GGUFInvalidMagicError, GGUFVersionError, GGUFParseError, GGUFTruncatedError, GGUFInvalidTypeError)
  - Create `TYPE_SIZES` dictionary mapping tensor types to (type_size, block_size) tuples
  - Set up pytest and hypothesis for testing
  - _Requirements: 1.5, 2.7, 5.1, 5.2, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 2. Implement low-level binary reading utilities
  - [x] 2.1 Implement `_read_string()` method to read length-prefixed UTF-8 strings
    - Read uint64 length
    - Read length bytes as UTF-8
    - Handle truncated strings with GGUFTruncatedError
    - _Requirements: 2.1, 2.5, 3.1, 7.5, 7.6_
  
  - [x] 2.2 Write property test for string round-trip
    - **Property 3 (partial): Metadata Round-Trip**
    - **Validates: Requirements 2.1, 2.5**
  
  - [x] 2.3 Implement `_read_value()` method to read metadata values by type
    - Handle all primitive types using struct.unpack
    - Handle STRING type by calling _read_string()
    - Handle ARRAY type by calling _read_array()
    - Raise GGUFInvalidTypeError for invalid types
    - _Requirements: 2.2, 2.3, 2.5, 2.7_
  
  - [x] 2.4 Write property test for primitive type parsing
    - **Property 3 (partial): Metadata Round-Trip**
    - **Validates: Requirements 2.3**
  
  - [x] 2.5 Implement `_read_array()` method to read array metadata values
    - Read array element type (uint32)
    - Read array length (uint64)
    - Read all elements based on type
    - Validate element type is not ARRAY or invalid
    - _Requirements: 2.4, 2.6, 2.7_
  
  - [x] 2.6 Write property test for array parsing
    - **Property 3 (partial): Metadata Round-Trip**
    - **Validates: Requirements 2.4, 2.6**

- [ ] 3. Implement header parsing
  - [x] 3.1 Implement `_read_header()` method
    - Read and validate 4-byte magic number (raise GGUFInvalidMagicError if invalid)
    - Read version (uint32)
    - Read tensor_count (uint64)
    - Read metadata_kv_count (uint64)
    - Store in header dictionary
    - Handle truncated files with GGUFTruncatedError
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.2, 7.5_
  
  - [x] 3.2 Write property test for header parsing
    - **Property 1: Magic Number Validation**
    - **Property 2: Header Field Round-Trip**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
  
  - [x] 3.3 Write unit tests for header edge cases
    - Test invalid magic number error message
    - Test truncated header error
    - Test various version numbers
    - _Requirements: 1.5, 7.2_

- [ ] 4. Implement metadata parsing
  - [x] 4.1 Implement `_read_metadata()` method
    - Loop metadata_kv_count times
    - For each pair: read key string, read type, read value
    - Store in metadata dictionary
    - Handle errors with context (file position, key name)
    - _Requirements: 2.1, 2.2, 2.8_
  
  - [x] 4.2 Write property test for complete metadata parsing
    - **Property 3: Metadata Round-Trip**
    - **Property 4: Invalid Metadata Type Rejection**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8**
  
  - [x] 4.3 Write unit tests for metadata edge cases
    - Test empty metadata (count = 0)
    - Test nested arrays
    - Test all primitive types with specific values
    - Test error messages for invalid types
    - _Requirements: 2.7, 2.8_

- [ ] 5. Implement tensor info parsing
  - [x] 5.1 Implement `_read_tensor_info()` method
    - Loop tensor_count times
    - For each tensor: read name, n_dims, dimension sizes, type, offset
    - Store in tensor_info list as dictionaries
    - Validate tensor type is valid
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [x] 5.2 Write property test for tensor info parsing
    - **Property 5: Tensor Info Round-Trip**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
  
  - [x] 5.3 Write unit tests for tensor info edge cases
    - Test tensors with 1, 2, 3, and 4 dimensions
    - Test all tensor types
    - Test multiple tensors
    - _Requirements: 3.6_

- [ ] 6. Implement alignment and offset calculations
  - [x] 6.1 Implement alignment calculation logic
    - Get alignment from metadata (default 32 if not present)
    - Calculate padding needed: (alignment - (position % alignment)) % alignment
    - Calculate aligned base offset for tensor data section
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 6.2 Write property test for alignment calculation
    - **Property 6: Alignment Calculation**
    - **Property 7: Default Alignment**
    - **Validates: Requirements 4.1, 4.2, 4.3**
  
  - [x] 6.3 Implement `_calculate_tensor_offset()` method
    - Add tensor's offset to aligned base position
    - Return absolute file position
    - _Requirements: 4.4_
  
  - [x] 6.4 Write property test for tensor offset calculation
    - **Property 8: Tensor Offset Calculation**
    - **Validates: Requirements 4.4**

- [ ] 7. Implement tensor data reading
  - [x] 7.1 Implement `_calculate_tensor_size()` method
    - Get type_size and block_size from TYPE_SIZES
    - Calculate: type_size * (dims[0] / block_size) * dims[1] * dims[2] * dims[3]
    - Return size in bytes
    - _Requirements: 5.3_
  
  - [x] 7.2 Write property test for size calculation
    - **Property 9: Tensor Size Calculation**
    - **Validates: Requirements 5.3**
  
  - [x] 7.3 Implement `get_tensor_data()` method
    - Look up tensor info by name
    - Calculate absolute file offset
    - Calculate data size
    - Seek to offset and read size bytes
    - Return raw bytes
    - Cache data to avoid re-reading
    - _Requirements: 5.4, 5.5, 6.5_
  
  - [x] 7.4 Write property test for tensor data reading
    - **Property 10: Tensor Data Round-Trip**
    - **Validates: Requirements 5.1, 5.2, 5.4, 5.5**
  
  - [x] 7.5 Write unit tests for tensor data edge cases
    - Test all standard types (F32, F16, I8, I16, I32)
    - Test all quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
    - Test tensors with different dimension counts
    - _Requirements: 5.1, 5.2_

- [ ] 8. Implement high-level access interface
  - [x] 8.1 Implement GGUFReader class initialization and context manager
    - `__init__()` to store filepath
    - `__enter__()` to open file and parse header, metadata, tensor info
    - `__exit__()` to close file
    - Ensure file is closed even if exception occurs
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [x] 8.2 Implement metadata access methods
    - `get_metadata()` returns full metadata dictionary
    - `get_metadata_value(key)` returns specific value or raises KeyError
    - _Requirements: 6.1, 6.2_
  
  - [x] 8.3 Implement tensor access methods
    - `list_tensors()` returns list of tensor names
    - `get_tensor_info(name)` returns tensor info dictionary
    - `get_tensor_count()` returns number of tensors
    - _Requirements: 6.3, 6.4, 6.6_
  
  - [x] 8.4 Implement version access method
    - `get_version()` returns GGUF version from header
    - _Requirements: 6.7_
  
  - [x] 8.5 Write property tests for access interface
    - **Property 11: Metadata Retrieval**
    - **Property 12: Tensor Retrieval**
    - **Property 13: Count Retrieval**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
  
  - [x] 8.6 Write unit tests for context manager
    - Test file is opened in __enter__
    - Test file is closed in __exit__
    - Test file is closed when exception occurs
    - Test KeyError for non-existent metadata key
    - Test KeyError for non-existent tensor name
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 9. Implement comprehensive error handling
  - [x] 9.1 Add error handling throughout parsing code
    - Wrap file operations in try-except for IOError
    - Check file size before reading
    - Validate all type codes before use
    - Add descriptive error messages with file position
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  
  - [x] 9.2 Write property tests for error detection
    - **Property 14: Truncated File Detection**
    - **Property 15: Invalid Version Rejection**
    - **Property 16: Invalid Type Rejection**
    - **Property 17: Invalid String Length Rejection**
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.6**
  
  - [x] 9.3 Write unit tests for error messages
    - Test file not found error includes path
    - Test truncated file error includes position
    - Test invalid version error includes version number
    - Test invalid type error includes type code
    - Test invalid string length error
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

- [ ] 10. Create example usage and documentation
  - [x] 10.1 Create example script demonstrating usage
    - Show context manager usage
    - Show metadata access
    - Show tensor listing and data reading
    - Show error handling
  
  - [x] 10.2 Add docstrings to all public methods
    - Include parameter types and return types
    - Include usage examples
    - Document exceptions that can be raised

- [x] 11. Final checkpoint - Ensure all tests pass
  - Run all property-based tests with 100+ iterations
  - Run all unit tests
  - Verify test coverage is comprehensive
  - Ask the user if questions arise

## Notes

- All tests are required for comprehensive validation
- Each property test should run minimum 100 iterations
- Property tests use hypothesis library for random data generation
- All tests should be tagged with feature name and property number
- Implementation builds incrementally from low-level utilities to high-level interface
- Error handling is integrated throughout, with comprehensive testing in task 9
