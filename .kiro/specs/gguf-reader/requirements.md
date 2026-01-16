# Requirements Document

## Introduction

This document specifies the requirements for a Python script that can read and parse GGUF (GPT-Generated Unified Format) files. GGUF is a binary file format designed for storing large language models with their metadata and tensor data. The format was created by Georgi Gerganov for the llama.cpp project and is widely used in the open-source LLM community.

## Glossary

- **GGUF**: GPT-Generated Unified Format, a binary file format for storing machine learning models
- **Reader**: The Python script that parses GGUF files
- **Header**: The initial section of a GGUF file containing magic number, version, and counts
- **Metadata**: Key-value pairs containing model information (architecture, parameters, etc.)
- **Tensor_Info**: Metadata about each tensor including name, dimensions, type, and offset
- **Tensor_Data**: The actual numerical data stored in tensors
- **Quantization**: Compression technique that reduces precision of model weights
- **Alignment**: Byte boundary alignment for tensor data storage

## Requirements

### Requirement 1: Parse GGUF File Header

**User Story:** As a developer, I want to parse the GGUF file header, so that I can validate the file format and understand its structure.

#### Acceptance Criteria

1. WHEN a GGUF file is provided, THE Reader SHALL read the magic number (4 bytes) and verify it matches the GGUF format identifier
2. WHEN reading the header, THE Reader SHALL extract the version number (uint32)
3. WHEN reading the header, THE Reader SHALL extract the tensor count (uint64)
4. WHEN reading the header, THE Reader SHALL extract the metadata key-value pair count (uint64)
5. IF the magic number is invalid, THEN THE Reader SHALL raise an error indicating an invalid GGUF file

### Requirement 2: Parse Metadata Key-Value Pairs

**User Story:** As a developer, I want to extract all metadata from the GGUF file, so that I can access model configuration and parameters.

#### Acceptance Criteria

1. WHEN parsing metadata, THE Reader SHALL read each key as a length-prefixed string (uint64 length + UTF-8 bytes)
2. WHEN parsing metadata, THE Reader SHALL read the value type (uint32 enum)
3. WHEN parsing metadata values, THE Reader SHALL support all primitive types (UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT32, FLOAT64, BOOL, STRING)
4. WHEN parsing metadata values, THE Reader SHALL support array types containing any valid primitive type
5. WHEN encountering a STRING type, THE Reader SHALL read it as a length-prefixed UTF-8 string
6. WHEN encountering an ARRAY type, THE Reader SHALL read the element type, array length, and all elements
7. IF an invalid type is encountered, THEN THE Reader SHALL raise an error
8. WHEN parsing is complete, THE Reader SHALL store all metadata in an accessible dictionary structure

### Requirement 3: Parse Tensor Information

**User Story:** As a developer, I want to extract tensor metadata, so that I can understand the model structure and locate tensor data.

#### Acceptance Criteria

1. WHEN parsing tensor info, THE Reader SHALL read the tensor name as a length-prefixed string
2. WHEN parsing tensor info, THE Reader SHALL read the number of dimensions (uint32)
3. WHEN parsing tensor info, THE Reader SHALL read the dimension sizes as an array of uint64 values
4. WHEN parsing tensor info, THE Reader SHALL read the tensor data type (uint32 enum)
5. WHEN parsing tensor info, THE Reader SHALL read the data offset (uint64) indicating where tensor data begins
6. WHEN all tensor info is parsed, THE Reader SHALL store it in an accessible list structure

### Requirement 4: Handle Tensor Data Alignment

**User Story:** As a developer, I want the reader to correctly handle data alignment, so that tensor data is read from the correct file positions.

#### Acceptance Criteria

1. WHEN the metadata contains a "general.alignment" key, THE Reader SHALL use that value for alignment calculations
2. WHEN no alignment is specified, THE Reader SHALL use a default alignment of 32 bytes
3. WHEN calculating tensor data positions, THE Reader SHALL apply alignment padding after the tensor info section
4. WHEN calculating each tensor's position, THE Reader SHALL add the tensor's offset to the aligned base position

### Requirement 5: Read Tensor Data

**User Story:** As a developer, I want to read the actual tensor data, so that I can access model weights and parameters.

#### Acceptance Criteria

1. WHEN reading tensor data, THE Reader SHALL support all standard types (F32, F16, I8, I16, I32)
2. WHEN reading tensor data, THE Reader SHALL support all quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
3. WHEN calculating data size, THE Reader SHALL use the formula: type_size * (ne[0] / block_size) * ne[1] * ne[2] * ne[3]
4. WHEN reading tensor data, THE Reader SHALL read the calculated number of bytes from the file at the correct offset
5. WHEN all tensors are read, THE Reader SHALL store the raw binary data for each tensor

### Requirement 6: Provide Data Access Interface

**User Story:** As a developer, I want a clean interface to access parsed data, so that I can easily work with GGUF file contents.

#### Acceptance Criteria

1. THE Reader SHALL provide a method to retrieve all metadata as a dictionary
2. THE Reader SHALL provide a method to retrieve a specific metadata value by key
3. THE Reader SHALL provide a method to list all tensor names
4. THE Reader SHALL provide a method to retrieve tensor information by name
5. THE Reader SHALL provide a method to retrieve tensor data by name
6. THE Reader SHALL provide a method to get the total number of tensors
7. THE Reader SHALL provide a method to get the GGUF version

### Requirement 7: Handle Errors Gracefully

**User Story:** As a developer, I want clear error messages when parsing fails, so that I can diagnose issues with GGUF files.

#### Acceptance Criteria

1. WHEN a file cannot be opened, THE Reader SHALL raise an error with the file path
2. WHEN the file is too short to contain a valid header, THE Reader SHALL raise an error indicating truncated file
3. WHEN an unsupported version is encountered, THE Reader SHALL raise an error with the version number
4. WHEN an invalid data type is encountered, THE Reader SHALL raise an error with the type value and location
5. WHEN reading past the end of file, THE Reader SHALL raise an error indicating unexpected EOF
6. WHEN an invalid string length is encountered, THE Reader SHALL raise an error

### Requirement 8: Support Context Manager Protocol

**User Story:** As a developer, I want to use the reader with Python's context manager, so that file resources are properly managed.

#### Acceptance Criteria

1. THE Reader SHALL implement the context manager protocol (__enter__ and __exit__ methods)
2. WHEN used in a with statement, THE Reader SHALL automatically open the file
3. WHEN exiting the context, THE Reader SHALL automatically close the file
4. WHEN an exception occurs within the context, THE Reader SHALL ensure the file is closed before propagating the exception
