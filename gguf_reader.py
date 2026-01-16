"""
GGUF Reader - A Python library for parsing GGUF (GPT-Generated Unified Format) files.

This module provides functionality to read and parse GGUF binary files used for
storing large language models with their metadata and tensor data.
"""

import struct
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Type Enumerations
# ============================================================================

class GGUFValueType:
    """Metadata value types in GGUF format."""
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
    """Tensor data types in GGML/GGUF format."""
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


# ============================================================================
# Type Size Mapping
# ============================================================================

# Maps tensor types to (type_size, block_size) tuples
# type_size: bytes per block
# block_size: number of elements per block
TYPE_SIZES: Dict[int, Tuple[int, int]] = {
    # Standard types
    GGMLType.F32: (4, 1),      # 4 bytes per element, 1 element per block
    GGMLType.F16: (2, 1),      # 2 bytes per element, 1 element per block
    GGMLType.I8: (1, 1),       # 1 byte per element, 1 element per block
    GGMLType.I16: (2, 1),      # 2 bytes per element, 1 element per block
    GGMLType.I32: (4, 1),      # 4 bytes per element, 1 element per block
    
    # Quantized types
    GGMLType.Q4_0: (18, 32),   # 18 bytes per 32-element block
    GGMLType.Q4_1: (20, 32),   # 20 bytes per 32-element block
    GGMLType.Q5_0: (22, 32),   # 22 bytes per 32-element block
    GGMLType.Q5_1: (24, 32),   # 24 bytes per 32-element block
    GGMLType.Q8_0: (34, 32),   # 34 bytes per 32-element block
    GGMLType.Q8_1: (40, 32),   # 40 bytes per 32-element block
    GGMLType.Q2_K: (82, 256),  # 82 bytes per 256-element block
    GGMLType.Q3_K: (110, 256), # 110 bytes per 256-element block
    GGMLType.Q4_K: (144, 256), # 144 bytes per 256-element block
    GGMLType.Q5_K: (176, 256), # 176 bytes per 256-element block
    GGMLType.Q6_K: (210, 256), # 210 bytes per 256-element block
    GGMLType.Q8_K: (292, 256), # 292 bytes per 256-element block
}


# ============================================================================
# Exception Classes
# ============================================================================

class GGUFFileError(Exception):
    """Base exception for all GGUF-related errors."""
    pass


class GGUFInvalidMagicError(GGUFFileError):
    """Raised when the magic number doesn't match GGUF format."""
    pass


class GGUFVersionError(GGUFFileError):
    """Raised when an unsupported GGUF version is encountered."""
    pass


class GGUFParseError(GGUFFileError):
    """Raised when a generic parsing error occurs."""
    pass


class GGUFTruncatedError(GGUFFileError):
    """Raised when the file ends unexpectedly."""
    pass


class GGUFInvalidTypeError(GGUFFileError):
    """Raised when an invalid type code is encountered."""
    pass


# ============================================================================
# GGUFReader Class
# ============================================================================

class GGUFReader:
    """
    Reader for GGUF (GPT-Generated Unified Format) files.
    
    This class provides methods to parse GGUF binary files and access their
    metadata, tensor information, and tensor data.
    
    Usage:
        with GGUFReader('model.gguf') as reader:
            metadata = reader.get_metadata()
            tensors = reader.list_tensors()
            data = reader.get_tensor_data('token_embd.weight')
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the GGUF reader with a file path.
        
        Args:
            filepath: Path to the GGUF file to read
        """
        self.filepath = filepath
        self.file = None
        self.header: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.tensor_info: List[Dict[str, Any]] = []
        self.tensor_data_base: int = 0
        self.tensor_data_cache: Dict[str, bytes] = {}
    
    def __enter__(self):
        """
        Context manager entry - opens and parses the file.
        
        This method:
        1. Opens the file in binary read mode
        2. Parses the header
        3. Parses all metadata
        4. Parses all tensor information
        5. Calculates alignment for tensor data section
        
        Returns:
            self for use in with statement
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            GGUFInvalidMagicError: If the file is not a valid GGUF file
            GGUFTruncatedError: If the file is truncated
            GGUFParseError: If parsing fails for any reason
        """
        try:
            # Open the file in binary read mode
            self.file = open(self.filepath, 'rb')
            
            # Parse the file structure
            self._read_header()
            self._read_metadata()
            self._read_tensor_info()
            self._calculate_alignment()
            
            return self
            
        except FileNotFoundError:
            # Re-raise with more context
            raise GGUFFileError(f"File not found: '{self.filepath}'")
        except Exception:
            # If any error occurs, ensure file is closed before re-raising
            if self.file is not None:
                self.file.close()
                self.file = None
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - closes the file.
        
        This method ensures the file is properly closed even if an exception
        occurs within the with block. It does not suppress exceptions.
        
        Args:
            exc_type: Exception type (if an exception occurred)
            exc_val: Exception value (if an exception occurred)
            exc_tb: Exception traceback (if an exception occurred)
            
        Returns:
            None (does not suppress exceptions)
        """
        if self.file is not None:
            self.file.close()
            self.file = None
        # Return None to propagate any exception that occurred
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return all metadata as a dictionary.
        
        Returns:
            Dictionary containing all metadata key-value pairs
        """
        # Implementation will be added in task 8
        return self.metadata
    
    def get_metadata_value(self, key: str) -> Any:
        """
        Return a specific metadata value by key.
        
        Args:
            key: The metadata key to retrieve
            
        Returns:
            The metadata value associated with the key
            
        Raises:
            KeyError: If the key doesn't exist in metadata
        """
        # Implementation will be added in task 8
        return self.metadata[key]
    
    def list_tensors(self) -> List[str]:
        """
        Return a list of all tensor names.
        
        Returns:
            List of tensor names as strings
        """
        # Implementation will be added in task 8
        return [info['name'] for info in self.tensor_info]
    
    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """
        Return tensor metadata (shape, type, offset) by name.
        
        Args:
            name: The tensor name to retrieve
            
        Returns:
            Dictionary containing tensor information
            
        Raises:
            KeyError: If the tensor name doesn't exist
        """
        # Implementation will be added in task 8
        for info in self.tensor_info:
            if info['name'] == name:
                return info
        raise KeyError(f"Tensor '{name}' not found")
    
    def get_tensor_data(self, name: str) -> bytes:
        """
        Return raw tensor data as bytes.
        
        This method retrieves the raw binary data for a tensor by name.
        The data is cached after the first read to avoid re-reading from disk.
        
        Args:
            name: The tensor name to retrieve data for
            
        Returns:
            Raw tensor data as bytes
            
        Raises:
            KeyError: If the tensor name doesn't exist
            GGUFParseError: If file is not open or reading fails
            GGUFTruncatedError: If the file ends before all tensor data is read
        """
        # Check if data is already cached
        if name in self.tensor_data_cache:
            return self.tensor_data_cache[name]
        
        # Ensure file is open
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Look up tensor info by name
        tensor_info = None
        for info in self.tensor_info:
            if info['name'] == name:
                tensor_info = info
                break
        
        if tensor_info is None:
            raise KeyError(f"Tensor '{name}' not found")
        
        # Calculate absolute file offset
        absolute_offset = self._calculate_tensor_offset(tensor_info)
        
        # Calculate data size
        data_size = self._calculate_tensor_size(tensor_info)
        
        # Seek to offset
        self.file.seek(absolute_offset)
        
        # Read size bytes
        data = self.file.read(data_size)
        
        # Verify we read the expected amount
        if len(data) < data_size:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {absolute_offset}: "
                f"expected to read {data_size} bytes for tensor '{name}', "
                f"only {len(data)} bytes available"
            )
        
        # Cache the data
        self.tensor_data_cache[name] = data
        
        return data
    
    def get_tensor_count(self) -> int:
        """
        Return the total number of tensors.
        
        Returns:
            Number of tensors in the file
        """
        # Implementation will be added in task 8
        return len(self.tensor_info)
    
    def get_version(self) -> int:
        """
        Return the GGUF format version.
        
        Returns:
            GGUF version number
        """
        # Implementation will be added in task 8
        return self.header.get('version', 0)
    
    # ========================================================================
    # Internal Parsing Methods
    # ========================================================================
    
    def _read_header(self) -> None:
        """
        Read and validate the GGUF file header.
        
        The GGUF header consists of:
        - 4 bytes: magic number (0x46554747 = "GGUF" in ASCII)
        - 4 bytes: version (uint32)
        - 8 bytes: tensor_count (uint64)
        - 8 bytes: metadata_kv_count (uint64)
        
        The parsed values are stored in self.header dictionary.
        
        Raises:
            GGUFInvalidMagicError: If the magic number doesn't match GGUF format
            GGUFTruncatedError: If the file ends before the header is fully read
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Expected GGUF magic number: 0x46554747 ("GGUF" in ASCII, little-endian)
        GGUF_MAGIC = 0x46554747
        
        position = self.file.tell()
        
        # Read magic number (4 bytes, uint32, little-endian)
        magic_bytes = self.file.read(4)
        if len(magic_bytes) < 4:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position}: "
                f"expected to read 4 bytes for magic number, only {len(magic_bytes)} bytes available"
            )
        
        magic = struct.unpack('<I', magic_bytes)[0]
        
        # Validate magic number
        if magic != GGUF_MAGIC:
            raise GGUFInvalidMagicError(
                f"Invalid GGUF magic number in file '{self.filepath}' at position {position}: "
                f"expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )
        
        # Read version (4 bytes, uint32, little-endian)
        version_bytes = self.file.read(4)
        if len(version_bytes) < 4:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position + 4}: "
                f"expected to read 4 bytes for version, only {len(version_bytes)} bytes available"
            )
        
        version = struct.unpack('<I', version_bytes)[0]
        
        # Read tensor_count (8 bytes, uint64, little-endian)
        tensor_count_bytes = self.file.read(8)
        if len(tensor_count_bytes) < 8:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position + 8}: "
                f"expected to read 8 bytes for tensor_count, only {len(tensor_count_bytes)} bytes available"
            )
        
        tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
        
        # Read metadata_kv_count (8 bytes, uint64, little-endian)
        metadata_kv_count_bytes = self.file.read(8)
        if len(metadata_kv_count_bytes) < 8:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position + 16}: "
                f"expected to read 8 bytes for metadata_kv_count, only {len(metadata_kv_count_bytes)} bytes available"
            )
        
        metadata_kv_count = struct.unpack('<Q', metadata_kv_count_bytes)[0]
        
        # Store header information
        self.header = {
            'magic': magic,
            'version': version,
            'tensor_count': tensor_count,
            'metadata_kv_count': metadata_kv_count
        }
    
    def _read_string(self) -> str:
        """
        Read a length-prefixed UTF-8 string from the file.
        
        GGUF strings are encoded as:
        - uint64: length of the string in bytes
        - bytes: UTF-8 encoded string data
        
        Returns:
            The decoded UTF-8 string
            
        Raises:
            GGUFTruncatedError: If the file ends before the string is fully read
            GGUFParseError: If the string length is invalid or unreasonably large
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Get current position for error reporting
        position = self.file.tell()
        
        # Read the string length (uint64, 8 bytes, little-endian)
        length_bytes = self.file.read(8)
        if len(length_bytes) < 8:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position}: "
                f"expected to read 8 bytes for string length, only {len(length_bytes)} bytes available"
            )
        
        # Unpack the length
        length = struct.unpack('<Q', length_bytes)[0]
        
        # Validate string length is reasonable (prevent memory exhaustion attacks)
        # Maximum reasonable string length: 100MB
        MAX_STRING_LENGTH = 100 * 1024 * 1024
        if length > MAX_STRING_LENGTH:
            raise GGUFParseError(
                f"Invalid string length in file '{self.filepath}' at position {position}: "
                f"length {length} exceeds maximum allowed length {MAX_STRING_LENGTH}"
            )
        
        # Read the string data
        if length == 0:
            return ""
        
        string_bytes = self.file.read(length)
        if len(string_bytes) < length:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position + 8}: "
                f"expected to read {length} bytes for string data, only {len(string_bytes)} bytes available"
            )
        
        # Decode as UTF-8
        try:
            return string_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise GGUFParseError(
                f"Invalid UTF-8 string in file '{self.filepath}' at position {position}: {e}"
            )

    def _read_value(self, value_type: int) -> Any:
        """
        Read a single metadata value based on its type.
        
        This method handles all GGUF metadata value types including primitives,
        strings, and arrays. It uses struct.unpack for binary data and delegates
        to specialized methods for complex types.
        
        Args:
            value_type: The GGUF value type code (from GGUFValueType)
            
        Returns:
            The parsed value in appropriate Python type:
            - Integers for UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64
            - Float for FLOAT32, FLOAT64
            - Bool for BOOL
            - String for STRING
            - List for ARRAY
            
        Raises:
            GGUFInvalidTypeError: If the value_type is not a valid GGUF type
            GGUFTruncatedError: If the file ends before the value is fully read
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        position = self.file.tell()
        
        # Handle primitive types with struct.unpack
        # Format: (struct_format, size_in_bytes)
        type_formats = {
            GGUFValueType.UINT8: ('<B', 1),    # unsigned char
            GGUFValueType.INT8: ('<b', 1),     # signed char
            GGUFValueType.UINT16: ('<H', 2),   # unsigned short
            GGUFValueType.INT16: ('<h', 2),    # signed short
            GGUFValueType.UINT32: ('<I', 4),   # unsigned int
            GGUFValueType.INT32: ('<i', 4),    # signed int
            GGUFValueType.UINT64: ('<Q', 8),   # unsigned long long
            GGUFValueType.INT64: ('<q', 8),    # signed long long
            GGUFValueType.FLOAT32: ('<f', 4),  # float
            GGUFValueType.FLOAT64: ('<d', 8),  # double
            GGUFValueType.BOOL: ('<B', 1),     # bool as unsigned char
        }
        
        # Handle primitive types
        if value_type in type_formats:
            fmt, size = type_formats[value_type]
            data = self.file.read(size)
            if len(data) < size:
                raise GGUFTruncatedError(
                    f"Unexpected end of file '{self.filepath}' at position {position}: "
                    f"expected to read {size} bytes for type {value_type}, only {len(data)} bytes available"
                )
            value = struct.unpack(fmt, data)[0]
            
            # Convert BOOL type to Python bool
            if value_type == GGUFValueType.BOOL:
                return bool(value)
            return value
        
        # Handle STRING type
        elif value_type == GGUFValueType.STRING:
            return self._read_string()
        
        # Handle ARRAY type
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array()
        
        # Invalid type
        else:
            raise GGUFInvalidTypeError(
                f"Invalid metadata type in file '{self.filepath}' at position {position}: "
                f"type code {value_type} is not a valid GGUF type"
            )

    def _read_array(self) -> List[Any]:
        """
        Read an array metadata value from the file.
        
        GGUF arrays are encoded as:
        - uint32: element type code
        - uint64: number of elements
        - elements: array of values of the specified type
        
        Returns:
            List containing all array elements
            
        Raises:
            GGUFInvalidTypeError: If the element type is invalid or is ARRAY (nested arrays not allowed)
            GGUFTruncatedError: If the file ends before the array is fully read
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        position = self.file.tell()
        
        # Read element type (uint32, 4 bytes)
        type_bytes = self.file.read(4)
        if len(type_bytes) < 4:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position}: "
                f"expected to read 4 bytes for array element type, only {len(type_bytes)} bytes available"
            )
        element_type = struct.unpack('<I', type_bytes)[0]
        
        # Validate element type is not ARRAY (nested arrays not allowed)
        if element_type == GGUFValueType.ARRAY:
            raise GGUFInvalidTypeError(
                f"Invalid array element type in file '{self.filepath}' at position {position}: "
                f"nested arrays are not allowed"
            )
        
        # Read array length (uint64, 8 bytes)
        length_bytes = self.file.read(8)
        if len(length_bytes) < 8:
            raise GGUFTruncatedError(
                f"Unexpected end of file '{self.filepath}' at position {position + 4}: "
                f"expected to read 8 bytes for array length, only {len(length_bytes)} bytes available"
            )
        length = struct.unpack('<Q', length_bytes)[0]
        
        # Read all elements
        result = []
        for i in range(length):
            try:
                value = self._read_value(element_type)
                result.append(value)
            except GGUFInvalidTypeError:
                # Re-raise with more context
                raise GGUFInvalidTypeError(
                    f"Invalid array element type in file '{self.filepath}' at position {position}: "
                    f"type code {element_type} is not a valid GGUF type"
                )
        
        return result

    def _read_metadata(self) -> None:
        """
        Parse all metadata key-value pairs from the file.
        
        This method reads the metadata section of the GGUF file, which consists
        of metadata_kv_count key-value pairs. Each pair is encoded as:
        - key: length-prefixed UTF-8 string
        - value_type: uint32 type code
        - value: data based on the type
        
        The parsed metadata is stored in self.metadata dictionary.
        
        Raises:
            GGUFParseError: If file is not open or parsing fails
            GGUFTruncatedError: If the file ends before all metadata is read
            GGUFInvalidTypeError: If an invalid type code is encountered
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Get the number of metadata key-value pairs from the header
        metadata_kv_count = self.header.get('metadata_kv_count', 0)
        
        # Clear any existing metadata
        self.metadata = {}
        
        # Read each key-value pair
        for i in range(metadata_kv_count):
            position = self.file.tell()
            key = None
            
            try:
                # Read the key (length-prefixed string)
                key = self._read_string()
                
                # Read the value type (uint32, 4 bytes)
                type_bytes = self.file.read(4)
                if len(type_bytes) < 4:
                    raise GGUFTruncatedError(
                        f"Unexpected end of file '{self.filepath}' at position {self.file.tell() - len(type_bytes)}: "
                        f"expected to read 4 bytes for metadata value type (key: '{key}'), "
                        f"only {len(type_bytes)} bytes available"
                    )
                
                value_type = struct.unpack('<I', type_bytes)[0]
                
                # Read the value based on its type
                value = self._read_value(value_type)
                
                # Store in metadata dictionary
                self.metadata[key] = value
                
            except GGUFTruncatedError:
                # Re-raise truncation errors with additional context
                raise
            except GGUFInvalidTypeError as e:
                # Add context about which metadata key failed
                if key:
                    raise GGUFInvalidTypeError(
                        f"Invalid type for metadata key '{key}' in file '{self.filepath}' at position {position}: {e}"
                    ) from e
                else:
                    raise
            except Exception as e:
                # Wrap other exceptions with context
                if key:
                    raise GGUFParseError(
                        f"Error parsing metadata key '{key}' in file '{self.filepath}' at position {position}: {e}"
                    ) from e
                else:
                    raise GGUFParseError(
                        f"Error parsing metadata in file '{self.filepath}' at position {position}: {e}"
                    ) from e

    def _read_tensor_info(self) -> None:
        """
        Parse all tensor information records from the file.
        
        This method reads the tensor info section of the GGUF file, which consists
        of tensor_count tensor information records. Each record is encoded as:
        - name: length-prefixed UTF-8 string
        - n_dims: uint32 number of dimensions
        - dims: array of uint64 dimension sizes (length = n_dims)
        - type: uint32 tensor data type code
        - offset: uint64 offset from tensor data section start
        
        The parsed tensor information is stored in self.tensor_info list.
        
        Raises:
            GGUFParseError: If file is not open or parsing fails
            GGUFTruncatedError: If the file ends before all tensor info is read
            GGUFInvalidTypeError: If an invalid tensor type code is encountered
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Get the number of tensors from the header
        tensor_count = self.header.get('tensor_count', 0)
        
        # Clear any existing tensor info
        self.tensor_info = []
        
        # Read each tensor information record
        for i in range(tensor_count):
            position = self.file.tell()
            tensor_name = None
            
            try:
                # Read the tensor name (length-prefixed string)
                tensor_name = self._read_string()
                
                # Read the number of dimensions (uint32, 4 bytes)
                n_dims_bytes = self.file.read(4)
                if len(n_dims_bytes) < 4:
                    raise GGUFTruncatedError(
                        f"Unexpected end of file '{self.filepath}' at position {self.file.tell() - len(n_dims_bytes)}: "
                        f"expected to read 4 bytes for n_dims (tensor: '{tensor_name}'), "
                        f"only {len(n_dims_bytes)} bytes available"
                    )
                
                n_dims = struct.unpack('<I', n_dims_bytes)[0]
                
                # Read the dimension sizes (array of uint64, 8 bytes each)
                dims = []
                for dim_idx in range(n_dims):
                    dim_bytes = self.file.read(8)
                    if len(dim_bytes) < 8:
                        raise GGUFTruncatedError(
                            f"Unexpected end of file '{self.filepath}' at position {self.file.tell() - len(dim_bytes)}: "
                            f"expected to read 8 bytes for dimension {dim_idx} (tensor: '{tensor_name}'), "
                            f"only {len(dim_bytes)} bytes available"
                        )
                    dim_size = struct.unpack('<Q', dim_bytes)[0]
                    dims.append(dim_size)
                
                # Read the tensor data type (uint32, 4 bytes)
                type_bytes = self.file.read(4)
                if len(type_bytes) < 4:
                    raise GGUFTruncatedError(
                        f"Unexpected end of file '{self.filepath}' at position {self.file.tell() - len(type_bytes)}: "
                        f"expected to read 4 bytes for tensor type (tensor: '{tensor_name}'), "
                        f"only {len(type_bytes)} bytes available"
                    )
                
                tensor_type = struct.unpack('<I', type_bytes)[0]
                
                # Validate tensor type is valid
                if tensor_type not in TYPE_SIZES:
                    raise GGUFInvalidTypeError(
                        f"Invalid tensor type in file '{self.filepath}' at position {position}: "
                        f"type code {tensor_type} is not a valid GGML tensor type (tensor: '{tensor_name}')"
                    )
                
                # Read the offset (uint64, 8 bytes)
                offset_bytes = self.file.read(8)
                if len(offset_bytes) < 8:
                    raise GGUFTruncatedError(
                        f"Unexpected end of file '{self.filepath}' at position {self.file.tell() - len(offset_bytes)}: "
                        f"expected to read 8 bytes for offset (tensor: '{tensor_name}'), "
                        f"only {len(offset_bytes)} bytes available"
                    )
                
                offset = struct.unpack('<Q', offset_bytes)[0]
                
                # Store tensor information
                tensor_info = {
                    'name': tensor_name,
                    'n_dims': n_dims,
                    'dims': dims,
                    'type': tensor_type,
                    'offset': offset
                }
                self.tensor_info.append(tensor_info)
                
            except GGUFTruncatedError:
                # Re-raise truncation errors with additional context
                raise
            except GGUFInvalidTypeError:
                # Re-raise type errors (already have context)
                raise
            except Exception as e:
                # Wrap other exceptions with context
                if tensor_name:
                    raise GGUFParseError(
                        f"Error parsing tensor info for '{tensor_name}' in file '{self.filepath}' at position {position}: {e}"
                    ) from e
                else:
                    raise GGUFParseError(
                        f"Error parsing tensor info in file '{self.filepath}' at position {position}: {e}"
                    ) from e

    def _calculate_alignment(self) -> None:
        """
        Calculate the aligned base offset for the tensor data section.
        
        After parsing all tensor information, the file position needs to be aligned
        to a specific byte boundary before tensor data begins. This method:
        1. Gets the alignment value from metadata (default 32 if not present)
        2. Calculates padding needed to reach the next aligned position
        3. Stores the aligned base offset in self.tensor_data_base
        
        The alignment calculation uses the formula:
        padding = (alignment - (position % alignment)) % alignment
        
        This ensures the tensor data section starts at a position that is a
        multiple of the alignment value.
        
        Raises:
            GGUFParseError: If file is not open
        """
        if self.file is None:
            raise GGUFParseError("File is not open")
        
        # Get alignment from metadata, default to 32 bytes if not present
        alignment = self.metadata.get('general.alignment', 32)
        
        # Get current file position (end of tensor info section)
        current_position = self.file.tell()
        
        # Calculate padding needed to reach next aligned position
        # Formula: (alignment - (position % alignment)) % alignment
        # The outer modulo handles the case where position is already aligned (returns 0)
        padding = (alignment - (current_position % alignment)) % alignment
        
        # Calculate aligned base offset for tensor data section
        self.tensor_data_base = current_position + padding

    def _calculate_tensor_offset(self, tensor_info: Dict[str, Any]) -> int:
        """
        Calculate the absolute file offset for a tensor's data.
        
        Each tensor has a relative offset stored in its tensor_info that indicates
        where its data begins relative to the start of the tensor data section.
        This method calculates the absolute file position by adding the tensor's
        relative offset to the aligned base position of the tensor data section.
        
        Args:
            tensor_info: Dictionary containing tensor information, must include 'offset' key
            
        Returns:
            Absolute file position (in bytes) where the tensor's data begins
            
        Raises:
            GGUFParseError: If tensor_info doesn't contain 'offset' key
        """
        if 'offset' not in tensor_info:
            raise GGUFParseError(
                f"Tensor info missing 'offset' key for tensor '{tensor_info.get('name', 'unknown')}'"
            )
        
        # Add tensor's relative offset to the aligned base position
        absolute_offset = self.tensor_data_base + tensor_info['offset']
        
        return absolute_offset

    def _calculate_tensor_size(self, tensor_info: Dict[str, Any]) -> int:
        """
        Calculate the size in bytes of a tensor's data.
        
        The size calculation depends on the tensor's type and dimensions.
        For quantized types, elements are grouped into blocks, so the calculation
        accounts for the block size.
        
        Formula: type_size * (dims[0] / block_size) * dims[1] * dims[2] * dims[3]
        
        Where:
        - type_size: bytes per block (from TYPE_SIZES)
        - block_size: number of elements per block (from TYPE_SIZES)
        - dims: dimension sizes [d0, d1, d2, d3]
        
        Args:
            tensor_info: Dictionary containing tensor information, must include 'type' and 'dims' keys
            
        Returns:
            Size of tensor data in bytes
            
        Raises:
            GGUFParseError: If tensor_info is missing required keys or has invalid type
        """
        # Validate tensor_info has required keys
        if 'type' not in tensor_info:
            raise GGUFParseError(
                f"Tensor info missing 'type' key for tensor '{tensor_info.get('name', 'unknown')}'"
            )
        if 'dims' not in tensor_info:
            raise GGUFParseError(
                f"Tensor info missing 'dims' key for tensor '{tensor_info.get('name', 'unknown')}'"
            )
        
        tensor_type = tensor_info['type']
        dims = tensor_info['dims']
        
        # Get type_size and block_size from TYPE_SIZES
        if tensor_type not in TYPE_SIZES:
            raise GGUFParseError(
                f"Invalid tensor type {tensor_type} for tensor '{tensor_info.get('name', 'unknown')}'"
            )
        
        type_size, block_size = TYPE_SIZES[tensor_type]
        
        # Ensure dims has at least 1 dimension, pad with 1s if needed
        # GGUF tensors can have 1-4 dimensions, we treat missing dimensions as 1
        padded_dims = dims + [1] * (4 - len(dims))
        
        # Calculate size: type_size * (dims[0] / block_size) * dims[1] * dims[2] * dims[3]
        # Note: dims[0] / block_size gives the number of blocks in the first dimension
        size = type_size * (padded_dims[0] // block_size) * padded_dims[1] * padded_dims[2] * padded_dims[3]
        
        return size
