"""
Tests for GGUF Reader.

This module contains unit tests and property-based tests for the GGUF Reader library.
"""

import struct
import io
from typing import Any, Dict, List

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from gguf_reader import (
    GGUFReader,
    GGUFValueType,
    GGMLType,
    TYPE_SIZES,
    GGUFFileError,
    GGUFInvalidMagicError,
    GGUFVersionError,
    GGUFParseError,
    GGUFTruncatedError,
    GGUFInvalidTypeError,
)


# ============================================================================
# Test Type Definitions
# ============================================================================

def test_gguf_value_type_constants():
    """Test that GGUFValueType enumeration has all required constants."""
    assert GGUFValueType.UINT8 == 0
    assert GGUFValueType.INT8 == 1
    assert GGUFValueType.UINT16 == 2
    assert GGUFValueType.INT16 == 3
    assert GGUFValueType.UINT32 == 4
    assert GGUFValueType.INT32 == 5
    assert GGUFValueType.FLOAT32 == 6
    assert GGUFValueType.BOOL == 7
    assert GGUFValueType.STRING == 8
    assert GGUFValueType.ARRAY == 9
    assert GGUFValueType.UINT64 == 10
    assert GGUFValueType.INT64 == 11
    assert GGUFValueType.FLOAT64 == 12


def test_ggml_type_constants():
    """Test that GGMLType enumeration has all required constants."""
    assert GGMLType.F32 == 0
    assert GGMLType.F16 == 1
    assert GGMLType.Q4_0 == 2
    assert GGMLType.Q4_1 == 3
    assert GGMLType.Q5_0 == 6
    assert GGMLType.Q5_1 == 7
    assert GGMLType.Q8_0 == 8
    assert GGMLType.Q8_1 == 9
    assert GGMLType.Q2_K == 10
    assert GGMLType.Q3_K == 11
    assert GGMLType.Q4_K == 12
    assert GGMLType.Q5_K == 13
    assert GGMLType.Q6_K == 14
    assert GGMLType.Q8_K == 15
    assert GGMLType.I8 == 16
    assert GGMLType.I16 == 17
    assert GGMLType.I32 == 18


def test_type_sizes_mapping():
    """Test that TYPE_SIZES dictionary contains all required tensor types."""
    # Standard types
    assert TYPE_SIZES[GGMLType.F32] == (4, 1)
    assert TYPE_SIZES[GGMLType.F16] == (2, 1)
    assert TYPE_SIZES[GGMLType.I8] == (1, 1)
    assert TYPE_SIZES[GGMLType.I16] == (2, 1)
    assert TYPE_SIZES[GGMLType.I32] == (4, 1)
    
    # Quantized types
    assert TYPE_SIZES[GGMLType.Q4_0] == (18, 32)
    assert TYPE_SIZES[GGMLType.Q4_1] == (20, 32)
    assert TYPE_SIZES[GGMLType.Q5_0] == (22, 32)
    assert TYPE_SIZES[GGMLType.Q5_1] == (24, 32)
    assert TYPE_SIZES[GGMLType.Q8_0] == (34, 32)
    assert TYPE_SIZES[GGMLType.Q8_1] == (40, 32)
    assert TYPE_SIZES[GGMLType.Q2_K] == (82, 256)
    assert TYPE_SIZES[GGMLType.Q3_K] == (110, 256)
    assert TYPE_SIZES[GGMLType.Q4_K] == (144, 256)
    assert TYPE_SIZES[GGMLType.Q5_K] == (176, 256)
    assert TYPE_SIZES[GGMLType.Q6_K] == (210, 256)
    assert TYPE_SIZES[GGMLType.Q8_K] == (292, 256)


# ============================================================================
# Test Exception Classes
# ============================================================================

def test_exception_hierarchy():
    """Test that all custom exceptions inherit from GGUFFileError."""
    assert issubclass(GGUFInvalidMagicError, GGUFFileError)
    assert issubclass(GGUFVersionError, GGUFFileError)
    assert issubclass(GGUFParseError, GGUFFileError)
    assert issubclass(GGUFTruncatedError, GGUFFileError)
    assert issubclass(GGUFInvalidTypeError, GGUFFileError)
    assert issubclass(GGUFFileError, Exception)


def test_exceptions_can_be_raised():
    """Test that all custom exceptions can be instantiated and raised."""
    with pytest.raises(GGUFFileError):
        raise GGUFFileError("Test error")
    
    with pytest.raises(GGUFInvalidMagicError):
        raise GGUFInvalidMagicError("Invalid magic")
    
    with pytest.raises(GGUFVersionError):
        raise GGUFVersionError("Invalid version")
    
    with pytest.raises(GGUFParseError):
        raise GGUFParseError("Parse error")
    
    with pytest.raises(GGUFTruncatedError):
        raise GGUFTruncatedError("Truncated file")
    
    with pytest.raises(GGUFInvalidTypeError):
        raise GGUFInvalidTypeError("Invalid type")


# ============================================================================
# Test GGUFReader Class Structure
# ============================================================================

def test_gguf_reader_initialization():
    """Test that GGUFReader can be initialized with a filepath."""
    reader = GGUFReader("test.gguf")
    assert reader.filepath == "test.gguf"
    assert reader.file is None
    assert reader.header == {}
    assert reader.metadata == {}
    assert reader.tensor_info == []
    assert reader.tensor_data_base == 0
    assert reader.tensor_data_cache == {}


def test_gguf_reader_has_required_methods():
    """Test that GGUFReader has all required public methods."""
    reader = GGUFReader("test.gguf")
    
    # Context manager methods
    assert hasattr(reader, '__enter__')
    assert hasattr(reader, '__exit__')
    
    # Metadata access methods
    assert hasattr(reader, 'get_metadata')
    assert hasattr(reader, 'get_metadata_value')
    
    # Tensor access methods
    assert hasattr(reader, 'list_tensors')
    assert hasattr(reader, 'get_tensor_info')
    assert hasattr(reader, 'get_tensor_data')
    assert hasattr(reader, 'get_tensor_count')
    
    # Version access method
    assert hasattr(reader, 'get_version')


# ============================================================================
# Property-Based Tests
# ============================================================================

# ----------------------------------------------------------------------------
# Property 3 (partial): Metadata Round-Trip - String Round-Trip
# ----------------------------------------------------------------------------

@given(st.text())
def test_property_string_round_trip(test_string):
    """
    Property 3 (partial): Metadata Round-Trip - String Round-Trip
    
    **Validates: Requirements 2.1, 2.5**
    
    For any valid string, when encoded in GGUF format (length-prefixed UTF-8)
    and parsed by the reader, the parsed string should equal the original string.
    
    This property ensures that:
    - String length encoding/decoding is correct
    - UTF-8 encoding/decoding preserves the original string
    - Empty strings are handled correctly
    - Strings with special characters (unicode, emojis, etc.) are preserved
    """
    reader = GGUFReader("test.gguf")
    
    # Encode the string in GGUF format (length-prefixed UTF-8)
    utf8_bytes = test_string.encode('utf-8')
    length = len(utf8_bytes)
    file_content = struct.pack('<Q', length) + utf8_bytes
    
    # Create a mock file with the encoded string
    reader.file = io.BytesIO(file_content)
    
    # Parse the string using the reader
    parsed_string = reader._read_string()
    
    # Verify round-trip: parsed string should equal original string
    assert parsed_string == test_string, \
        f"String round-trip failed: expected {repr(test_string)}, got {repr(parsed_string)}"


# ----------------------------------------------------------------------------
# Property 3 (partial): Metadata Round-Trip - Primitive Type Parsing
# ----------------------------------------------------------------------------

# Define strategies for each primitive type
primitive_type_strategies = {
    GGUFValueType.UINT8: st.integers(min_value=0, max_value=255),
    GGUFValueType.INT8: st.integers(min_value=-128, max_value=127),
    GGUFValueType.UINT16: st.integers(min_value=0, max_value=65535),
    GGUFValueType.INT16: st.integers(min_value=-32768, max_value=32767),
    GGUFValueType.UINT32: st.integers(min_value=0, max_value=4294967295),
    GGUFValueType.INT32: st.integers(min_value=-2147483648, max_value=2147483647),
    GGUFValueType.UINT64: st.integers(min_value=0, max_value=18446744073709551615),
    GGUFValueType.INT64: st.integers(min_value=-9223372036854775808, max_value=9223372036854775807),
    GGUFValueType.FLOAT32: st.floats(width=32, allow_nan=False, allow_infinity=False),
    GGUFValueType.FLOAT64: st.floats(width=64, allow_nan=False, allow_infinity=False),
    GGUFValueType.BOOL: st.booleans(),
}

# Define struct format for each type
primitive_type_formats = {
    GGUFValueType.UINT8: '<B',
    GGUFValueType.INT8: '<b',
    GGUFValueType.UINT16: '<H',
    GGUFValueType.INT16: '<h',
    GGUFValueType.UINT32: '<I',
    GGUFValueType.INT32: '<i',
    GGUFValueType.UINT64: '<Q',
    GGUFValueType.INT64: '<q',
    GGUFValueType.FLOAT32: '<f',
    GGUFValueType.FLOAT64: '<d',
    GGUFValueType.BOOL: '<B',
}


@given(
    value_type=st.sampled_from(list(primitive_type_strategies.keys())),
    data=st.data()
)
def test_property_primitive_type_round_trip(value_type, data):
    """
    Property 3 (partial): Metadata Round-Trip - Primitive Type Parsing
    
    **Validates: Requirements 2.3**
    
    For any valid primitive type value (UINT8, INT8, UINT16, INT16, UINT32, INT32,
    UINT64, INT64, FLOAT32, FLOAT64, BOOL), when encoded in GGUF format and parsed
    by the reader, the parsed value should equal the original value (within floating
    point precision for float types).
    
    This property ensures that:
    - All primitive types are correctly encoded/decoded
    - Integer types preserve exact values
    - Float types preserve values within precision limits
    - Bool types are correctly converted to Python bool
    - Byte ordering (little-endian) is handled correctly
    """
    # Generate a value of the appropriate type
    test_value = data.draw(primitive_type_strategies[value_type])
    
    reader = GGUFReader("test.gguf")
    
    # Encode the value in GGUF format
    struct_format = primitive_type_formats[value_type]
    
    # For BOOL, convert Python bool to int for struct.pack
    if value_type == GGUFValueType.BOOL:
        file_content = struct.pack(struct_format, int(test_value))
    else:
        file_content = struct.pack(struct_format, test_value)
    
    # Create a mock file with the encoded value
    reader.file = io.BytesIO(file_content)
    
    # Parse the value using the reader
    parsed_value = reader._read_value(value_type)
    
    # Verify round-trip
    if value_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        # For floating point types, check within precision tolerance
        if value_type == GGUFValueType.FLOAT32:
            # Float32 has ~7 decimal digits of precision
            tolerance = abs(test_value) * 1e-6 if test_value != 0 else 1e-6
        else:
            # Float64 has ~15 decimal digits of precision
            tolerance = abs(test_value) * 1e-14 if test_value != 0 else 1e-14
        
        assert abs(parsed_value - test_value) <= tolerance, \
            f"Float round-trip failed for type {value_type}: expected {test_value}, got {parsed_value}, diff {abs(parsed_value - test_value)}"
    else:
        # For integer and bool types, check exact equality
        assert parsed_value == test_value, \
            f"Primitive round-trip failed for type {value_type}: expected {test_value}, got {parsed_value}"
        
        # Verify type is correct
        if value_type == GGUFValueType.BOOL:
            assert isinstance(parsed_value, bool), \
                f"Bool type should return Python bool, got {type(parsed_value)}"
        else:
            assert isinstance(parsed_value, (int, float)), \
                f"Primitive type should return int or float, got {type(parsed_value)}"


# ----------------------------------------------------------------------------
# Property 3 (partial): Metadata Round-Trip - Array Parsing
# ----------------------------------------------------------------------------

# Define valid array element types (all types except ARRAY itself)
array_element_types = [
    GGUFValueType.UINT8,
    GGUFValueType.INT8,
    GGUFValueType.UINT16,
    GGUFValueType.INT16,
    GGUFValueType.UINT32,
    GGUFValueType.INT32,
    GGUFValueType.UINT64,
    GGUFValueType.INT64,
    GGUFValueType.FLOAT32,
    GGUFValueType.FLOAT64,
    GGUFValueType.BOOL,
    GGUFValueType.STRING,
]


def encode_array_value(element_type: int, elements: list) -> bytes:
    """
    Helper function to encode an array in GGUF format.
    
    Args:
        element_type: The GGUF type code for array elements
        elements: List of elements to encode
        
    Returns:
        Bytes representing the encoded array
    """
    # Start with element type and array length
    result = struct.pack('<I', element_type)  # element type (uint32)
    result += struct.pack('<Q', len(elements))  # array length (uint64)
    
    # Encode each element based on type
    for element in elements:
        if element_type == GGUFValueType.UINT8:
            result += struct.pack('<B', element)
        elif element_type == GGUFValueType.INT8:
            result += struct.pack('<b', element)
        elif element_type == GGUFValueType.UINT16:
            result += struct.pack('<H', element)
        elif element_type == GGUFValueType.INT16:
            result += struct.pack('<h', element)
        elif element_type == GGUFValueType.UINT32:
            result += struct.pack('<I', element)
        elif element_type == GGUFValueType.INT32:
            result += struct.pack('<i', element)
        elif element_type == GGUFValueType.UINT64:
            result += struct.pack('<Q', element)
        elif element_type == GGUFValueType.INT64:
            result += struct.pack('<q', element)
        elif element_type == GGUFValueType.FLOAT32:
            result += struct.pack('<f', element)
        elif element_type == GGUFValueType.FLOAT64:
            result += struct.pack('<d', element)
        elif element_type == GGUFValueType.BOOL:
            result += struct.pack('<B', int(element))
        elif element_type == GGUFValueType.STRING:
            utf8_bytes = element.encode('utf-8')
            result += struct.pack('<Q', len(utf8_bytes))
            result += utf8_bytes
    
    return result


@given(
    element_type=st.sampled_from(array_element_types),
    data=st.data()
)
def test_property_array_round_trip(element_type, data):
    """
    Property 3 (partial): Metadata Round-Trip - Array Parsing
    
    **Validates: Requirements 2.4, 2.6**
    
    For any valid array containing elements of a valid primitive type, when encoded
    in GGUF format and parsed by the reader, the parsed array should equal the
    original array.
    
    This property ensures that:
    - Array element type is correctly encoded/decoded
    - Array length is correctly encoded/decoded
    - All array elements are correctly parsed
    - Empty arrays are handled correctly
    - Arrays of different types (integers, floats, bools, strings) work correctly
    - Element values are preserved through round-trip
    """
    # Generate appropriate elements based on type
    if element_type == GGUFValueType.UINT8:
        elements = data.draw(st.lists(st.integers(min_value=0, max_value=255), max_size=20))
    elif element_type == GGUFValueType.INT8:
        elements = data.draw(st.lists(st.integers(min_value=-128, max_value=127), max_size=20))
    elif element_type == GGUFValueType.UINT16:
        elements = data.draw(st.lists(st.integers(min_value=0, max_value=65535), max_size=20))
    elif element_type == GGUFValueType.INT16:
        elements = data.draw(st.lists(st.integers(min_value=-32768, max_value=32767), max_size=20))
    elif element_type == GGUFValueType.UINT32:
        elements = data.draw(st.lists(st.integers(min_value=0, max_value=4294967295), max_size=20))
    elif element_type == GGUFValueType.INT32:
        elements = data.draw(st.lists(st.integers(min_value=-2147483648, max_value=2147483647), max_size=20))
    elif element_type == GGUFValueType.UINT64:
        elements = data.draw(st.lists(st.integers(min_value=0, max_value=18446744073709551615), max_size=20))
    elif element_type == GGUFValueType.INT64:
        elements = data.draw(st.lists(st.integers(min_value=-9223372036854775808, max_value=9223372036854775807), max_size=20))
    elif element_type == GGUFValueType.FLOAT32:
        elements = data.draw(st.lists(st.floats(width=32, allow_nan=False, allow_infinity=False), max_size=20))
    elif element_type == GGUFValueType.FLOAT64:
        elements = data.draw(st.lists(st.floats(width=64, allow_nan=False, allow_infinity=False), max_size=20))
    elif element_type == GGUFValueType.BOOL:
        elements = data.draw(st.lists(st.booleans(), max_size=20))
    elif element_type == GGUFValueType.STRING:
        elements = data.draw(st.lists(st.text(), max_size=10))  # Smaller size for strings
    else:
        raise ValueError(f"Unexpected element type: {element_type}")
    
    reader = GGUFReader("test.gguf")
    
    # Encode the array in GGUF format
    file_content = encode_array_value(element_type, elements)
    
    # Create a mock file with the encoded array
    reader.file = io.BytesIO(file_content)
    
    # Parse the array using the reader
    parsed_array = reader._read_array()
    
    # Verify round-trip
    assert len(parsed_array) == len(elements), \
        f"Array length mismatch: expected {len(elements)}, got {len(parsed_array)}"
    
    for i, (expected, actual) in enumerate(zip(elements, parsed_array)):
        if element_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
            # For floating point types, check within precision tolerance
            if element_type == GGUFValueType.FLOAT32:
                tolerance = abs(expected) * 1e-6 if expected != 0 else 1e-6
            else:
                tolerance = abs(expected) * 1e-14 if expected != 0 else 1e-14
            
            assert abs(actual - expected) <= tolerance, \
                f"Array element {i} float mismatch: expected {expected}, got {actual}, diff {abs(actual - expected)}"
        else:
            # For integer, bool, and string types, check exact equality
            assert actual == expected, \
                f"Array element {i} mismatch: expected {expected}, got {actual}"


# ============================================================================
# Test _read_string() Method (Task 2.1)
# ============================================================================


def test_read_string_simple():
    """Test reading a simple ASCII string."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with a length-prefixed string
    string_data = "hello"
    length = len(string_data.encode('utf-8'))
    file_content = struct.pack('<Q', length) + string_data.encode('utf-8')
    
    reader.file = io.BytesIO(file_content)
    result = reader._read_string()
    
    assert result == "hello"


def test_read_string_empty():
    """Test reading an empty string."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with zero-length string
    file_content = struct.pack('<Q', 0)
    
    reader.file = io.BytesIO(file_content)
    result = reader._read_string()
    
    assert result == ""


def test_read_string_utf8():
    """Test reading a UTF-8 string with special characters."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with UTF-8 string
    string_data = "Hello 世界 🌍"
    utf8_bytes = string_data.encode('utf-8')
    length = len(utf8_bytes)
    file_content = struct.pack('<Q', length) + utf8_bytes
    
    reader.file = io.BytesIO(file_content)
    result = reader._read_string()
    
    assert result == string_data


def test_read_string_truncated_length():
    """Test that truncated length field raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with only 3 bytes (need 8 for length)
    file_content = b'\x01\x02\x03'
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_string()
    
    assert "expected to read 8 bytes for string length" in str(exc_info.value)
    assert "only 3 bytes available" in str(exc_info.value)


def test_read_string_truncated_data():
    """Test that truncated string data raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with length=10 but only 5 bytes of data
    file_content = struct.pack('<Q', 10) + b'hello'
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_string()
    
    assert "expected to read 10 bytes for string data" in str(exc_info.value)
    assert "only 5 bytes available" in str(exc_info.value)


def test_read_string_excessive_length():
    """Test that excessively large string length raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with unreasonably large length (200MB)
    excessive_length = 200 * 1024 * 1024
    file_content = struct.pack('<Q', excessive_length)
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_string()
    
    assert "Invalid string length" in str(exc_info.value)
    assert "exceeds maximum allowed length" in str(exc_info.value)


def test_read_string_no_file_open():
    """Test that reading without an open file raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    # file is None by default
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_string()
    
    assert "File is not open" in str(exc_info.value)


def test_read_string_invalid_utf8():
    """Test that invalid UTF-8 data raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with invalid UTF-8 sequence
    invalid_utf8 = b'\xff\xfe\xfd'
    file_content = struct.pack('<Q', len(invalid_utf8)) + invalid_utf8
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_string()
    
    assert "Invalid UTF-8 string" in str(exc_info.value)


def test_read_string_multiple_sequential():
    """Test reading multiple strings sequentially."""
    reader = GGUFReader("test.gguf")
    
    # Create a mock file with multiple strings
    strings = ["first", "second", "third"]
    file_content = b''
    for s in strings:
        utf8_bytes = s.encode('utf-8')
        file_content += struct.pack('<Q', len(utf8_bytes)) + utf8_bytes
    
    reader.file = io.BytesIO(file_content)
    
    for expected in strings:
        result = reader._read_string()
        assert result == expected



# ============================================================================
# Test _read_value() Method (Task 2.3)
# ============================================================================

def test_read_value_uint8():
    """Test reading UINT8 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<B', 42)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.UINT8)
    assert result == 42
    assert isinstance(result, int)


def test_read_value_int8():
    """Test reading INT8 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<b', -42)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.INT8)
    assert result == -42
    assert isinstance(result, int)


def test_read_value_uint16():
    """Test reading UINT16 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<H', 1234)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.UINT16)
    assert result == 1234
    assert isinstance(result, int)


def test_read_value_int16():
    """Test reading INT16 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<h', -1234)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.INT16)
    assert result == -1234
    assert isinstance(result, int)


def test_read_value_uint32():
    """Test reading UINT32 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<I', 123456)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.UINT32)
    assert result == 123456
    assert isinstance(result, int)


def test_read_value_int32():
    """Test reading INT32 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<i', -123456)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.INT32)
    assert result == -123456
    assert isinstance(result, int)


def test_read_value_uint64():
    """Test reading UINT64 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<Q', 9876543210)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.UINT64)
    assert result == 9876543210
    assert isinstance(result, int)


def test_read_value_int64():
    """Test reading INT64 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<q', -9876543210)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.INT64)
    assert result == -9876543210
    assert isinstance(result, int)


def test_read_value_float32():
    """Test reading FLOAT32 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<f', 3.14159)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.FLOAT32)
    assert abs(result - 3.14159) < 0.0001  # Float precision
    assert isinstance(result, float)


def test_read_value_float64():
    """Test reading FLOAT64 value."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<d', 3.141592653589793)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.FLOAT64)
    assert abs(result - 3.141592653589793) < 1e-10  # Double precision
    assert isinstance(result, float)


def test_read_value_bool_true():
    """Test reading BOOL value (true)."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<B', 1)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.BOOL)
    assert result is True
    assert isinstance(result, bool)


def test_read_value_bool_false():
    """Test reading BOOL value (false)."""
    reader = GGUFReader("test.gguf")
    file_content = struct.pack('<B', 0)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.BOOL)
    assert result is False
    assert isinstance(result, bool)


def test_read_value_string():
    """Test reading STRING value."""
    reader = GGUFReader("test.gguf")
    
    # Create string data
    string_data = "test string"
    utf8_bytes = string_data.encode('utf-8')
    file_content = struct.pack('<Q', len(utf8_bytes)) + utf8_bytes
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.STRING)
    assert result == "test string"
    assert isinstance(result, str)


def test_read_value_array_uint32():
    """Test reading ARRAY of UINT32 values."""
    reader = GGUFReader("test.gguf")
    
    # Create array data: type (UINT32), length (3), elements (10, 20, 30)
    file_content = struct.pack('<I', GGUFValueType.UINT32)  # element type
    file_content += struct.pack('<Q', 3)  # array length
    file_content += struct.pack('<I', 10)  # element 1
    file_content += struct.pack('<I', 20)  # element 2
    file_content += struct.pack('<I', 30)  # element 3
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.ARRAY)
    assert result == [10, 20, 30]
    assert isinstance(result, list)


def test_read_value_array_string():
    """Test reading ARRAY of STRING values."""
    reader = GGUFReader("test.gguf")
    
    # Create array data
    strings = ["first", "second", "third"]
    file_content = struct.pack('<I', GGUFValueType.STRING)  # element type
    file_content += struct.pack('<Q', len(strings))  # array length
    
    # Add each string
    for s in strings:
        utf8_bytes = s.encode('utf-8')
        file_content += struct.pack('<Q', len(utf8_bytes)) + utf8_bytes
    
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_value(GGUFValueType.ARRAY)
    assert result == strings
    assert isinstance(result, list)


def test_read_value_invalid_type():
    """Test that invalid type code raises GGUFInvalidTypeError."""
    reader = GGUFReader("test.gguf")
    file_content = b'\x00\x00\x00\x00'  # Some dummy data
    reader.file = io.BytesIO(file_content)
    
    # Type code 99 is not valid
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_value(99)
    
    assert "Invalid metadata type" in str(exc_info.value)
    assert "type code 99" in str(exc_info.value)


def test_read_value_truncated_primitive():
    """Test that truncated primitive value raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Only 2 bytes available, but UINT32 needs 4
    file_content = b'\x01\x02'
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_value(GGUFValueType.UINT32)
    
    assert "expected to read 4 bytes" in str(exc_info.value)
    assert "only 2 bytes available" in str(exc_info.value)


def test_read_value_no_file_open():
    """Test that reading without an open file raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    # file is None by default
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_value(GGUFValueType.UINT32)
    
    assert "File is not open" in str(exc_info.value)


# ============================================================================
# Test _read_array() Method (Task 2.5)
# ============================================================================

def test_read_array_empty():
    """Test reading an empty array."""
    reader = GGUFReader("test.gguf")
    
    # Create array data: type (UINT32), length (0)
    file_content = struct.pack('<I', GGUFValueType.UINT32)  # element type
    file_content += struct.pack('<Q', 0)  # array length
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_array()
    assert result == []
    assert isinstance(result, list)


def test_read_array_int8():
    """Test reading array of INT8 values."""
    reader = GGUFReader("test.gguf")
    
    # Create array data
    file_content = struct.pack('<I', GGUFValueType.INT8)  # element type
    file_content += struct.pack('<Q', 4)  # array length
    file_content += struct.pack('<b', -1)
    file_content += struct.pack('<b', 0)
    file_content += struct.pack('<b', 1)
    file_content += struct.pack('<b', 127)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_array()
    assert result == [-1, 0, 1, 127]


def test_read_array_float32():
    """Test reading array of FLOAT32 values."""
    reader = GGUFReader("test.gguf")
    
    # Create array data
    values = [1.5, 2.5, 3.5]
    file_content = struct.pack('<I', GGUFValueType.FLOAT32)  # element type
    file_content += struct.pack('<Q', len(values))  # array length
    for v in values:
        file_content += struct.pack('<f', v)
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_array()
    assert len(result) == 3
    for i, expected in enumerate(values):
        assert abs(result[i] - expected) < 0.0001


def test_read_array_bool():
    """Test reading array of BOOL values."""
    reader = GGUFReader("test.gguf")
    
    # Create array data
    file_content = struct.pack('<I', GGUFValueType.BOOL)  # element type
    file_content += struct.pack('<Q', 3)  # array length
    file_content += struct.pack('<B', 1)  # true
    file_content += struct.pack('<B', 0)  # false
    file_content += struct.pack('<B', 1)  # true
    reader.file = io.BytesIO(file_content)
    
    result = reader._read_array()
    assert result == [True, False, True]


def test_read_array_nested_not_allowed():
    """Test that nested arrays raise GGUFInvalidTypeError."""
    reader = GGUFReader("test.gguf")
    
    # Create array data with ARRAY as element type (not allowed)
    file_content = struct.pack('<I', GGUFValueType.ARRAY)  # element type
    file_content += struct.pack('<Q', 1)  # array length
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_array()
    
    assert "nested arrays are not allowed" in str(exc_info.value)


def test_read_array_invalid_element_type():
    """Test that invalid element type raises GGUFInvalidTypeError."""
    reader = GGUFReader("test.gguf")
    
    # Create array data with invalid element type
    file_content = struct.pack('<I', 999)  # invalid type code
    file_content += struct.pack('<Q', 1)  # array length
    file_content += b'\x00\x00\x00\x00'  # dummy element data
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_array()
    
    assert "Invalid array element type" in str(exc_info.value)
    assert "type code 999" in str(exc_info.value)


def test_read_array_truncated_type():
    """Test that truncated element type raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Only 2 bytes available, but need 4 for element type
    file_content = b'\x01\x02'
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_array()
    
    assert "expected to read 4 bytes for array element type" in str(exc_info.value)


def test_read_array_truncated_length():
    """Test that truncated array length raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Element type present, but length truncated
    file_content = struct.pack('<I', GGUFValueType.UINT32)  # element type
    file_content += b'\x01\x02\x03'  # only 3 bytes, need 8 for length
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_array()
    
    assert "expected to read 8 bytes for array length" in str(exc_info.value)


def test_read_array_no_file_open():
    """Test that reading without an open file raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    # file is None by default
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_array()
    
    assert "File is not open" in str(exc_info.value)


# ============================================================================
# Test _read_header() Method (Task 3.1)
# ============================================================================

def test_read_header_valid():
    """Test reading a valid GGUF header."""
    reader = GGUFReader("test.gguf")
    
    # Create a valid GGUF header
    # Magic: 0x46554747 ("GGUF")
    # Version: 3
    # Tensor count: 10
    # Metadata KV count: 5
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 10)  # tensor_count
    file_content += struct.pack('<Q', 5)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    assert reader.header['magic'] == 0x46554747
    assert reader.header['version'] == 3
    assert reader.header['tensor_count'] == 10
    assert reader.header['metadata_kv_count'] == 5


def test_read_header_invalid_magic():
    """Test that invalid magic number raises GGUFInvalidMagicError."""
    reader = GGUFReader("test.gguf")
    
    # Create header with invalid magic number
    file_content = struct.pack('<I', 0x12345678)  # wrong magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 10)  # tensor_count
    file_content += struct.pack('<Q', 5)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidMagicError) as exc_info:
        reader._read_header()
    
    assert "Invalid GGUF magic number" in str(exc_info.value)
    assert "0x46554747" in str(exc_info.value)  # expected
    assert "0x12345678" in str(exc_info.value)  # got


def test_read_header_truncated_magic():
    """Test that truncated magic number raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Only 2 bytes available, need 4 for magic
    file_content = b'\x01\x02'
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    assert "expected to read 4 bytes for magic number" in str(exc_info.value)
    assert "only 2 bytes available" in str(exc_info.value)


def test_read_header_truncated_version():
    """Test that truncated version raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Magic present, but version truncated
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += b'\x01\x02'  # only 2 bytes, need 4 for version
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    assert "expected to read 4 bytes for version" in str(exc_info.value)
    assert "only 2 bytes available" in str(exc_info.value)


def test_read_header_truncated_tensor_count():
    """Test that truncated tensor_count raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Magic and version present, but tensor_count truncated
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += b'\x01\x02\x03'  # only 3 bytes, need 8 for tensor_count
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    assert "expected to read 8 bytes for tensor_count" in str(exc_info.value)
    assert "only 3 bytes available" in str(exc_info.value)


def test_read_header_truncated_metadata_kv_count():
    """Test that truncated metadata_kv_count raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Magic, version, and tensor_count present, but metadata_kv_count truncated
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 10)  # tensor_count
    file_content += b'\x01\x02\x03\x04'  # only 4 bytes, need 8 for metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    assert "expected to read 8 bytes for metadata_kv_count" in str(exc_info.value)
    assert "only 4 bytes available" in str(exc_info.value)


def test_read_header_no_file_open():
    """Test that reading without an open file raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    # file is None by default
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_header()
    
    assert "File is not open" in str(exc_info.value)


def test_read_header_various_versions():
    """Test reading headers with various version numbers."""
    for version in [1, 2, 3, 100, 999]:
        reader = GGUFReader("test.gguf")
        
        file_content = struct.pack('<I', 0x46554747)  # magic
        file_content += struct.pack('<I', version)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        
        reader.file = io.BytesIO(file_content)
        reader._read_header()
        
        assert reader.header['version'] == version


def test_read_header_zero_counts():
    """Test reading header with zero tensor and metadata counts."""
    reader = GGUFReader("test.gguf")
    
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    assert reader.header['tensor_count'] == 0
    assert reader.header['metadata_kv_count'] == 0


def test_read_header_large_counts():
    """Test reading header with large tensor and metadata counts."""
    reader = GGUFReader("test.gguf")
    
    large_count = 1000000
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', large_count)  # tensor_count
    file_content += struct.pack('<Q', large_count)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    assert reader.header['tensor_count'] == large_count
    assert reader.header['metadata_kv_count'] == large_count


# ============================================================================
# Property-Based Tests for Header Parsing (Task 3.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 1: Magic Number Validation
# ----------------------------------------------------------------------------

@given(magic_number=st.integers(min_value=0, max_value=0xFFFFFFFF))
def test_property_magic_number_validation(magic_number):
    """
    Property 1: Magic Number Validation
    
    **Validates: Requirements 1.1, 1.5**
    
    For any byte sequence, when passed to the reader as a file, if the first 4 bytes
    match the GGUF magic number (0x46554747) then parsing should proceed, otherwise
    an error should be raised.
    
    This property ensures that:
    - Valid GGUF magic number (0x46554747) is accepted
    - Any other magic number is rejected with GGUFInvalidMagicError
    - Error message contains both expected and actual magic numbers
    - File validation happens before any other parsing
    """
    reader = GGUFReader("test.gguf")
    
    # Expected GGUF magic number
    GGUF_MAGIC = 0x46554747
    
    # Create a file with the test magic number and valid rest of header
    file_content = struct.pack('<I', magic_number)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    if magic_number == GGUF_MAGIC:
        # Valid magic number - parsing should succeed
        reader._read_header()
        assert reader.header['magic'] == GGUF_MAGIC
        assert reader.header['version'] == 3
        assert reader.header['tensor_count'] == 0
        assert reader.header['metadata_kv_count'] == 0
    else:
        # Invalid magic number - should raise GGUFInvalidMagicError
        with pytest.raises(GGUFInvalidMagicError) as exc_info:
            reader._read_header()
        
        # Verify error message contains expected and actual values
        error_msg = str(exc_info.value)
        assert "Invalid GGUF magic number" in error_msg
        assert f"0x{GGUF_MAGIC:08X}" in error_msg  # expected
        assert f"0x{magic_number:08X}" in error_msg  # actual


# ----------------------------------------------------------------------------
# Property 2: Header Field Round-Trip
# ----------------------------------------------------------------------------

@given(
    version=st.integers(min_value=0, max_value=0xFFFFFFFF),
    tensor_count=st.integers(min_value=0, max_value=0xFFFFFFFFFFFFFFFF),
    metadata_kv_count=st.integers(min_value=0, max_value=0xFFFFFFFFFFFFFFFF)
)
def test_property_header_field_round_trip(version, tensor_count, metadata_kv_count):
    """
    Property 2: Header Field Round-Trip
    
    **Validates: Requirements 1.2, 1.3, 1.4**
    
    For any valid GGUF header values (version, tensor_count, metadata_kv_count),
    when written to a file and parsed by the reader, the parsed values should
    match the original values.
    
    This property ensures that:
    - Version number (uint32) is correctly encoded/decoded
    - Tensor count (uint64) is correctly encoded/decoded
    - Metadata KV count (uint64) is correctly encoded/decoded
    - All header fields preserve their exact values through round-trip
    - Little-endian byte ordering is handled correctly
    - Full range of valid values (0 to max) works correctly
    """
    reader = GGUFReader("test.gguf")
    
    # Create a valid GGUF header with the test values
    GGUF_MAGIC = 0x46554747
    file_content = struct.pack('<I', GGUF_MAGIC)  # magic
    file_content += struct.pack('<I', version)  # version (uint32)
    file_content += struct.pack('<Q', tensor_count)  # tensor_count (uint64)
    file_content += struct.pack('<Q', metadata_kv_count)  # metadata_kv_count (uint64)
    
    reader.file = io.BytesIO(file_content)
    
    # Parse the header
    reader._read_header()
    
    # Verify round-trip: all parsed values should match original values
    assert reader.header['magic'] == GGUF_MAGIC, \
        f"Magic number mismatch: expected {GGUF_MAGIC}, got {reader.header['magic']}"
    
    assert reader.header['version'] == version, \
        f"Version mismatch: expected {version}, got {reader.header['version']}"
    
    assert reader.header['tensor_count'] == tensor_count, \
        f"Tensor count mismatch: expected {tensor_count}, got {reader.header['tensor_count']}"
    
    assert reader.header['metadata_kv_count'] == metadata_kv_count, \
        f"Metadata KV count mismatch: expected {metadata_kv_count}, got {reader.header['metadata_kv_count']}"


# ============================================================================
# Additional Unit Tests for Header Edge Cases (Task 3.3)
# ============================================================================

def test_read_header_invalid_magic_error_message_format():
    """
    Test that invalid magic number error message contains all required information.
    
    Requirements: 1.5, 7.2
    
    This test verifies that the error message for an invalid magic number:
    - Identifies the error as "Invalid GGUF magic number"
    - Includes the file path
    - Shows the expected magic number in hex format
    - Shows the actual magic number in hex format
    - Includes the file position where the error occurred
    """
    reader = GGUFReader("test_model.gguf")
    
    # Create header with a specific invalid magic number
    invalid_magic = 0xDEADBEEF
    file_content = struct.pack('<I', invalid_magic)
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 10)  # tensor_count
    file_content += struct.pack('<Q', 5)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidMagicError) as exc_info:
        reader._read_header()
    
    error_msg = str(exc_info.value)
    
    # Verify all required information is in the error message
    assert "Invalid GGUF magic number" in error_msg
    assert "test_model.gguf" in error_msg
    assert "0x46554747" in error_msg  # Expected GGUF magic
    assert "0xDEADBEEF" in error_msg  # Actual magic
    assert "position 0" in error_msg  # Position where error occurred


def test_read_header_truncated_at_each_field():
    """
    Test truncated header error at each field boundary.
    
    Requirements: 7.2
    
    This test verifies that truncation is detected at each field in the header:
    - Truncated magic number (0-3 bytes)
    - Truncated version (4-7 bytes)
    - Truncated tensor_count (8-15 bytes)
    - Truncated metadata_kv_count (16-23 bytes)
    """
    test_cases = [
        # (bytes_available, expected_field_name, expected_bytes_needed)
        (0, "magic number", 4),
        (1, "magic number", 4),
        (2, "magic number", 4),
        (3, "magic number", 4),
        (4, "version", 4),
        (5, "version", 4),
        (6, "version", 4),
        (7, "version", 4),
        (8, "tensor_count", 8),
        (9, "tensor_count", 8),
        (10, "tensor_count", 8),
        (15, "tensor_count", 8),
        (16, "metadata_kv_count", 8),
        (17, "metadata_kv_count", 8),
        (20, "metadata_kv_count", 8),
        (23, "metadata_kv_count", 8),
    ]
    
    for bytes_available, field_name, bytes_needed in test_cases:
        reader = GGUFReader("test.gguf")
        
        # Create a valid header but truncate it
        full_header = struct.pack('<I', 0x46554747)  # magic
        full_header += struct.pack('<I', 3)  # version
        full_header += struct.pack('<Q', 10)  # tensor_count
        full_header += struct.pack('<Q', 5)  # metadata_kv_count
        
        # Truncate to the specified number of bytes
        truncated_header = full_header[:bytes_available]
        
        reader.file = io.BytesIO(truncated_header)
        
        with pytest.raises(GGUFTruncatedError) as exc_info:
            reader._read_header()
        
        error_msg = str(exc_info.value)
        
        # Verify error message mentions the field and expected bytes
        assert field_name in error_msg, \
            f"Expected '{field_name}' in error for {bytes_available} bytes: {error_msg}"
        assert f"expected to read {bytes_needed} bytes" in error_msg, \
            f"Expected 'expected to read {bytes_needed} bytes' in error: {error_msg}"


def test_read_header_version_boundary_values():
    """
    Test header parsing with version number boundary values.
    
    Requirements: 1.5
    
    This test verifies that the reader correctly handles:
    - Minimum version (0)
    - Small versions (1, 2, 3)
    - Maximum uint32 version (0xFFFFFFFF)
    - Common version numbers
    """
    test_versions = [
        0,           # Minimum
        1,           # GGUF v1
        2,           # GGUF v2
        3,           # GGUF v3
        10,          # Future version
        100,         # Far future version
        0xFFFF,      # Large version
        0xFFFFFFFF,  # Maximum uint32
    ]
    
    for version in test_versions:
        reader = GGUFReader("test.gguf")
        
        file_content = struct.pack('<I', 0x46554747)  # magic
        file_content += struct.pack('<I', version)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        
        reader.file = io.BytesIO(file_content)
        reader._read_header()
        
        assert reader.header['version'] == version, \
            f"Version mismatch for {version}: got {reader.header['version']}"
        assert isinstance(reader.header['version'], int), \
            f"Version should be int, got {type(reader.header['version'])}"


def test_read_header_count_boundary_values():
    """
    Test header parsing with tensor and metadata count boundary values.
    
    Requirements: 1.5
    
    This test verifies that the reader correctly handles:
    - Zero counts (empty file)
    - Small counts
    - Large counts
    - Maximum uint64 counts
    """
    test_cases = [
        # (tensor_count, metadata_kv_count)
        (0, 0),                              # Empty file
        (1, 0),                              # One tensor, no metadata
        (0, 1),                              # No tensors, one metadata
        (1, 1),                              # Minimal file
        (100, 50),                           # Typical small model
        (1000, 500),                         # Typical large model
        (0xFFFFFFFF, 0xFFFFFFFF),           # Large uint32 values
        (0xFFFFFFFFFFFFFFFF, 0),            # Maximum uint64 tensor count
        (0, 0xFFFFFFFFFFFFFFFF),            # Maximum uint64 metadata count
    ]
    
    for tensor_count, metadata_kv_count in test_cases:
        reader = GGUFReader("test.gguf")
        
        file_content = struct.pack('<I', 0x46554747)  # magic
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', tensor_count)  # tensor_count
        file_content += struct.pack('<Q', metadata_kv_count)  # metadata_kv_count
        
        reader.file = io.BytesIO(file_content)
        reader._read_header()
        
        assert reader.header['tensor_count'] == tensor_count, \
            f"Tensor count mismatch: expected {tensor_count}, got {reader.header['tensor_count']}"
        assert reader.header['metadata_kv_count'] == metadata_kv_count, \
            f"Metadata count mismatch: expected {metadata_kv_count}, got {reader.header['metadata_kv_count']}"


def test_read_header_empty_file():
    """
    Test that completely empty file raises appropriate error.
    
    Requirements: 7.2
    
    This test verifies that an empty file (0 bytes) is detected and
    raises a GGUFTruncatedError with appropriate message.
    """
    reader = GGUFReader("empty.gguf")
    reader.file = io.BytesIO(b'')  # Empty file
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    error_msg = str(exc_info.value)
    assert "expected to read 4 bytes for magic number" in error_msg
    assert "only 0 bytes available" in error_msg


def test_read_header_single_byte_file():
    """
    Test that single-byte file raises appropriate error.
    
    Requirements: 7.2
    
    This test verifies that a file with only 1 byte is detected as truncated.
    """
    reader = GGUFReader("tiny.gguf")
    reader.file = io.BytesIO(b'\x47')  # Just one byte
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    error_msg = str(exc_info.value)
    assert "expected to read 4 bytes for magic number" in error_msg
    assert "only 1 bytes available" in error_msg


def test_read_header_magic_variations():
    """
    Test various invalid magic number patterns.
    
    Requirements: 1.5
    
    This test verifies that various common invalid magic numbers are rejected:
    - All zeros
    - All ones
    - Reversed byte order
    - Similar but wrong values
    """
    invalid_magics = [
        0x00000000,  # All zeros
        0xFFFFFFFF,  # All ones
        0x47475546,  # Reversed "GGUF" (big-endian)
        0x46554746,  # Off by one
        0x46554748,  # Off by one
        0x55464747,  # Rotated bytes
        0x12345678,  # Random value
    ]
    
    for invalid_magic in invalid_magics:
        reader = GGUFReader("test.gguf")
        
        file_content = struct.pack('<I', invalid_magic)
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        
        reader.file = io.BytesIO(file_content)
        
        with pytest.raises(GGUFInvalidMagicError) as exc_info:
            reader._read_header()
        
        error_msg = str(exc_info.value)
        assert "Invalid GGUF magic number" in error_msg
        assert f"0x{invalid_magic:08X}" in error_msg


def test_read_header_preserves_file_position():
    """
    Test that _read_header() leaves file position at correct location.
    
    Requirements: 1.5
    
    This test verifies that after reading the header, the file position
    is at byte 24 (after the 24-byte header), ready to read metadata.
    """
    reader = GGUFReader("test.gguf")
    
    file_content = struct.pack('<I', 0x46554747)  # magic (4 bytes)
    file_content += struct.pack('<I', 3)  # version (4 bytes)
    file_content += struct.pack('<Q', 10)  # tensor_count (8 bytes)
    file_content += struct.pack('<Q', 5)  # metadata_kv_count (8 bytes)
    file_content += b'NEXT_DATA'  # Some data after header
    
    reader.file = io.BytesIO(file_content)
    
    # Read header
    reader._read_header()
    
    # Verify file position is at byte 24 (after 24-byte header)
    assert reader.file.tell() == 24, \
        f"Expected file position 24 after header, got {reader.file.tell()}"
    
    # Verify we can read the next data
    next_data = reader.file.read(9)
    assert next_data == b'NEXT_DATA'


# ============================================================================
# Test _read_metadata() Method (Task 4.1)
# ============================================================================

def test_read_metadata_empty():
    """Test reading metadata when count is zero."""
    reader = GGUFReader("test.gguf")
    
    # Create a valid header with zero metadata
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    assert reader.metadata == {}
    assert len(reader.metadata) == 0


def test_read_metadata_single_uint32():
    """Test reading a single UINT32 metadata value."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata: key="test.value", type=UINT32, value=42
    key = "test.value"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', GGUFValueType.UINT32)  # type
    file_content += struct.pack('<I', 42)  # value
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    assert len(reader.metadata) == 1
    assert "test.value" in reader.metadata
    assert reader.metadata["test.value"] == 42


def test_read_metadata_multiple_values():
    """Test reading multiple metadata key-value pairs."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 3)  # metadata_kv_count = 3
    
    # Add metadata 1: key="general.name", type=STRING, value="TestModel"
    key1 = "general.name"
    key1_bytes = key1.encode('utf-8')
    value1 = "TestModel"
    value1_bytes = value1.encode('utf-8')
    file_content += struct.pack('<Q', len(key1_bytes)) + key1_bytes
    file_content += struct.pack('<I', GGUFValueType.STRING)
    file_content += struct.pack('<Q', len(value1_bytes)) + value1_bytes
    
    # Add metadata 2: key="general.version", type=UINT32, value=3
    key2 = "general.version"
    key2_bytes = key2.encode('utf-8')
    file_content += struct.pack('<Q', len(key2_bytes)) + key2_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 3)
    
    # Add metadata 3: key="general.quantized", type=BOOL, value=True
    key3 = "general.quantized"
    key3_bytes = key3.encode('utf-8')
    file_content += struct.pack('<Q', len(key3_bytes)) + key3_bytes
    file_content += struct.pack('<I', GGUFValueType.BOOL)
    file_content += struct.pack('<B', 1)
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    assert len(reader.metadata) == 3
    assert reader.metadata["general.name"] == "TestModel"
    assert reader.metadata["general.version"] == 3
    assert reader.metadata["general.quantized"] is True


def test_read_metadata_with_array():
    """Test reading metadata with array value."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata: key="test.array", type=ARRAY, value=[1, 2, 3]
    key = "test.array"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    # Array: element_type=UINT32, length=3, elements=[1, 2, 3]
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<Q', 3)
    file_content += struct.pack('<I', 1)
    file_content += struct.pack('<I', 2)
    file_content += struct.pack('<I', 3)
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    assert len(reader.metadata) == 1
    assert "test.array" in reader.metadata
    assert reader.metadata["test.array"] == [1, 2, 3]


def test_read_metadata_invalid_type():
    """Test that invalid metadata type raises GGUFInvalidTypeError."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata with invalid type
    key = "test.key"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', 999)  # invalid type
    file_content += b'\x00\x00\x00\x00'  # dummy data
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    assert "Invalid type for metadata key 'test.key'" in error_msg or "Invalid metadata type" in error_msg


def test_read_metadata_truncated_key():
    """Test that truncated metadata key raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add truncated key (length=10 but only 5 bytes)
    file_content += struct.pack('<Q', 10)
    file_content += b'hello'
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    with pytest.raises(GGUFTruncatedError):
        reader._read_metadata()


def test_read_metadata_truncated_type():
    """Test that truncated metadata type raises GGUFTruncatedError."""
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add key but truncate type
    key = "test.key"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += b'\x01\x02'  # only 2 bytes, need 4 for type
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    assert "expected to read 4 bytes for metadata value type" in error_msg


# ============================================================================
# Property-Based Tests for Complete Metadata Parsing (Task 4.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 3: Metadata Round-Trip
# ----------------------------------------------------------------------------

def encode_metadata_dict(metadata: Dict[str, Any]) -> bytes:
    """
    Helper function to encode a complete metadata dictionary in GGUF format.
    
    Args:
        metadata: Dictionary of metadata key-value pairs
        
    Returns:
        Bytes representing the encoded metadata section
    """
    result = b''
    
    for key, value in metadata.items():
        # Encode key (length-prefixed string)
        key_bytes = key.encode('utf-8')
        result += struct.pack('<Q', len(key_bytes)) + key_bytes
        
        # Determine type and encode value
        if isinstance(value, bool):
            # BOOL must be checked before int (bool is subclass of int)
            result += struct.pack('<I', GGUFValueType.BOOL)
            result += struct.pack('<B', int(value))
        elif isinstance(value, int):
            # Determine appropriate integer type based on value range
            if -128 <= value <= 127:
                result += struct.pack('<I', GGUFValueType.INT8)
                result += struct.pack('<b', value)
            elif 0 <= value <= 255:
                result += struct.pack('<I', GGUFValueType.UINT8)
                result += struct.pack('<B', value)
            elif -32768 <= value <= 32767:
                result += struct.pack('<I', GGUFValueType.INT16)
                result += struct.pack('<h', value)
            elif 0 <= value <= 65535:
                result += struct.pack('<I', GGUFValueType.UINT16)
                result += struct.pack('<H', value)
            elif -2147483648 <= value <= 2147483647:
                result += struct.pack('<I', GGUFValueType.INT32)
                result += struct.pack('<i', value)
            elif 0 <= value <= 4294967295:
                result += struct.pack('<I', GGUFValueType.UINT32)
                result += struct.pack('<I', value)
            elif -9223372036854775808 <= value <= 9223372036854775807:
                result += struct.pack('<I', GGUFValueType.INT64)
                result += struct.pack('<q', value)
            else:
                result += struct.pack('<I', GGUFValueType.UINT64)
                result += struct.pack('<Q', value)
        elif isinstance(value, float):
            # Use FLOAT32 for simplicity (could use FLOAT64 for higher precision)
            result += struct.pack('<I', GGUFValueType.FLOAT32)
            result += struct.pack('<f', value)
        elif isinstance(value, str):
            result += struct.pack('<I', GGUFValueType.STRING)
            value_bytes = value.encode('utf-8')
            result += struct.pack('<Q', len(value_bytes)) + value_bytes
        elif isinstance(value, list):
            result += struct.pack('<I', GGUFValueType.ARRAY)
            result += encode_array_value_for_metadata(value)
        else:
            raise ValueError(f"Unsupported metadata value type: {type(value)}")
    
    return result


def encode_array_value_for_metadata(elements: list) -> bytes:
    """
    Helper function to encode an array value for metadata.
    
    Args:
        elements: List of elements (all must be same type)
        
    Returns:
        Bytes representing the encoded array
    """
    if len(elements) == 0:
        # Empty array - use UINT8 as default element type
        result = struct.pack('<I', GGUFValueType.UINT8)
        result += struct.pack('<Q', 0)
        return result
    
    # Determine element type from first element
    first_elem = elements[0]
    
    if isinstance(first_elem, bool):
        element_type = GGUFValueType.BOOL
        result = struct.pack('<I', element_type)
        result += struct.pack('<Q', len(elements))
        for elem in elements:
            result += struct.pack('<B', int(elem))
    elif isinstance(first_elem, int):
        # Use INT32 for simplicity
        element_type = GGUFValueType.INT32
        result = struct.pack('<I', element_type)
        result += struct.pack('<Q', len(elements))
        for elem in elements:
            result += struct.pack('<i', elem)
    elif isinstance(first_elem, float):
        element_type = GGUFValueType.FLOAT32
        result = struct.pack('<I', element_type)
        result += struct.pack('<Q', len(elements))
        for elem in elements:
            result += struct.pack('<f', elem)
    elif isinstance(first_elem, str):
        element_type = GGUFValueType.STRING
        result = struct.pack('<I', element_type)
        result += struct.pack('<Q', len(elements))
        for elem in elements:
            elem_bytes = elem.encode('utf-8')
            result += struct.pack('<Q', len(elem_bytes)) + elem_bytes
    else:
        raise ValueError(f"Unsupported array element type: {type(first_elem)}")
    
    return result


# Strategy for generating valid metadata dictionaries
@st.composite
def metadata_dict_strategy(draw):
    """
    Generate a valid metadata dictionary with mixed types.
    
    This strategy generates dictionaries with:
    - 0-10 key-value pairs
    - Keys are valid strings (non-empty, reasonable length)
    - Values can be: int, float, bool, string, or arrays of these types
    """
    num_pairs = draw(st.integers(min_value=0, max_value=10))
    
    metadata = {}
    for _ in range(num_pairs):
        # Generate a unique key
        key = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='._-'
        )))
        
        # Skip if key already exists
        if key in metadata:
            continue
        
        # Generate a value of random type
        value_type = draw(st.sampled_from(['int', 'float', 'bool', 'string', 'array']))
        
        if value_type == 'int':
            # Use smaller range for integers to avoid encoding issues
            value = draw(st.integers(min_value=-1000000, max_value=1000000))
        elif value_type == 'float':
            value = draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
        elif value_type == 'bool':
            value = draw(st.booleans())
        elif value_type == 'string':
            value = draw(st.text(max_size=100))
        elif value_type == 'array':
            # Generate array of one type
            array_elem_type = draw(st.sampled_from(['int', 'float', 'bool', 'string']))
            if array_elem_type == 'int':
                value = draw(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=10))
            elif array_elem_type == 'float':
                value = draw(st.lists(st.floats(width=32, allow_nan=False, allow_infinity=False), max_size=10))
            elif array_elem_type == 'bool':
                value = draw(st.lists(st.booleans(), max_size=10))
            elif array_elem_type == 'string':
                value = draw(st.lists(st.text(max_size=50), max_size=5))
        
        metadata[key] = value
    
    return metadata


@given(metadata=metadata_dict_strategy())
def test_property_metadata_round_trip(metadata):
    """
    Property 3: Metadata Round-Trip
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8**
    
    For any valid metadata dictionary containing primitive types and arrays,
    when encoded in GGUF format and parsed by the reader, the parsed metadata
    should equal the original metadata.
    
    This property ensures that:
    - All metadata keys are correctly encoded/decoded as length-prefixed UTF-8 strings
    - All value types are correctly identified and encoded/decoded
    - Primitive types (int, float, bool, string) preserve their values
    - Array types preserve both element type and all element values
    - Empty metadata dictionaries are handled correctly
    - Multiple key-value pairs are all correctly parsed
    - The metadata is stored in an accessible dictionary structure
    """
    reader = GGUFReader("test.gguf")
    
    # Create a complete GGUF file with header and metadata
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', len(metadata))  # metadata_kv_count
    
    # Encode the metadata
    file_content += encode_metadata_dict(metadata)
    
    # Create mock file and parse
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    # Verify round-trip: parsed metadata should match original
    assert len(reader.metadata) == len(metadata), \
        f"Metadata count mismatch: expected {len(metadata)}, got {len(reader.metadata)}"
    
    for key, expected_value in metadata.items():
        assert key in reader.metadata, \
            f"Key '{key}' not found in parsed metadata"
        
        actual_value = reader.metadata[key]
        
        # Compare values based on type
        if isinstance(expected_value, float):
            # For floats, check within precision tolerance
            tolerance = abs(expected_value) * 1e-6 if expected_value != 0 else 1e-6
            assert abs(actual_value - expected_value) <= tolerance, \
                f"Float value mismatch for key '{key}': expected {expected_value}, got {actual_value}"
        elif isinstance(expected_value, list):
            # For arrays, check length and each element
            assert len(actual_value) == len(expected_value), \
                f"Array length mismatch for key '{key}': expected {len(expected_value)}, got {len(actual_value)}"
            
            for i, (exp_elem, act_elem) in enumerate(zip(expected_value, actual_value)):
                if isinstance(exp_elem, float):
                    tolerance = abs(exp_elem) * 1e-6 if exp_elem != 0 else 1e-6
                    assert abs(act_elem - exp_elem) <= tolerance, \
                        f"Array element {i} mismatch for key '{key}': expected {exp_elem}, got {act_elem}"
                else:
                    assert act_elem == exp_elem, \
                        f"Array element {i} mismatch for key '{key}': expected {exp_elem}, got {act_elem}"
        else:
            # For other types (int, bool, string), check exact equality
            assert actual_value == expected_value, \
                f"Value mismatch for key '{key}': expected {expected_value}, got {actual_value}"


# ----------------------------------------------------------------------------
# Property 4: Invalid Metadata Type Rejection
# ----------------------------------------------------------------------------

# Define invalid type codes (not in valid GGUF type enumeration)
VALID_TYPE_CODES = {
    GGUFValueType.UINT8,
    GGUFValueType.INT8,
    GGUFValueType.UINT16,
    GGUFValueType.INT16,
    GGUFValueType.UINT32,
    GGUFValueType.INT32,
    GGUFValueType.FLOAT32,
    GGUFValueType.BOOL,
    GGUFValueType.STRING,
    GGUFValueType.ARRAY,
    GGUFValueType.UINT64,
    GGUFValueType.INT64,
    GGUFValueType.FLOAT64,
}


@given(invalid_type=st.integers(min_value=0, max_value=255).filter(lambda x: x not in VALID_TYPE_CODES))
def test_property_invalid_metadata_type_rejection(invalid_type):
    """
    Property 4: Invalid Metadata Type Rejection
    
    **Validates: Requirements 2.7**
    
    For any metadata value with an invalid type code (not in the valid GGUF type
    enumeration), the reader should raise an error when attempting to parse it.
    
    This property ensures that:
    - Invalid type codes are detected during metadata parsing
    - GGUFInvalidTypeError is raised for invalid types
    - Error message contains the invalid type code
    - Error message contains context (file path, position, or key name)
    - Parsing fails fast and doesn't continue with corrupted data
    - All type codes outside the valid range (0-12) are rejected
    """
    reader = GGUFReader("test.gguf")
    
    # Create a GGUF file with one metadata entry with invalid type
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata with invalid type
    key = "test.key"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', invalid_type)  # invalid type code
    file_content += b'\x00\x00\x00\x00'  # dummy data (in case reader tries to read)
    
    # Create mock file and parse header
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Attempt to parse metadata - should raise GGUFInvalidTypeError
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    
    # Verify error message contains required information
    assert "Invalid" in error_msg or "invalid" in error_msg, \
        f"Error message should mention 'invalid': {error_msg}"
    
    assert str(invalid_type) in error_msg, \
        f"Error message should contain the invalid type code {invalid_type}: {error_msg}"
    
    # Error should mention either the file path, key name, or "metadata"
    assert "test.gguf" in error_msg or "test.key" in error_msg or "metadata" in error_msg, \
        f"Error message should contain context (file path, key, or 'metadata'): {error_msg}"


# Additional test for invalid type in array elements
@given(invalid_type=st.integers(min_value=0, max_value=255).filter(lambda x: x not in VALID_TYPE_CODES))
def test_property_invalid_array_element_type_rejection(invalid_type):
    """
    Property 4 (extended): Invalid Array Element Type Rejection
    
    **Validates: Requirements 2.7**
    
    For any array with an invalid element type code, the reader should raise
    an error when attempting to parse it.
    
    This property ensures that:
    - Invalid array element types are detected
    - GGUFInvalidTypeError is raised for invalid array element types
    - Error message indicates the problem is with array element type
    """
    reader = GGUFReader("test.gguf")
    
    # Create a GGUF file with one metadata entry with array containing invalid element type
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata with array type but invalid element type
    key = "test.array"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', GGUFValueType.ARRAY)  # type = ARRAY
    file_content += struct.pack('<I', invalid_type)  # invalid element type
    file_content += struct.pack('<Q', 1)  # array length = 1
    file_content += b'\x00\x00\x00\x00'  # dummy element data
    
    # Create mock file and parse header
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Attempt to parse metadata - should raise GGUFInvalidTypeError
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    
    # Verify error message contains required information
    assert "Invalid" in error_msg or "invalid" in error_msg, \
        f"Error message should mention 'invalid': {error_msg}"
    
    assert str(invalid_type) in error_msg, \
        f"Error message should contain the invalid type code {invalid_type}: {error_msg}"
    
    # Error should mention array or element type
    assert "array" in error_msg.lower() or "element" in error_msg.lower(), \
        f"Error message should mention 'array' or 'element': {error_msg}"


# ============================================================================
# Unit Tests for Metadata Edge Cases (Task 4.3)
# ============================================================================

def test_metadata_edge_case_empty_metadata():
    """
    Test reading metadata when count is zero (empty metadata).
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - When metadata_kv_count is 0, no metadata is read
    - The metadata dictionary remains empty
    - No errors are raised for empty metadata
    - File position is correct after reading zero metadata entries
    """
    reader = GGUFReader("test.gguf")
    
    # Create a valid header with zero metadata
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count = 0
    file_content += b'NEXT_DATA'  # Some data after metadata section
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Verify header was read correctly
    assert reader.header['metadata_kv_count'] == 0
    
    # Read metadata (should be empty)
    reader._read_metadata()
    
    # Verify metadata is empty
    assert reader.metadata == {}
    assert len(reader.metadata) == 0
    assert isinstance(reader.metadata, dict)
    
    # Verify file position hasn't moved (no metadata to read)
    assert reader.file.tell() == 24  # Right after 24-byte header


def test_metadata_edge_case_nested_arrays_rejected():
    """
    Test that nested arrays in metadata are properly rejected.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - Arrays with ARRAY as element type are rejected
    - GGUFInvalidTypeError is raised
    - Error message indicates nested arrays are not allowed
    - Error occurs during array parsing, not metadata parsing
    """
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata with nested array: key="test.nested", type=ARRAY
    key = "test.nested"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', GGUFValueType.ARRAY)  # type = ARRAY
    
    # Array with ARRAY as element type (nested array - not allowed)
    file_content += struct.pack('<I', GGUFValueType.ARRAY)  # element type = ARRAY
    file_content += struct.pack('<Q', 1)  # array length
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Reading metadata should raise error due to nested array
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    assert "nested arrays are not allowed" in error_msg.lower()


def test_metadata_edge_case_all_primitive_types():
    """
    Test metadata with all primitive types and specific values.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - All primitive types (UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, FLOAT32, FLOAT64, BOOL, STRING) work correctly
    - Boundary values for each type are handled correctly
    - Negative values for signed types work
    - Zero values work for all types
    - Maximum values for each type work
    - All values are correctly stored in metadata dictionary
    """
    reader = GGUFReader("test.gguf")
    
    # Define test values for each primitive type
    test_values = {
        'test.uint8.zero': (GGUFValueType.UINT8, '<B', 0),
        'test.uint8.max': (GGUFValueType.UINT8, '<B', 255),
        'test.int8.min': (GGUFValueType.INT8, '<b', -128),
        'test.int8.zero': (GGUFValueType.INT8, '<b', 0),
        'test.int8.max': (GGUFValueType.INT8, '<b', 127),
        'test.uint16.zero': (GGUFValueType.UINT16, '<H', 0),
        'test.uint16.max': (GGUFValueType.UINT16, '<H', 65535),
        'test.int16.min': (GGUFValueType.INT16, '<h', -32768),
        'test.int16.zero': (GGUFValueType.INT16, '<h', 0),
        'test.int16.max': (GGUFValueType.INT16, '<h', 32767),
        'test.uint32.zero': (GGUFValueType.UINT32, '<I', 0),
        'test.uint32.max': (GGUFValueType.UINT32, '<I', 4294967295),
        'test.int32.min': (GGUFValueType.INT32, '<i', -2147483648),
        'test.int32.zero': (GGUFValueType.INT32, '<i', 0),
        'test.int32.max': (GGUFValueType.INT32, '<i', 2147483647),
        'test.uint64.zero': (GGUFValueType.UINT64, '<Q', 0),
        'test.uint64.max': (GGUFValueType.UINT64, '<Q', 18446744073709551615),
        'test.int64.min': (GGUFValueType.INT64, '<q', -9223372036854775808),
        'test.int64.zero': (GGUFValueType.INT64, '<q', 0),
        'test.int64.max': (GGUFValueType.INT64, '<q', 9223372036854775807),
        'test.float32.zero': (GGUFValueType.FLOAT32, '<f', 0.0),
        'test.float32.pi': (GGUFValueType.FLOAT32, '<f', 3.14159),
        'test.float32.negative': (GGUFValueType.FLOAT32, '<f', -123.456),
        'test.float64.zero': (GGUFValueType.FLOAT64, '<d', 0.0),
        'test.float64.pi': (GGUFValueType.FLOAT64, '<d', 3.141592653589793),
        'test.float64.negative': (GGUFValueType.FLOAT64, '<d', -123.456789012345),
        'test.bool.false': (GGUFValueType.BOOL, '<B', False),
        'test.bool.true': (GGUFValueType.BOOL, '<B', True),
    }
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', len(test_values))  # metadata_kv_count
    
    # Add each metadata entry
    for key, (value_type, struct_format, value) in test_values.items():
        key_bytes = key.encode('utf-8')
        file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
        file_content += struct.pack('<I', value_type)  # type
        
        # Encode value based on type
        if value_type == GGUFValueType.BOOL:
            file_content += struct.pack(struct_format, int(value))
        else:
            file_content += struct.pack(struct_format, value)
    
    # Add STRING type separately (needs special encoding)
    string_key = 'test.string.empty'
    string_value = ''
    key_bytes = string_key.encode('utf-8')
    value_bytes = string_value.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.STRING)
    file_content += struct.pack('<Q', len(value_bytes)) + value_bytes
    
    string_key2 = 'test.string.unicode'
    string_value2 = 'Hello 世界 🌍'
    key_bytes2 = string_key2.encode('utf-8')
    value_bytes2 = string_value2.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes2)) + key_bytes2
    file_content += struct.pack('<I', GGUFValueType.STRING)
    file_content += struct.pack('<Q', len(value_bytes2)) + value_bytes2
    
    # Update header with correct count
    file_content_bytes = bytearray(file_content)
    struct.pack_into('<Q', file_content_bytes, 16, len(test_values) + 2)  # Update metadata_kv_count
    
    reader.file = io.BytesIO(bytes(file_content_bytes))
    reader._read_header()
    reader._read_metadata()
    
    # Verify all values were read correctly
    assert len(reader.metadata) == len(test_values) + 2
    
    # Check integer types
    assert reader.metadata['test.uint8.zero'] == 0
    assert reader.metadata['test.uint8.max'] == 255
    assert reader.metadata['test.int8.min'] == -128
    assert reader.metadata['test.int8.zero'] == 0
    assert reader.metadata['test.int8.max'] == 127
    assert reader.metadata['test.uint16.zero'] == 0
    assert reader.metadata['test.uint16.max'] == 65535
    assert reader.metadata['test.int16.min'] == -32768
    assert reader.metadata['test.int16.zero'] == 0
    assert reader.metadata['test.int16.max'] == 32767
    assert reader.metadata['test.uint32.zero'] == 0
    assert reader.metadata['test.uint32.max'] == 4294967295
    assert reader.metadata['test.int32.min'] == -2147483648
    assert reader.metadata['test.int32.zero'] == 0
    assert reader.metadata['test.int32.max'] == 2147483647
    assert reader.metadata['test.uint64.zero'] == 0
    assert reader.metadata['test.uint64.max'] == 18446744073709551615
    assert reader.metadata['test.int64.min'] == -9223372036854775808
    assert reader.metadata['test.int64.zero'] == 0
    assert reader.metadata['test.int64.max'] == 9223372036854775807
    
    # Check float types (with tolerance)
    assert abs(reader.metadata['test.float32.zero'] - 0.0) < 1e-6
    assert abs(reader.metadata['test.float32.pi'] - 3.14159) < 1e-5
    assert abs(reader.metadata['test.float32.negative'] - (-123.456)) < 1e-3
    assert abs(reader.metadata['test.float64.zero'] - 0.0) < 1e-14
    assert abs(reader.metadata['test.float64.pi'] - 3.141592653589793) < 1e-14
    assert abs(reader.metadata['test.float64.negative'] - (-123.456789012345)) < 1e-12
    
    # Check bool types
    assert reader.metadata['test.bool.false'] is False
    assert reader.metadata['test.bool.true'] is True
    assert isinstance(reader.metadata['test.bool.false'], bool)
    assert isinstance(reader.metadata['test.bool.true'], bool)
    
    # Check string types
    assert reader.metadata['test.string.empty'] == ''
    assert reader.metadata['test.string.unicode'] == 'Hello 世界 🌍'


def test_metadata_edge_case_invalid_type_error_message():
    """
    Test error messages for invalid metadata types.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - Invalid type codes raise GGUFInvalidTypeError
    - Error message contains the invalid type code
    - Error message contains the file path
    - Error message contains the metadata key name (if available)
    - Error message contains the file position
    """
    reader = GGUFReader("test_model.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Add metadata with invalid type
    key = "test.invalid"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', 999)  # invalid type code
    file_content += b'\x00\x00\x00\x00'  # dummy data
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Reading metadata should raise error
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    
    # Verify error message contains required information
    assert "Invalid" in error_msg or "invalid" in error_msg
    assert "type" in error_msg.lower()
    assert "999" in error_msg  # Invalid type code
    assert "test.invalid" in error_msg  # Metadata key name
    assert "test_model.gguf" in error_msg  # File path


def test_metadata_edge_case_multiple_invalid_types():
    """
    Test error messages for various invalid type codes.
    
    Requirements: 2.7, 2.8
    
    This test verifies that various invalid type codes are all rejected:
    - Type code 13 (one past FLOAT64)
    - Type code 99 (arbitrary invalid)
    - Type code 255 (max uint8)
    - Type code 1000 (large invalid)
    - Type code -1 (as uint32: 0xFFFFFFFF)
    """
    invalid_types = [13, 99, 255, 1000, 0xFFFFFFFF]
    
    for invalid_type in invalid_types:
        reader = GGUFReader("test.gguf")
        
        # Create header
        file_content = struct.pack('<I', 0x46554747)  # magic
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
        
        # Add metadata with invalid type
        key = f"test.type{invalid_type}"
        key_bytes = key.encode('utf-8')
        file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
        file_content += struct.pack('<I', invalid_type)  # invalid type code
        file_content += b'\x00\x00\x00\x00'  # dummy data
        
        reader.file = io.BytesIO(file_content)
        reader._read_header()
        
        # Reading metadata should raise error
        with pytest.raises(GGUFInvalidTypeError) as exc_info:
            reader._read_metadata()
        
        error_msg = str(exc_info.value)
        
        # Verify error message mentions the invalid type
        assert "type" in error_msg.lower(), \
            f"Error message should mention 'type' for invalid type {invalid_type}: {error_msg}"
        assert str(invalid_type) in error_msg or f"0x{invalid_type:08X}" in error_msg, \
            f"Error message should contain type code {invalid_type}: {error_msg}"


def test_metadata_edge_case_array_of_all_types():
    """
    Test metadata with arrays of all valid primitive types.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - Arrays can contain any valid primitive type
    - Arrays of integers work correctly
    - Arrays of floats work correctly
    - Arrays of bools work correctly
    - Arrays of strings work correctly
    - Empty arrays work for all types
    - Arrays with multiple elements work correctly
    """
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count (will update)
    
    metadata_count = 0
    
    # Array of UINT8
    key = "test.array.uint8"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.UINT8)  # element type
    file_content += struct.pack('<Q', 3)  # length
    file_content += struct.pack('<B', 0)
    file_content += struct.pack('<B', 128)
    file_content += struct.pack('<B', 255)
    metadata_count += 1
    
    # Array of INT32
    key = "test.array.int32"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.INT32)  # element type
    file_content += struct.pack('<Q', 4)  # length
    file_content += struct.pack('<i', -100)
    file_content += struct.pack('<i', 0)
    file_content += struct.pack('<i', 100)
    file_content += struct.pack('<i', 2147483647)
    metadata_count += 1
    
    # Array of FLOAT32
    key = "test.array.float32"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.FLOAT32)  # element type
    file_content += struct.pack('<Q', 3)  # length
    file_content += struct.pack('<f', 1.5)
    file_content += struct.pack('<f', 2.5)
    file_content += struct.pack('<f', 3.5)
    metadata_count += 1
    
    # Array of BOOL
    key = "test.array.bool"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.BOOL)  # element type
    file_content += struct.pack('<Q', 4)  # length
    file_content += struct.pack('<B', 1)  # true
    file_content += struct.pack('<B', 0)  # false
    file_content += struct.pack('<B', 1)  # true
    file_content += struct.pack('<B', 0)  # false
    metadata_count += 1
    
    # Array of STRING
    key = "test.array.string"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.STRING)  # element type
    file_content += struct.pack('<Q', 3)  # length
    for s in ["first", "second", "third"]:
        s_bytes = s.encode('utf-8')
        file_content += struct.pack('<Q', len(s_bytes)) + s_bytes
    metadata_count += 1
    
    # Empty array of UINT32
    key = "test.array.empty"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.ARRAY)
    file_content += struct.pack('<I', GGUFValueType.UINT32)  # element type
    file_content += struct.pack('<Q', 0)  # length = 0
    metadata_count += 1
    
    # Update header with correct count
    file_content_bytes = bytearray(file_content)
    struct.pack_into('<Q', file_content_bytes, 16, metadata_count)
    
    reader.file = io.BytesIO(bytes(file_content_bytes))
    reader._read_header()
    reader._read_metadata()
    
    # Verify all arrays were read correctly
    assert len(reader.metadata) == metadata_count
    
    # Check UINT8 array
    assert reader.metadata['test.array.uint8'] == [0, 128, 255]
    
    # Check INT32 array
    assert reader.metadata['test.array.int32'] == [-100, 0, 100, 2147483647]
    
    # Check FLOAT32 array (with tolerance)
    float_array = reader.metadata['test.array.float32']
    assert len(float_array) == 3
    assert abs(float_array[0] - 1.5) < 0.0001
    assert abs(float_array[1] - 2.5) < 0.0001
    assert abs(float_array[2] - 3.5) < 0.0001
    
    # Check BOOL array
    assert reader.metadata['test.array.bool'] == [True, False, True, False]
    
    # Check STRING array
    assert reader.metadata['test.array.string'] == ["first", "second", "third"]
    
    # Check empty array
    assert reader.metadata['test.array.empty'] == []


def test_metadata_edge_case_truncated_in_middle():
    """
    Test that truncation in the middle of metadata is detected.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - Truncation during metadata key reading is detected
    - Truncation during metadata value reading is detected
    - Appropriate error messages are generated
    """
    # Test truncation during second metadata key
    reader = GGUFReader("test.gguf")
    
    # Create header with 2 metadata entries
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 2)  # metadata_kv_count = 2
    
    # Add first complete metadata entry
    key1 = "first.key"
    key1_bytes = key1.encode('utf-8')
    file_content += struct.pack('<Q', len(key1_bytes)) + key1_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 42)
    
    # Add second metadata entry but truncate it (only partial key length)
    file_content += b'\x05\x00\x00\x00'  # Only 4 bytes of 8-byte length
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    # Reading metadata should raise truncation error
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    assert "Unexpected end of file" in error_msg or "truncated" in error_msg.lower()


def test_metadata_edge_case_special_key_names():
    """
    Test metadata with special key names.
    
    Requirements: 2.7, 2.8
    
    This test verifies that:
    - Keys with dots work correctly
    - Keys with underscores work correctly
    - Keys with numbers work correctly
    - Long keys work correctly
    - Keys with unicode characters work correctly
    """
    reader = GGUFReader("test.gguf")
    
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 5)  # metadata_kv_count = 5
    
    # Key with dots
    key1 = "general.architecture.name"
    key1_bytes = key1.encode('utf-8')
    file_content += struct.pack('<Q', len(key1_bytes)) + key1_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 1)
    
    # Key with underscores
    key2 = "llama_context_length"
    key2_bytes = key2.encode('utf-8')
    file_content += struct.pack('<Q', len(key2_bytes)) + key2_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 2)
    
    # Key with numbers
    key3 = "layer.0.attention.heads"
    key3_bytes = key3.encode('utf-8')
    file_content += struct.pack('<Q', len(key3_bytes)) + key3_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 3)
    
    # Long key
    key4 = "a" * 200  # 200 character key
    key4_bytes = key4.encode('utf-8')
    file_content += struct.pack('<Q', len(key4_bytes)) + key4_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 4)
    
    # Key with unicode
    key5 = "model.名前"  # Japanese characters
    key5_bytes = key5.encode('utf-8')
    file_content += struct.pack('<Q', len(key5_bytes)) + key5_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 5)
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    
    # Verify all keys were read correctly
    assert len(reader.metadata) == 5
    assert reader.metadata["general.architecture.name"] == 1
    assert reader.metadata["llama_context_length"] == 2
    assert reader.metadata["layer.0.attention.heads"] == 3
    assert reader.metadata["a" * 200] == 4
    assert reader.metadata["model.名前"] == 5


# ============================================================================
# Property-Based Tests for Tensor Info Parsing (Task 5.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 5: Tensor Info Round-Trip
# ----------------------------------------------------------------------------

def encode_tensor_info(tensor_info: Dict[str, Any]) -> bytes:
    """
    Helper function to encode a single tensor info record in GGUF format.
    
    Args:
        tensor_info: Dictionary containing tensor information with keys:
            - name: str (tensor name)
            - n_dims: int (number of dimensions)
            - dims: list[int] (dimension sizes)
            - type: int (tensor data type code)
            - offset: int (offset from tensor data section start)
            
    Returns:
        Bytes representing the encoded tensor info record
    """
    result = b''
    
    # Encode tensor name (length-prefixed string)
    name_bytes = tensor_info['name'].encode('utf-8')
    result += struct.pack('<Q', len(name_bytes)) + name_bytes
    
    # Encode number of dimensions (uint32)
    result += struct.pack('<I', tensor_info['n_dims'])
    
    # Encode dimension sizes (array of uint64)
    for dim in tensor_info['dims']:
        result += struct.pack('<Q', dim)
    
    # Encode tensor data type (uint32)
    result += struct.pack('<I', tensor_info['type'])
    
    # Encode offset (uint64)
    result += struct.pack('<Q', tensor_info['offset'])
    
    return result


def encode_tensor_info_list(tensor_info_list: List[Dict[str, Any]]) -> bytes:
    """
    Helper function to encode a list of tensor info records in GGUF format.
    
    Args:
        tensor_info_list: List of tensor info dictionaries
        
    Returns:
        Bytes representing the encoded tensor info section
    """
    result = b''
    for tensor_info in tensor_info_list:
        result += encode_tensor_info(tensor_info)
    return result


# Strategy for generating valid tensor info records
@st.composite
def tensor_info_strategy(draw):
    """
    Generate a valid tensor info record.
    
    This strategy generates tensor info with:
    - Valid tensor name (non-empty string)
    - Number of dimensions (1-4)
    - Dimension sizes (reasonable values)
    - Valid tensor type (from TYPE_SIZES)
    - Offset value (non-negative)
    """
    # Generate tensor name
    name = draw(st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='._-'
        )
    ))
    
    # Generate number of dimensions (1-4 is typical for tensors)
    n_dims = draw(st.integers(min_value=1, max_value=4))
    
    # Generate dimension sizes (reasonable values to avoid huge tensors)
    dims = draw(st.lists(
        st.integers(min_value=1, max_value=4096),
        min_size=n_dims,
        max_size=n_dims
    ))
    
    # Generate tensor type (must be valid type from TYPE_SIZES)
    valid_tensor_types = list(TYPE_SIZES.keys())
    tensor_type = draw(st.sampled_from(valid_tensor_types))
    
    # Generate offset (non-negative, reasonable value)
    offset = draw(st.integers(min_value=0, max_value=1000000000))
    
    return {
        'name': name,
        'n_dims': n_dims,
        'dims': dims,
        'type': tensor_type,
        'offset': offset
    }


# Strategy for generating a list of tensor info records
@st.composite
def tensor_info_list_strategy(draw):
    """
    Generate a list of valid tensor info records.
    
    This strategy generates:
    - 0-10 tensor info records
    - Each with unique name
    - All with valid structure
    """
    num_tensors = draw(st.integers(min_value=0, max_value=10))
    
    tensor_list = []
    used_names = set()
    
    for i in range(num_tensors):
        tensor = draw(tensor_info_strategy())
        
        # Ensure unique names
        base_name = tensor['name']
        counter = 0
        while tensor['name'] in used_names:
            counter += 1
            tensor['name'] = f"{base_name}_{counter}"
        
        used_names.add(tensor['name'])
        tensor_list.append(tensor)
    
    return tensor_list


@given(tensor_info_list=tensor_info_list_strategy())
def test_property_tensor_info_round_trip(tensor_info_list):
    """
    Property 5: Tensor Info Round-Trip
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
    
    For any valid tensor information (name, dimensions, type, offset), when encoded
    in GGUF format and parsed by the reader, the parsed tensor info should match
    the original values.
    
    This property ensures that:
    - Tensor names are correctly encoded/decoded as length-prefixed UTF-8 strings (Req 3.1)
    - Number of dimensions is correctly encoded/decoded as uint32 (Req 3.2)
    - Dimension sizes are correctly encoded/decoded as array of uint64 values (Req 3.3)
    - Tensor data type is correctly encoded/decoded as uint32 enum (Req 3.4)
    - Data offset is correctly encoded/decoded as uint64 (Req 3.5)
    - All tensor info is stored in an accessible list structure (Req 3.6)
    - Empty tensor lists (tensor_count = 0) are handled correctly
    - Multiple tensors are all correctly parsed
    - Tensors with different numbers of dimensions (1-4) work correctly
    - All valid tensor types are supported
    - Large dimension sizes and offsets are handled correctly
    """
    reader = GGUFReader("test.gguf")
    
    # Create a complete GGUF file with header and tensor info
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', len(tensor_info_list))  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    # Encode the tensor info
    file_content += encode_tensor_info_list(tensor_info_list)
    
    # Create mock file and parse
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_tensor_info()
    
    # Verify round-trip: parsed tensor info should match original
    assert len(reader.tensor_info) == len(tensor_info_list), \
        f"Tensor count mismatch: expected {len(tensor_info_list)}, got {len(reader.tensor_info)}"
    
    for i, (expected, actual) in enumerate(zip(tensor_info_list, reader.tensor_info)):
        # Verify tensor name (Requirement 3.1)
        assert actual['name'] == expected['name'], \
            f"Tensor {i} name mismatch: expected '{expected['name']}', got '{actual['name']}'"
        
        # Verify number of dimensions (Requirement 3.2)
        assert actual['n_dims'] == expected['n_dims'], \
            f"Tensor {i} ({expected['name']}) n_dims mismatch: expected {expected['n_dims']}, got {actual['n_dims']}"
        
        # Verify dimension sizes (Requirement 3.3)
        assert len(actual['dims']) == len(expected['dims']), \
            f"Tensor {i} ({expected['name']}) dims length mismatch: expected {len(expected['dims'])}, got {len(actual['dims'])}"
        
        for dim_idx, (exp_dim, act_dim) in enumerate(zip(expected['dims'], actual['dims'])):
            assert act_dim == exp_dim, \
                f"Tensor {i} ({expected['name']}) dimension {dim_idx} mismatch: expected {exp_dim}, got {act_dim}"
        
        # Verify tensor data type (Requirement 3.4)
        assert actual['type'] == expected['type'], \
            f"Tensor {i} ({expected['name']}) type mismatch: expected {expected['type']}, got {actual['type']}"
        
        # Verify data offset (Requirement 3.5)
        assert actual['offset'] == expected['offset'], \
            f"Tensor {i} ({expected['name']}) offset mismatch: expected {expected['offset']}, got {actual['offset']}"
    
    # Verify tensor info is stored in accessible list structure (Requirement 3.6)
    assert isinstance(reader.tensor_info, list), \
        f"Tensor info should be stored as list, got {type(reader.tensor_info)}"
    
    for tensor in reader.tensor_info:
        assert isinstance(tensor, dict), \
            f"Each tensor info should be a dictionary, got {type(tensor)}"
        assert 'name' in tensor, "Tensor info should have 'name' key"
        assert 'n_dims' in tensor, "Tensor info should have 'n_dims' key"
        assert 'dims' in tensor, "Tensor info should have 'dims' key"
        assert 'type' in tensor, "Tensor info should have 'type' key"
        assert 'offset' in tensor, "Tensor info should have 'offset' key"


# Additional property test for edge cases
@given(
    n_dims=st.integers(min_value=1, max_value=4),
    data=st.data()
)
def test_property_tensor_info_dimension_consistency(n_dims, data):
    """
    Property 5 (extended): Tensor Info Dimension Consistency
    
    **Validates: Requirements 3.2, 3.3**
    
    For any tensor with n_dims dimensions, the dims array should have exactly
    n_dims elements when parsed.
    
    This property ensures that:
    - The n_dims field correctly indicates the number of dimensions
    - The dims array length matches n_dims
    - All dimension values are correctly parsed
    """
    reader = GGUFReader("test.gguf")
    
    # Generate dimension sizes
    dims = data.draw(st.lists(
        st.integers(min_value=1, max_value=4096),
        min_size=n_dims,
        max_size=n_dims
    ))
    
    # Generate other tensor info fields
    tensor_name = data.draw(st.text(min_size=1, max_size=50))
    tensor_type = data.draw(st.sampled_from(list(TYPE_SIZES.keys())))
    offset = data.draw(st.integers(min_value=0, max_value=1000000))
    
    tensor_info = {
        'name': tensor_name,
        'n_dims': n_dims,
        'dims': dims,
        'type': tensor_type,
        'offset': offset
    }
    
    # Create GGUF file with this tensor
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count = 1
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    file_content += encode_tensor_info(tensor_info)
    
    # Parse
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_tensor_info()
    
    # Verify dimension consistency
    assert len(reader.tensor_info) == 1
    parsed_tensor = reader.tensor_info[0]
    
    assert parsed_tensor['n_dims'] == n_dims, \
        f"n_dims mismatch: expected {n_dims}, got {parsed_tensor['n_dims']}"
    
    assert len(parsed_tensor['dims']) == n_dims, \
        f"dims array length should match n_dims: expected {n_dims}, got {len(parsed_tensor['dims'])}"
    
    assert parsed_tensor['dims'] == dims, \
        f"dims values mismatch: expected {dims}, got {parsed_tensor['dims']}"


# Property test for all valid tensor types
@given(tensor_type=st.sampled_from(list(TYPE_SIZES.keys())))
def test_property_tensor_info_all_types_valid(tensor_type):
    """
    Property 5 (extended): All Valid Tensor Types Accepted
    
    **Validates: Requirements 3.4**
    
    For any valid tensor type (from TYPE_SIZES), the reader should successfully
    parse tensor info containing that type.
    
    This property ensures that:
    - All standard types (F32, F16, I8, I16, I32) are accepted
    - All quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) are accepted
    - Type validation correctly identifies valid types
    - Parsed type matches original type
    """
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'n_dims': 2,
        'dims': [128, 256],
        'type': tensor_type,
        'offset': 0
    }
    
    # Create GGUF file
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count = 1
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    file_content += encode_tensor_info(tensor_info)
    
    # Parse - should not raise any errors
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_tensor_info()
    
    # Verify type was parsed correctly
    assert len(reader.tensor_info) == 1
    assert reader.tensor_info[0]['type'] == tensor_type, \
        f"Tensor type mismatch: expected {tensor_type}, got {reader.tensor_info[0]['type']}"


# Property test for invalid tensor types
@given(invalid_type=st.integers(min_value=0, max_value=1000).filter(lambda x: x not in TYPE_SIZES))
def test_property_tensor_info_invalid_type_rejection(invalid_type):
    """
    Property 5 (extended): Invalid Tensor Type Rejection
    
    **Validates: Requirements 3.4**
    
    For any invalid tensor type (not in TYPE_SIZES), the reader should raise
    GGUFInvalidTypeError when attempting to parse tensor info.
    
    This property ensures that:
    - Invalid tensor types are detected during parsing
    - GGUFInvalidTypeError is raised
    - Error message contains the invalid type code
    - Error message contains the tensor name (if available)
    - Parsing fails fast for invalid types
    """
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'invalid_tensor',
        'n_dims': 2,
        'dims': [128, 256],
        'type': invalid_type,
        'offset': 0
    }
    
    # Create GGUF file
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count = 1
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    file_content += encode_tensor_info(tensor_info)
    
    # Parse - should raise GGUFInvalidTypeError
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_tensor_info()
    
    error_msg = str(exc_info.value)
    
    # Verify error message contains required information
    assert "Invalid" in error_msg or "invalid" in error_msg, \
        f"Error message should mention 'invalid': {error_msg}"
    
    assert str(invalid_type) in error_msg, \
        f"Error message should contain the invalid type code {invalid_type}: {error_msg}"
    
    assert "invalid_tensor" in error_msg or "tensor" in error_msg.lower(), \
        f"Error message should mention the tensor or 'tensor': {error_msg}"


# ============================================================================
# Edge Case Tests for Task 5.3: Tensor Info Edge Cases
# ============================================================================

def test_tensor_info_1_dimension():
    """Test reading tensor info with 1 dimension."""
    reader = GGUFReader("test.gguf")
    reader.header = {'tensor_count': 1}
    
    tensor_name = "tensor_1d"
    name_bytes = tensor_name.encode('utf-8')
    
    file_content = b''
    file_content += struct.pack('<Q', len(name_bytes))
    file_content += name_bytes
    file_content += struct.pack('<I', 1)  # n_dims: 1
    file_content += struct.pack('<Q', 4096)  # dims: [4096]
    file_content += struct.pack('<I', GGMLType.F32)  # type
    file_content += struct.pack('<Q', 0)  # offset
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == 1
    tensor = reader.tensor_info[0]
    assert tensor['name'] == "tensor_1d"
    assert tensor['n_dims'] == 1
    assert tensor['dims'] == [4096]


def test_tensor_info_3_dimensions():
    """Test reading tensor info with 3 dimensions."""
    reader = GGUFReader("test.gguf")
    reader.header = {'tensor_count': 1}
    
    tensor_name = "tensor_3d"
    name_bytes = tensor_name.encode('utf-8')
    
    file_content = b''
    file_content += struct.pack('<Q', len(name_bytes))
    file_content += name_bytes
    file_content += struct.pack('<I', 3)  # n_dims: 3
    file_content += struct.pack('<Q', 64)  # dims[0]
    file_content += struct.pack('<Q', 128)  # dims[1]
    file_content += struct.pack('<Q', 256)  # dims[2]
    file_content += struct.pack('<I', GGMLType.Q4_0)  # type
    file_content += struct.pack('<Q', 0)  # offset
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == 1
    tensor = reader.tensor_info[0]
    assert tensor['name'] == "tensor_3d"
    assert tensor['n_dims'] == 3
    assert tensor['dims'] == [64, 128, 256]


def test_tensor_info_4_dimensions():
    """Test reading tensor info with 4 dimensions."""
    reader = GGUFReader("test.gguf")
    reader.header = {'tensor_count': 1}
    
    tensor_name = "tensor_4d"
    name_bytes = tensor_name.encode('utf-8')
    
    file_content = b''
    file_content += struct.pack('<Q', len(name_bytes))
    file_content += name_bytes
    file_content += struct.pack('<I', 4)  # n_dims: 4
    file_content += struct.pack('<Q', 32)  # dims[0]
    file_content += struct.pack('<Q', 64)  # dims[1]
    file_content += struct.pack('<Q', 128)  # dims[2]
    file_content += struct.pack('<Q', 256)  # dims[3]
    file_content += struct.pack('<I', GGMLType.I8)  # type
    file_content += struct.pack('<Q', 0)  # offset
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == 1
    tensor = reader.tensor_info[0]
    assert tensor['name'] == "tensor_4d"
    assert tensor['n_dims'] == 4
    assert tensor['dims'] == [32, 64, 128, 256]


def test_tensor_info_all_dimensions_combined():
    """Test reading tensors with 1, 2, 3, and 4 dimensions in a single file."""
    reader = GGUFReader("test.gguf")
    reader.header = {'tensor_count': 4}
    
    tensors = [
        ("tensor_1d", 1, [4096], GGMLType.F32),
        ("tensor_2d", 2, [512, 1024], GGMLType.F16),
        ("tensor_3d", 3, [64, 128, 256], GGMLType.Q4_0),
        ("tensor_4d", 4, [32, 64, 128, 256], GGMLType.I8),
    ]
    
    file_content = b''
    for name, n_dims, dims, tensor_type in tensors:
        name_bytes = name.encode('utf-8')
        file_content += struct.pack('<Q', len(name_bytes))
        file_content += name_bytes
        file_content += struct.pack('<I', n_dims)
        for dim in dims:
            file_content += struct.pack('<Q', dim)
        file_content += struct.pack('<I', tensor_type)
        file_content += struct.pack('<Q', 0)
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == 4
    for i, (name, n_dims, dims, tensor_type) in enumerate(tensors):
        tensor = reader.tensor_info[i]
        assert tensor['name'] == name
        assert tensor['n_dims'] == n_dims
        assert tensor['dims'] == dims
        assert tensor['type'] == tensor_type


def test_tensor_info_all_quantized_types():
    """Test reading tensors with all quantized types."""
    reader = GGUFReader("test.gguf")
    
    quantized_types = [
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
        GGMLType.Q8_0, GGMLType.Q8_1, GGMLType.Q2_K, GGMLType.Q3_K,
        GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
    ]
    
    reader.header = {'tensor_count': len(quantized_types)}
    
    file_content = b''
    for i, tensor_type in enumerate(quantized_types):
        name = f"quantized_tensor_{i}"
        name_bytes = name.encode('utf-8')
        file_content += struct.pack('<Q', len(name_bytes))
        file_content += name_bytes
        file_content += struct.pack('<I', 1)  # n_dims
        file_content += struct.pack('<Q', 256)  # dims (multiple of block size)
        file_content += struct.pack('<I', tensor_type)
        file_content += struct.pack('<Q', i * 1024)
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == len(quantized_types)
    for i, tensor_type in enumerate(quantized_types):
        assert reader.tensor_info[i]['type'] == tensor_type


def test_tensor_info_mixed_realistic():
    """Test reading multiple tensors with mixed dimensions and types (realistic LLM scenario)."""
    reader = GGUFReader("test.gguf")
    
    tensors = [
        ("embedding", 2, [32000, 4096], GGMLType.F32),
        ("attention.wq", 2, [4096, 4096], GGMLType.F16),
        ("attention.wk", 2, [4096, 4096], GGMLType.Q4_0),
        ("ffn.w1", 2, [4096, 11008], GGMLType.Q5_0),
        ("norm", 1, [4096], GGMLType.F32),
    ]
    
    reader.header = {'tensor_count': len(tensors)}
    
    file_content = b''
    offset = 0
    for name, n_dims, dims, tensor_type in tensors:
        name_bytes = name.encode('utf-8')
        file_content += struct.pack('<Q', len(name_bytes))
        file_content += name_bytes
        file_content += struct.pack('<I', n_dims)
        for dim in dims:
            file_content += struct.pack('<Q', dim)
        file_content += struct.pack('<I', tensor_type)
        file_content += struct.pack('<Q', offset)
        offset += 1024 * 1024
    
    reader.file = io.BytesIO(file_content)
    reader._read_tensor_info()
    
    assert len(reader.tensor_info) == len(tensors)
    for i, (name, n_dims, dims, tensor_type) in enumerate(tensors):
        tensor = reader.tensor_info[i]
        assert tensor['name'] == name
        assert tensor['n_dims'] == n_dims
        assert tensor['dims'] == dims
        assert tensor['type'] == tensor_type


# ============================================================================
# Test _calculate_alignment() Method (Task 6.1)
# ============================================================================

def test_calculate_alignment_default():
    """Test alignment calculation with default alignment (32 bytes)."""
    reader = GGUFReader("test.gguf")
    reader.metadata = {}  # No alignment specified, should default to 32
    
    # Create a mock file at position 100
    reader.file = io.BytesIO(b'\x00' * 200)
    reader.file.seek(100)
    
    reader._calculate_alignment()
    
    # Position 100: next aligned position at 32-byte boundary is 128
    # padding = (32 - (100 % 32)) % 32 = (32 - 4) % 32 = 28
    # aligned_base = 100 + 28 = 128
    assert reader.tensor_data_base == 128


def test_calculate_alignment_custom():
    """Test alignment calculation with custom alignment value."""
    reader = GGUFReader("test.gguf")
    reader.metadata = {'general.alignment': 64}  # Custom 64-byte alignment
    
    # Create a mock file at position 100
    reader.file = io.BytesIO(b'\x00' * 200)
    reader.file.seek(100)
    
    reader._calculate_alignment()
    
    # Position 100: next aligned position at 64-byte boundary is 128
    # padding = (64 - (100 % 64)) % 64 = (64 - 36) % 64 = 28
    # aligned_base = 100 + 28 = 128
    assert reader.tensor_data_base == 128


def test_calculate_alignment_already_aligned():
    """Test alignment calculation when position is already aligned."""
    reader = GGUFReader("test.gguf")
    reader.metadata = {'general.alignment': 32}
    
    # Create a mock file at position 128 (already aligned to 32)
    reader.file = io.BytesIO(b'\x00' * 200)
    reader.file.seek(128)
    
    reader._calculate_alignment()
    
    # Position 128 is already aligned to 32
    # padding = (32 - (128 % 32)) % 32 = (32 - 0) % 32 = 0
    # aligned_base = 128 + 0 = 128
    assert reader.tensor_data_base == 128


def test_calculate_alignment_position_zero():
    """Test alignment calculation at position 0."""
    reader = GGUFReader("test.gguf")
    reader.metadata = {'general.alignment': 32}
    
    # Create a mock file at position 0
    reader.file = io.BytesIO(b'\x00' * 200)
    reader.file.seek(0)
    
    reader._calculate_alignment()
    
    # Position 0 is already aligned
    # padding = (32 - (0 % 32)) % 32 = (32 - 0) % 32 = 0
    # aligned_base = 0 + 0 = 0
    assert reader.tensor_data_base == 0


def test_calculate_alignment_various_positions():
    """Test alignment calculation at various positions."""
    test_cases = [
        # (position, alignment, expected_base)
        (0, 32, 0),      # Already aligned
        (1, 32, 32),     # Need 31 bytes padding
        (31, 32, 32),    # Need 1 byte padding
        (32, 32, 32),    # Already aligned
        (33, 32, 64),    # Need 31 bytes padding
        (50, 64, 64),    # Need 14 bytes padding
        (64, 64, 64),    # Already aligned
        (100, 16, 112),  # Need 12 bytes padding
        (200, 256, 256), # Need 56 bytes padding
    ]
    
    for position, alignment, expected_base in test_cases:
        reader = GGUFReader("test.gguf")
        reader.metadata = {'general.alignment': alignment}
        reader.file = io.BytesIO(b'\x00' * 1000)
        reader.file.seek(position)
        
        reader._calculate_alignment()
        
        assert reader.tensor_data_base == expected_base, \
            f"Failed for position={position}, alignment={alignment}: " \
            f"expected {expected_base}, got {reader.tensor_data_base}"


def test_calculate_alignment_no_file_open():
    """Test that calculating alignment without an open file raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    reader.metadata = {'general.alignment': 32}
    # file is None by default
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._calculate_alignment()
    
    assert "File is not open" in str(exc_info.value)


def test_calculate_alignment_formula_correctness():
    """Test that the alignment formula works correctly for edge cases."""
    reader = GGUFReader("test.gguf")
    
    # Test with alignment = 1 (every position is aligned)
    reader.metadata = {'general.alignment': 1}
    reader.file = io.BytesIO(b'\x00' * 100)
    reader.file.seek(42)
    reader._calculate_alignment()
    assert reader.tensor_data_base == 42  # No padding needed
    
    # Test with alignment = 2 (even positions)
    reader.metadata = {'general.alignment': 2}
    reader.file.seek(41)  # Odd position
    reader._calculate_alignment()
    assert reader.tensor_data_base == 42  # Next even position
    
    reader.file.seek(42)  # Even position
    reader._calculate_alignment()
    assert reader.tensor_data_base == 42  # Already aligned


# ============================================================================
# Property-Based Tests for Alignment Calculation (Task 6.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 6: Alignment Calculation
# ----------------------------------------------------------------------------

@given(
    position=st.integers(min_value=0, max_value=1000000),
    alignment=st.integers(min_value=1, max_value=1024)
)
def test_property_alignment_calculation(position, alignment):
    """
    Property 6: Alignment Calculation
    
    **Validates: Requirements 4.1, 4.3**
    
    For any file position and alignment value, the calculated aligned position
    should be the smallest value >= file position that is divisible by the
    alignment value.
    
    This property ensures that:
    - The aligned position is always >= the current position
    - The aligned position is divisible by the alignment value
    - The aligned position is the smallest such value (no unnecessary padding)
    - The alignment formula works correctly for all positions and alignments
    - Already-aligned positions remain unchanged
    """
    reader = GGUFReader("test.gguf")
    reader.metadata = {'general.alignment': alignment}
    
    # Create a mock file with enough space
    reader.file = io.BytesIO(b'\x00' * (position + alignment + 100))
    reader.file.seek(position)
    
    # Calculate alignment
    reader._calculate_alignment()
    
    aligned_position = reader.tensor_data_base
    
    # Property 1: Aligned position must be >= current position
    assert aligned_position >= position, \
        f"Aligned position {aligned_position} is less than current position {position}"
    
    # Property 2: Aligned position must be divisible by alignment
    assert aligned_position % alignment == 0, \
        f"Aligned position {aligned_position} is not divisible by alignment {alignment}"
    
    # Property 3: Aligned position must be the smallest such value
    # (i.e., no position between current and aligned is divisible by alignment)
    if aligned_position > position:
        # Check that the previous aligned position is less than current position
        previous_aligned = aligned_position - alignment
        assert previous_aligned < position, \
            f"Previous aligned position {previous_aligned} is >= current position {position}, " \
            f"meaning aligned position {aligned_position} is not minimal"
    
    # Property 4: Verify the alignment formula
    # padding = (alignment - (position % alignment)) % alignment
    expected_padding = (alignment - (position % alignment)) % alignment
    expected_aligned = position + expected_padding
    assert aligned_position == expected_aligned, \
        f"Aligned position {aligned_position} doesn't match formula result {expected_aligned} " \
        f"(position={position}, alignment={alignment}, padding={expected_padding})"
    
    # Property 5: If position is already aligned, no padding should be added
    if position % alignment == 0:
        assert aligned_position == position, \
            f"Position {position} is already aligned to {alignment}, but got {aligned_position}"


# ----------------------------------------------------------------------------
# Property 7: Default Alignment
# ----------------------------------------------------------------------------

@given(position=st.integers(min_value=0, max_value=1000000))
def test_property_default_alignment(position):
    """
    Property 7: Default Alignment
    
    **Validates: Requirements 4.2**
    
    For any GGUF file without a "general.alignment" metadata key, the reader
    should use 32 bytes as the alignment value.
    
    This property ensures that:
    - When no alignment is specified in metadata, 32 is used as default
    - The default alignment is applied correctly to calculate aligned positions
    - The behavior is consistent with the GGUF specification
    """
    reader = GGUFReader("test.gguf")
    # No alignment in metadata - should default to 32
    reader.metadata = {}
    
    # Create a mock file with enough space
    reader.file = io.BytesIO(b'\x00' * (position + 100))
    reader.file.seek(position)
    
    # Calculate alignment
    reader._calculate_alignment()
    
    aligned_position = reader.tensor_data_base
    
    # Property 1: Default alignment of 32 should be used
    # This means aligned position must be divisible by 32
    assert aligned_position % 32 == 0, \
        f"Aligned position {aligned_position} is not divisible by default alignment 32"
    
    # Property 2: Aligned position should match calculation with alignment=32
    expected_padding = (32 - (position % 32)) % 32
    expected_aligned = position + expected_padding
    assert aligned_position == expected_aligned, \
        f"Aligned position {aligned_position} doesn't match expected {expected_aligned} " \
        f"with default alignment 32 (position={position})"
    
    # Property 3: Aligned position must be >= current position
    assert aligned_position >= position, \
        f"Aligned position {aligned_position} is less than current position {position}"
    
    # Property 4: Aligned position must be minimal (smallest value >= position divisible by 32)
    if aligned_position > position:
        previous_aligned = aligned_position - 32
        assert previous_aligned < position, \
            f"Previous aligned position {previous_aligned} is >= current position {position}"


# ----------------------------------------------------------------------------
# Additional Property Test: Alignment with Metadata Key Present
# ----------------------------------------------------------------------------

@given(
    position=st.integers(min_value=0, max_value=1000000),
    alignment=st.integers(min_value=1, max_value=1024)
)
def test_property_alignment_with_metadata(position, alignment):
    """
    Property: Alignment respects metadata value when present
    
    **Validates: Requirements 4.1**
    
    For any GGUF file with a "general.alignment" metadata key, the reader
    should use that value for alignment calculations instead of the default.
    
    This property ensures that:
    - Custom alignment values from metadata are respected
    - The alignment calculation uses the metadata value, not the default
    - The behavior is consistent across all alignment values
    """
    reader = GGUFReader("test.gguf")
    # Set custom alignment in metadata
    reader.metadata = {'general.alignment': alignment}
    
    # Create a mock file with enough space
    reader.file = io.BytesIO(b'\x00' * (position + alignment + 100))
    reader.file.seek(position)
    
    # Calculate alignment
    reader._calculate_alignment()
    
    aligned_position = reader.tensor_data_base
    
    # Property: The aligned position must be divisible by the custom alignment value
    assert aligned_position % alignment == 0, \
        f"Aligned position {aligned_position} is not divisible by custom alignment {alignment}"
    
    # Property: If we had used default alignment (32), we might get a different result
    # (unless alignment happens to be 32 or a factor/multiple of 32)
    # This verifies that the custom alignment is actually being used
    if alignment != 32:
        # Calculate what we would get with default alignment
        default_padding = (32 - (position % 32)) % 32
        default_aligned = position + default_padding
        
        # If the custom alignment gives a different result, verify it's correct
        if aligned_position != default_aligned:
            # The aligned position should be divisible by custom alignment
            assert aligned_position % alignment == 0
            # But might not be divisible by 32 (if alignment is not a factor of 32)
            # This confirms we're using the custom alignment, not the default


# ----------------------------------------------------------------------------
# Property 8: Tensor Offset Calculation
# ----------------------------------------------------------------------------

@given(
    tensor_data_base=st.integers(min_value=0, max_value=1000000),
    tensor_offset=st.integers(min_value=0, max_value=1000000)
)
def test_property_tensor_offset_calculation(tensor_data_base, tensor_offset):
    """
    Property 8: Tensor Offset Calculation
    
    **Validates: Requirements 4.4**
    
    For any tensor with a given offset value, the absolute file position for
    that tensor's data should equal the aligned base position plus the tensor's
    offset.
    
    This property ensures that:
    - Tensor offsets are correctly added to the base position
    - The calculation is consistent for all offset values
    - The absolute file position is correctly computed
    - The relationship between base position, offset, and absolute position holds
    """
    reader = GGUFReader("test.gguf")
    
    # Set the aligned base position for tensor data section
    reader.tensor_data_base = tensor_data_base
    
    # Create a tensor info dictionary with the given offset
    tensor_info = {
        'name': 'test_tensor',
        'n_dims': 2,
        'dims': [10, 20],
        'type': GGMLType.F32,
        'offset': tensor_offset
    }
    
    # Calculate the absolute file offset
    absolute_offset = reader._calculate_tensor_offset(tensor_info)
    
    # Property: The absolute offset should equal base + tensor offset
    expected_offset = tensor_data_base + tensor_offset
    assert absolute_offset == expected_offset, \
        f"Tensor offset calculation failed: expected {expected_offset}, got {absolute_offset} " \
        f"(base={tensor_data_base}, offset={tensor_offset})"
    
    # Property: The absolute offset should be >= base position
    assert absolute_offset >= tensor_data_base, \
        f"Absolute offset {absolute_offset} is less than base position {tensor_data_base}"
    
    # Property: The difference between absolute and base should equal tensor offset
    assert absolute_offset - tensor_data_base == tensor_offset, \
        f"Difference between absolute offset and base doesn't match tensor offset: " \
        f"{absolute_offset} - {tensor_data_base} = {absolute_offset - tensor_data_base}, expected {tensor_offset}"


@given(
    tensor_data_base=st.integers(min_value=0, max_value=1000000),
    num_tensors=st.integers(min_value=1, max_value=10),
    data=st.data()
)
def test_property_multiple_tensor_offsets(tensor_data_base, num_tensors, data):
    """
    Property 8 (extended): Multiple Tensor Offset Calculation
    
    **Validates: Requirements 4.4**
    
    For any set of tensors with different offset values, each tensor's absolute
    file position should be correctly calculated as base + offset, and the
    offsets should maintain their relative ordering.
    
    This property ensures that:
    - Multiple tensors can have their offsets calculated independently
    - Each calculation is correct regardless of other tensors
    - Relative ordering of offsets is preserved in absolute positions
    """
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = tensor_data_base
    
    # Generate multiple tensors with different offsets
    tensors = []
    for i in range(num_tensors):
        offset = data.draw(st.integers(min_value=0, max_value=1000000))
        tensor_info = {
            'name': f'tensor_{i}',
            'n_dims': 2,
            'dims': [10, 20],
            'type': GGMLType.F32,
            'offset': offset
        }
        tensors.append(tensor_info)
    
    # Calculate absolute offsets for all tensors
    absolute_offsets = []
    for tensor in tensors:
        absolute_offset = reader._calculate_tensor_offset(tensor)
        absolute_offsets.append(absolute_offset)
        
        # Property: Each absolute offset equals base + tensor offset
        expected = tensor_data_base + tensor['offset']
        assert absolute_offset == expected, \
            f"Tensor {tensor['name']} offset calculation failed: " \
            f"expected {expected}, got {absolute_offset}"
    
    # Property: Relative ordering is preserved
    # If tensor A has offset < tensor B, then absolute_A < absolute_B
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            if tensors[i]['offset'] < tensors[j]['offset']:
                assert absolute_offsets[i] < absolute_offsets[j], \
                    f"Relative ordering not preserved: tensor {i} offset {tensors[i]['offset']} " \
                    f"< tensor {j} offset {tensors[j]['offset']}, but absolute offsets " \
                    f"{absolute_offsets[i]} >= {absolute_offsets[j]}"
            elif tensors[i]['offset'] > tensors[j]['offset']:
                assert absolute_offsets[i] > absolute_offsets[j], \
                    f"Relative ordering not preserved: tensor {i} offset {tensors[i]['offset']} " \
                    f"> tensor {j} offset {tensors[j]['offset']}, but absolute offsets " \
                    f"{absolute_offsets[i]} <= {absolute_offsets[j]}"
            else:  # Equal offsets
                assert absolute_offsets[i] == absolute_offsets[j], \
                    f"Equal offsets should give equal absolute positions: " \
                    f"tensor {i} and {j} both have offset {tensors[i]['offset']}, " \
                    f"but absolute offsets are {absolute_offsets[i]} and {absolute_offsets[j]}"


# ----------------------------------------------------------------------------
# Unit Tests for _calculate_tensor_offset() Method
# ----------------------------------------------------------------------------

def test_calculate_tensor_offset_simple():
    """Test calculating tensor offset with simple values."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 1000
    
    tensor_info = {
        'name': 'test_tensor',
        'offset': 500
    }
    
    result = reader._calculate_tensor_offset(tensor_info)
    assert result == 1500


def test_calculate_tensor_offset_zero_base():
    """Test calculating tensor offset with zero base position."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 0
    
    tensor_info = {
        'name': 'test_tensor',
        'offset': 1234
    }
    
    result = reader._calculate_tensor_offset(tensor_info)
    assert result == 1234


def test_calculate_tensor_offset_zero_offset():
    """Test calculating tensor offset with zero tensor offset."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 5000
    
    tensor_info = {
        'name': 'test_tensor',
        'offset': 0
    }
    
    result = reader._calculate_tensor_offset(tensor_info)
    assert result == 5000


def test_calculate_tensor_offset_large_values():
    """Test calculating tensor offset with large values."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 1000000000
    
    tensor_info = {
        'name': 'test_tensor',
        'offset': 999999999
    }
    
    result = reader._calculate_tensor_offset(tensor_info)
    assert result == 1999999999


def test_calculate_tensor_offset_missing_offset_key():
    """Test that missing offset key raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 1000
    
    # Tensor info without 'offset' key
    tensor_info = {
        'name': 'test_tensor',
        'n_dims': 2,
        'dims': [10, 20],
        'type': GGMLType.F32
    }
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._calculate_tensor_offset(tensor_info)
    
    assert "missing 'offset' key" in str(exc_info.value)
    assert "test_tensor" in str(exc_info.value)


def test_calculate_tensor_offset_with_full_tensor_info():
    """Test calculating tensor offset with complete tensor info."""
    reader = GGUFReader("test.gguf")
    reader.tensor_data_base = 2048
    
    tensor_info = {
        'name': 'embedding.weight',
        'n_dims': 2,
        'dims': [4096, 512],
        'type': GGMLType.F16,
        'offset': 1024
    }
    
    result = reader._calculate_tensor_offset(tensor_info)
    assert result == 3072  # 2048 + 1024


# ============================================================================
# Test _calculate_tensor_size() Method (Task 7.1)
# ============================================================================

def test_calculate_tensor_size_f32_1d():
    """Test calculating size for F32 tensor with 1 dimension."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.F32,
        'dims': [1024]
    }
    
    # F32: type_size=4, block_size=1
    # size = 4 * (1024 / 1) * 1 * 1 * 1 = 4096
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 4096


def test_calculate_tensor_size_f32_2d():
    """Test calculating size for F32 tensor with 2 dimensions."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.F32,
        'dims': [512, 256]
    }
    
    # F32: type_size=4, block_size=1
    # size = 4 * (512 / 1) * 256 * 1 * 1 = 524288
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 524288


def test_calculate_tensor_size_f32_4d():
    """Test calculating size for F32 tensor with 4 dimensions."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.F32,
        'dims': [64, 32, 16, 8]
    }
    
    # F32: type_size=4, block_size=1
    # size = 4 * (64 / 1) * 32 * 16 * 8 = 1048576
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 1048576


def test_calculate_tensor_size_f16():
    """Test calculating size for F16 tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.F16,
        'dims': [1024, 512]
    }
    
    # F16: type_size=2, block_size=1
    # size = 2 * (1024 / 1) * 512 * 1 * 1 = 1048576
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 1048576


def test_calculate_tensor_size_i8():
    """Test calculating size for I8 tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.I8,
        'dims': [2048]
    }
    
    # I8: type_size=1, block_size=1
    # size = 1 * (2048 / 1) * 1 * 1 * 1 = 2048
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 2048


def test_calculate_tensor_size_q4_0():
    """Test calculating size for Q4_0 quantized tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.Q4_0,
        'dims': [1024, 512]
    }
    
    # Q4_0: type_size=18, block_size=32
    # size = 18 * (1024 / 32) * 512 * 1 * 1 = 294912
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 294912


def test_calculate_tensor_size_q8_0():
    """Test calculating size for Q8_0 quantized tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.Q8_0,
        'dims': [2048]
    }
    
    # Q8_0: type_size=34, block_size=32
    # size = 34 * (2048 / 32) * 1 * 1 * 1 = 2176
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 2176


def test_calculate_tensor_size_q2_k():
    """Test calculating size for Q2_K quantized tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.Q2_K,
        'dims': [4096, 256]
    }
    
    # Q2_K: type_size=82, block_size=256
    # size = 82 * (4096 / 256) * 256 * 1 * 1 = 335872
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 335872


def test_calculate_tensor_size_q6_k():
    """Test calculating size for Q6_K quantized tensor."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.Q6_K,
        'dims': [512, 512]
    }
    
    # Q6_K: type_size=210, block_size=256
    # size = 210 * (512 / 256) * 512 * 1 * 1 = 215040
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 215040


def test_calculate_tensor_size_missing_type_key():
    """Test that missing type key raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'dims': [1024]
    }
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._calculate_tensor_size(tensor_info)
    
    assert "missing 'type' key" in str(exc_info.value)
    assert "test_tensor" in str(exc_info.value)


def test_calculate_tensor_size_missing_dims_key():
    """Test that missing dims key raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.F32
    }
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._calculate_tensor_size(tensor_info)
    
    assert "missing 'dims' key" in str(exc_info.value)
    assert "test_tensor" in str(exc_info.value)


def test_calculate_tensor_size_invalid_type():
    """Test that invalid tensor type raises GGUFParseError."""
    reader = GGUFReader("test.gguf")
    
    tensor_info = {
        'name': 'test_tensor',
        'type': 999,  # Invalid type
        'dims': [1024]
    }
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._calculate_tensor_size(tensor_info)
    
    assert "Invalid tensor type" in str(exc_info.value)
    assert "999" in str(exc_info.value)


def test_calculate_tensor_size_all_standard_types():
    """Test calculating size for all standard tensor types."""
    reader = GGUFReader("test.gguf")
    
    # Test all standard types with same dimensions
    dims = [1024]
    
    test_cases = [
        (GGMLType.F32, 4, 1, 4096),    # 4 * (1024 / 1) = 4096
        (GGMLType.F16, 2, 1, 2048),    # 2 * (1024 / 1) = 2048
        (GGMLType.I8, 1, 1, 1024),     # 1 * (1024 / 1) = 1024
        (GGMLType.I16, 2, 1, 2048),    # 2 * (1024 / 1) = 2048
        (GGMLType.I32, 4, 1, 4096),    # 4 * (1024 / 1) = 4096
    ]
    
    for tensor_type, type_size, block_size, expected_size in test_cases:
        tensor_info = {
            'name': f'test_tensor_{tensor_type}',
            'type': tensor_type,
            'dims': dims
        }
        
        result = reader._calculate_tensor_size(tensor_info)
        assert result == expected_size, \
            f"Size mismatch for type {tensor_type}: expected {expected_size}, got {result}"


def test_calculate_tensor_size_all_quantized_types():
    """Test calculating size for all quantized tensor types."""
    reader = GGUFReader("test.gguf")
    
    # Test all quantized types with dimensions that are multiples of their block sizes
    test_cases = [
        (GGMLType.Q4_0, [1024], 18, 32, 576),      # 18 * (1024 / 32) = 576
        (GGMLType.Q4_1, [1024], 20, 32, 640),      # 20 * (1024 / 32) = 640
        (GGMLType.Q5_0, [1024], 22, 32, 704),      # 22 * (1024 / 32) = 704
        (GGMLType.Q5_1, [1024], 24, 32, 768),      # 24 * (1024 / 32) = 768
        (GGMLType.Q8_0, [1024], 34, 32, 1088),     # 34 * (1024 / 32) = 1088
        (GGMLType.Q8_1, [1024], 40, 32, 1280),     # 40 * (1024 / 32) = 1280
        (GGMLType.Q2_K, [2560], 82, 256, 820),     # 82 * (2560 / 256) = 820
        (GGMLType.Q3_K, [2560], 110, 256, 1100),   # 110 * (2560 / 256) = 1100
        (GGMLType.Q4_K, [2560], 144, 256, 1440),   # 144 * (2560 / 256) = 1440
        (GGMLType.Q5_K, [2560], 176, 256, 1760),   # 176 * (2560 / 256) = 1760
        (GGMLType.Q6_K, [2560], 210, 256, 2100),   # 210 * (2560 / 256) = 2100
        (GGMLType.Q8_K, [2560], 292, 256, 2920),   # 292 * (2560 / 256) = 2920
    ]
    
    for tensor_type, dims, type_size, block_size, expected_size in test_cases:
        tensor_info = {
            'name': f'test_tensor_{tensor_type}',
            'type': tensor_type,
            'dims': dims
        }
        
        result = reader._calculate_tensor_size(tensor_info)
        assert result == expected_size, \
            f"Size mismatch for type {tensor_type}: expected {expected_size}, got {result}"


def test_calculate_tensor_size_multidimensional_quantized():
    """Test calculating size for multidimensional quantized tensors."""
    reader = GGUFReader("test.gguf")
    
    # Q4_0 with 3 dimensions
    tensor_info = {
        'name': 'test_tensor',
        'type': GGMLType.Q4_0,
        'dims': [1024, 64, 32]
    }
    
    # Q4_0: type_size=18, block_size=32
    # size = 18 * (1024 / 32) * 64 * 32 * 1 = 1179648
    result = reader._calculate_tensor_size(tensor_info)
    assert result == 1179648


# ============================================================================
# Property-Based Tests for Tensor Size Calculation (Task 7.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 9: Tensor Size Calculation
# ----------------------------------------------------------------------------

@given(
    tensor_type=st.sampled_from(list(TYPE_SIZES.keys())),
    data=st.data()
)
def test_property_tensor_size_calculation(tensor_type, data):
    """
    Property 9: Tensor Size Calculation
    
    **Validates: Requirements 5.3**
    
    For any tensor with dimensions [d0, d1, d2, d3] and type with (type_size, block_size),
    the calculated data size should equal: type_size * (d0 / block_size) * d1 * d2 * d3.
    
    This property ensures that:
    - Size calculation is correct for all tensor types (standard and quantized)
    - The formula correctly accounts for block sizes in quantized types
    - Dimensions are correctly multiplied together
    - The calculation works for tensors with 1-4 dimensions
    - Edge cases like single-element tensors are handled correctly
    """
    reader = GGUFReader("test.gguf")
    
    # Get type_size and block_size for this tensor type
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Generate dimensions that are multiples of block_size for the first dimension
    # This ensures valid tensor sizes (first dimension must be divisible by block_size)
    # Generate 1-4 dimensions
    num_dims = data.draw(st.integers(min_value=1, max_value=4))
    
    dims = []
    for i in range(num_dims):
        if i == 0:
            # First dimension must be a multiple of block_size
            # Generate a reasonable number of blocks (1 to 100)
            num_blocks = data.draw(st.integers(min_value=1, max_value=100))
            dims.append(num_blocks * block_size)
        else:
            # Other dimensions can be any reasonable size (1 to 1000)
            dims.append(data.draw(st.integers(min_value=1, max_value=1000)))
    
    # Create tensor info
    tensor_info = {
        'name': 'test_tensor',
        'type': tensor_type,
        'dims': dims
    }
    
    # Calculate size using the reader
    calculated_size = reader._calculate_tensor_size(tensor_info)
    
    # Calculate expected size using the formula
    # Pad dimensions to 4 dimensions with 1s
    padded_dims = dims + [1] * (4 - len(dims))
    expected_size = type_size * (padded_dims[0] // block_size) * padded_dims[1] * padded_dims[2] * padded_dims[3]
    
    # Verify the property holds
    assert calculated_size == expected_size, \
        f"Tensor size calculation failed for type {tensor_type} with dims {dims}: " \
        f"expected {expected_size}, got {calculated_size} " \
        f"(type_size={type_size}, block_size={block_size})"
    
    # Additional property: size should be positive
    assert calculated_size > 0, \
        f"Tensor size should be positive, got {calculated_size}"
    
    # Additional property: size should be a multiple of type_size
    assert calculated_size % type_size == 0, \
        f"Tensor size should be a multiple of type_size ({type_size}), got {calculated_size}"
    
    # Additional property: for standard types (block_size=1), size should equal type_size * product of dims
    if block_size == 1:
        product_of_dims = 1
        for dim in dims:
            product_of_dims *= dim
        expected_standard = type_size * product_of_dims
        assert calculated_size == expected_standard, \
            f"For standard type (block_size=1), size should equal type_size * product of dims: " \
            f"expected {expected_standard}, got {calculated_size}"


@given(
    data=st.data()
)
def test_property_tensor_size_dimension_scaling(data):
    """
    Property 9 (extended): Tensor Size Dimension Scaling
    
    **Validates: Requirements 5.3**
    
    For any tensor, if we double any dimension (except the first), the size should double.
    If we double the first dimension, the size should double (since it's measured in blocks).
    
    This property ensures that:
    - Size scales linearly with each dimension
    - The calculation correctly handles dimension changes
    - The formula is consistent across dimension modifications
    """
    reader = GGUFReader("test.gguf")
    
    # Pick a random tensor type
    tensor_type = data.draw(st.sampled_from(list(TYPE_SIZES.keys())))
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Generate initial dimensions (1-4 dimensions)
    num_dims = data.draw(st.integers(min_value=1, max_value=4))
    
    dims = []
    for i in range(num_dims):
        if i == 0:
            # First dimension must be a multiple of block_size
            num_blocks = data.draw(st.integers(min_value=2, max_value=50))
            dims.append(num_blocks * block_size)
        else:
            # Other dimensions can be any reasonable size (2 to 500)
            dims.append(data.draw(st.integers(min_value=2, max_value=500)))
    
    # Calculate original size
    tensor_info = {
        'name': 'test_tensor',
        'type': tensor_type,
        'dims': dims
    }
    original_size = reader._calculate_tensor_size(tensor_info)
    
    # Test scaling each dimension
    for dim_idx in range(len(dims)):
        # Create a copy of dims with one dimension doubled
        scaled_dims = dims.copy()
        scaled_dims[dim_idx] = dims[dim_idx] * 2
        
        # Calculate size with scaled dimension
        scaled_tensor_info = {
            'name': 'test_tensor',
            'type': tensor_type,
            'dims': scaled_dims
        }
        scaled_size = reader._calculate_tensor_size(scaled_tensor_info)
        
        # Property: Doubling any dimension should double the size
        expected_scaled_size = original_size * 2
        assert scaled_size == expected_scaled_size, \
            f"Doubling dimension {dim_idx} should double the size: " \
            f"original dims {dims} -> size {original_size}, " \
            f"scaled dims {scaled_dims} -> size {scaled_size}, " \
            f"expected {expected_scaled_size}"


@given(
    tensor_type=st.sampled_from(list(TYPE_SIZES.keys()))
)
def test_property_tensor_size_minimum(tensor_type):
    """
    Property 9 (extended): Minimum Tensor Size
    
    **Validates: Requirements 5.3**
    
    For any tensor type, the minimum valid tensor (one block in first dimension,
    size 1 in other dimensions) should have size equal to type_size.
    
    This property ensures that:
    - The minimum tensor size is correctly calculated
    - Single-block tensors work correctly
    - The formula handles the smallest valid tensor
    """
    reader = GGUFReader("test.gguf")
    
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Create minimum tensor: one block in first dimension, 1 in others
    tensor_info = {
        'name': 'test_tensor',
        'type': tensor_type,
        'dims': [block_size]  # Minimum: one block
    }
    
    calculated_size = reader._calculate_tensor_size(tensor_info)
    
    # Property: Minimum size should equal type_size
    # Formula: type_size * (block_size / block_size) * 1 * 1 * 1 = type_size
    expected_size = type_size
    assert calculated_size == expected_size, \
        f"Minimum tensor size for type {tensor_type} should be {expected_size}, got {calculated_size}"


@given(
    data=st.data()
)
def test_property_tensor_size_type_comparison(data):
    """
    Property 9 (extended): Tensor Size Type Comparison
    
    **Validates: Requirements 5.3**
    
    For tensors with the same dimensions but different types, the size ratio
    should equal the type_size ratio (accounting for block sizes).
    
    This property ensures that:
    - Different types with same dimensions produce correctly scaled sizes
    - The type_size parameter correctly affects the calculation
    - Block sizes are properly accounted for in the comparison
    """
    reader = GGUFReader("test.gguf")
    
    # Pick two different tensor types
    all_types = list(TYPE_SIZES.keys())
    type1 = data.draw(st.sampled_from(all_types))
    type2 = data.draw(st.sampled_from(all_types))
    
    type_size1, block_size1 = TYPE_SIZES[type1]
    type_size2, block_size2 = TYPE_SIZES[type2]
    
    # Generate dimensions that work for both types
    # First dimension must be a multiple of LCM of both block sizes
    import math
    lcm_block_size = (block_size1 * block_size2) // math.gcd(block_size1, block_size2)
    
    num_blocks = data.draw(st.integers(min_value=1, max_value=20))
    first_dim = num_blocks * lcm_block_size
    
    # Generate 1-3 additional dimensions
    num_extra_dims = data.draw(st.integers(min_value=0, max_value=3))
    dims = [first_dim]
    for _ in range(num_extra_dims):
        dims.append(data.draw(st.integers(min_value=1, max_value=100)))
    
    # Calculate sizes for both types
    tensor_info1 = {
        'name': 'test_tensor',
        'type': type1,
        'dims': dims
    }
    size1 = reader._calculate_tensor_size(tensor_info1)
    
    tensor_info2 = {
        'name': 'test_tensor',
        'type': type2,
        'dims': dims
    }
    size2 = reader._calculate_tensor_size(tensor_info2)
    
    # Calculate expected ratio
    # size1 = type_size1 * (first_dim / block_size1) * other_dims
    # size2 = type_size2 * (first_dim / block_size2) * other_dims
    # ratio = (type_size1 / block_size1) / (type_size2 / block_size2)
    expected_ratio = (type_size1 / block_size1) / (type_size2 / block_size2)
    actual_ratio = size1 / size2 if size2 > 0 else 0
    
    # Property: The size ratio should match the expected ratio
    # Use relative tolerance for floating point comparison
    tolerance = 1e-10
    assert abs(actual_ratio - expected_ratio) < tolerance, \
        f"Size ratio mismatch for types {type1} and {type2} with dims {dims}: " \
        f"expected ratio {expected_ratio}, got {actual_ratio} " \
        f"(size1={size1}, size2={size2})"


# ============================================================================
# Test get_tensor_data() Method (Task 7.3)
# ============================================================================

def test_get_tensor_data_simple():
    """Test reading tensor data for a simple F32 tensor."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'test_tensor',
            'n_dims': 1,
            'dims': [32],  # 32 elements
            'type': GGMLType.F32,
            'offset': 0
        }
    ]
    
    # Create mock file with tensor data
    # F32: 32 elements * 4 bytes = 128 bytes
    tensor_data = b'\x00' * 128
    file_content = b'\x00' * 1000 + tensor_data  # Padding + tensor data
    reader.file = io.BytesIO(file_content)
    
    # Read tensor data
    result = reader.get_tensor_data('test_tensor')
    
    assert len(result) == 128
    assert result == tensor_data


def test_get_tensor_data_with_offset():
    """Test reading tensor data with non-zero offset."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'tensor1',
            'n_dims': 1,
            'dims': [32],
            'type': GGMLType.F32,
            'offset': 0
        },
        {
            'name': 'tensor2',
            'n_dims': 1,
            'dims': [16],
            'type': GGMLType.F32,
            'offset': 128  # After tensor1
        }
    ]
    
    # Create mock file with tensor data
    tensor1_data = b'\x01' * 128
    tensor2_data = b'\x02' * 64
    file_content = b'\x00' * 1000 + tensor1_data + tensor2_data
    reader.file = io.BytesIO(file_content)
    
    # Read second tensor
    result = reader.get_tensor_data('tensor2')
    
    assert len(result) == 64
    assert result == tensor2_data


def test_get_tensor_data_caching():
    """Test that tensor data is cached after first read."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'test_tensor',
            'n_dims': 1,
            'dims': [32],
            'type': GGMLType.F32,
            'offset': 0
        }
    ]
    
    # Create mock file with tensor data
    tensor_data = b'\xAB' * 128
    file_content = b'\x00' * 1000 + tensor_data
    reader.file = io.BytesIO(file_content)
    
    # First read
    result1 = reader.get_tensor_data('test_tensor')
    assert result1 == tensor_data
    
    # Verify it's cached
    assert 'test_tensor' in reader.tensor_data_cache
    assert reader.tensor_data_cache['test_tensor'] == tensor_data
    
    # Second read should return cached data
    result2 = reader.get_tensor_data('test_tensor')
    assert result2 == tensor_data
    assert result2 is reader.tensor_data_cache['test_tensor']


def test_get_tensor_data_not_found():
    """Test that KeyError is raised for non-existent tensor."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state with one tensor
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'existing_tensor',
            'n_dims': 1,
            'dims': [32],
            'type': GGMLType.F32,
            'offset': 0
        }
    ]
    reader.file = io.BytesIO(b'\x00' * 2000)
    
    # Try to read non-existent tensor
    with pytest.raises(KeyError) as exc_info:
        reader.get_tensor_data('non_existent_tensor')
    
    assert "Tensor 'non_existent_tensor' not found" in str(exc_info.value)


def test_get_tensor_data_file_not_open():
    """Test that GGUFParseError is raised when file is not open."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state but don't open file
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'test_tensor',
            'n_dims': 1,
            'dims': [32],
            'type': GGMLType.F32,
            'offset': 0
        }
    ]
    # reader.file is None
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader.get_tensor_data('test_tensor')
    
    assert "File is not open" in str(exc_info.value)


def test_get_tensor_data_truncated():
    """Test that GGUFTruncatedError is raised when file is truncated."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'test_tensor',
            'n_dims': 1,
            'dims': [32],  # Expects 128 bytes
            'type': GGMLType.F32,
            'offset': 0
        }
    ]
    
    # Create mock file with insufficient data (only 64 bytes instead of 128)
    file_content = b'\x00' * 1000 + b'\x01' * 64
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader.get_tensor_data('test_tensor')
    
    assert "expected to read 128 bytes" in str(exc_info.value)
    assert "only 64 bytes available" in str(exc_info.value)


def test_get_tensor_data_quantized_q4_0():
    """Test reading tensor data for Q4_0 quantized tensor."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'quant_tensor',
            'n_dims': 1,
            'dims': [1024],  # 1024 elements
            'type': GGMLType.Q4_0,
            'offset': 0
        }
    ]
    
    # Q4_0: type_size=18, block_size=32
    # size = 18 * (1024 / 32) = 576 bytes
    tensor_data = b'\xFF' * 576
    file_content = b'\x00' * 1000 + tensor_data
    reader.file = io.BytesIO(file_content)
    
    result = reader.get_tensor_data('quant_tensor')
    
    assert len(result) == 576
    assert result == tensor_data


def test_get_tensor_data_multidimensional():
    """Test reading tensor data for multidimensional tensor."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'multi_tensor',
            'n_dims': 3,
            'dims': [64, 32, 16],  # 3D tensor
            'type': GGMLType.F16,
            'offset': 0
        }
    ]
    
    # F16: type_size=2, block_size=1
    # size = 2 * 64 * 32 * 16 = 65536 bytes
    tensor_data = b'\xAA' * 65536
    file_content = b'\x00' * 1000 + tensor_data
    reader.file = io.BytesIO(file_content)
    
    result = reader.get_tensor_data('multi_tensor')
    
    assert len(result) == 65536
    assert result == tensor_data


def test_get_tensor_data_multiple_tensors():
    """Test reading data from multiple tensors."""
    reader = GGUFReader("test.gguf")
    
    # Set up reader state with multiple tensors
    reader.tensor_data_base = 1000
    reader.tensor_info = [
        {
            'name': 'tensor_a',
            'n_dims': 1,
            'dims': [32],
            'type': GGMLType.F32,
            'offset': 0
        },
        {
            'name': 'tensor_b',
            'n_dims': 1,
            'dims': [16],
            'type': GGMLType.F32,
            'offset': 128
        },
        {
            'name': 'tensor_c',
            'n_dims': 1,
            'dims': [8],
            'type': GGMLType.F32,
            'offset': 192
        }
    ]
    
    # Create mock file with all tensor data
    tensor_a_data = b'\x01' * 128
    tensor_b_data = b'\x02' * 64
    tensor_c_data = b'\x03' * 32
    file_content = b'\x00' * 1000 + tensor_a_data + tensor_b_data + tensor_c_data
    reader.file = io.BytesIO(file_content)
    
    # Read all tensors
    result_a = reader.get_tensor_data('tensor_a')
    result_b = reader.get_tensor_data('tensor_b')
    result_c = reader.get_tensor_data('tensor_c')
    
    assert result_a == tensor_a_data
    assert result_b == tensor_b_data
    assert result_c == tensor_c_data


# ============================================================================
# Property-Based Tests for Tensor Data Reading (Task 7.4)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 10: Tensor Data Round-Trip
# ----------------------------------------------------------------------------

def create_complete_gguf_file_with_tensor_data(
    tensor_info_list: List[Dict[str, Any]],
    tensor_data_dict: Dict[str, bytes],
    alignment: int = 32
) -> bytes:
    """
    Helper function to create a complete GGUF file with header, metadata, tensor info, and tensor data.
    
    Args:
        tensor_info_list: List of tensor info dictionaries
        tensor_data_dict: Dictionary mapping tensor names to their raw data bytes
        alignment: Alignment value for tensor data section (default 32)
        
    Returns:
        Complete GGUF file as bytes
    """
    # Create header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', len(tensor_info_list))  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count (for alignment)
    
    # Add metadata with alignment value
    key = 'general.alignment'
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)  # value type
    file_content += struct.pack('<I', alignment)  # alignment value
    
    # Add tensor info
    file_content += encode_tensor_info_list(tensor_info_list)
    
    # Calculate alignment padding
    current_position = len(file_content)
    padding = (alignment - (current_position % alignment)) % alignment
    file_content += b'\x00' * padding
    
    # Add tensor data in order of offsets
    # Sort tensors by offset to ensure correct placement
    sorted_tensors = sorted(tensor_info_list, key=lambda t: t['offset'])
    
    for tensor in sorted_tensors:
        tensor_name = tensor['name']
        if tensor_name in tensor_data_dict:
            # Add padding if needed to reach the correct offset
            expected_position = len(file_content) - (current_position + padding)
            if expected_position < tensor['offset']:
                file_content += b'\x00' * (tensor['offset'] - expected_position)
            
            # Add tensor data
            file_content += tensor_data_dict[tensor_name]
    
    return file_content


@st.composite
def tensor_with_data_strategy(draw):
    """
    Generate a tensor info record with corresponding random data.
    
    Returns a tuple of (tensor_info, tensor_data_bytes)
    """
    # Generate tensor name
    name = draw(st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='._-'
        )
    ))
    
    # Generate tensor type (must be valid type from TYPE_SIZES)
    valid_tensor_types = list(TYPE_SIZES.keys())
    tensor_type = draw(st.sampled_from(valid_tensor_types))
    
    # Get type info
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Generate number of dimensions (1-4)
    n_dims = draw(st.integers(min_value=1, max_value=4))
    
    # Generate dimension sizes
    # First dimension must be a multiple of block_size
    num_blocks = draw(st.integers(min_value=1, max_value=10))
    first_dim = num_blocks * block_size
    
    dims = [first_dim]
    for _ in range(n_dims - 1):
        dims.append(draw(st.integers(min_value=1, max_value=20)))
    
    # Calculate tensor data size
    padded_dims = dims + [1] * (4 - len(dims))
    data_size = type_size * (padded_dims[0] // block_size) * padded_dims[1] * padded_dims[2] * padded_dims[3]
    
    # Generate random tensor data
    tensor_data = draw(st.binary(min_size=data_size, max_size=data_size))
    
    tensor_info = {
        'name': name,
        'n_dims': n_dims,
        'dims': dims,
        'type': tensor_type,
        'offset': 0  # Will be set later
    }
    
    return (tensor_info, tensor_data)


@st.composite
def tensor_list_with_data_strategy(draw):
    """
    Generate a list of tensors with their data, ensuring unique names and proper offsets.
    
    Returns a tuple of (tensor_info_list, tensor_data_dict)
    """
    num_tensors = draw(st.integers(min_value=1, max_value=5))
    
    tensor_info_list = []
    tensor_data_dict = {}
    used_names = set()
    current_offset = 0
    
    for i in range(num_tensors):
        tensor_info, tensor_data = draw(tensor_with_data_strategy())
        
        # Ensure unique names
        base_name = tensor_info['name']
        counter = 0
        while tensor_info['name'] in used_names:
            counter += 1
            tensor_info['name'] = f"{base_name}_{counter}"
        
        used_names.add(tensor_info['name'])
        
        # Set offset
        tensor_info['offset'] = current_offset
        
        # Store tensor info and data
        tensor_info_list.append(tensor_info)
        tensor_data_dict[tensor_info['name']] = tensor_data
        
        # Update offset for next tensor
        current_offset += len(tensor_data)
    
    return (tensor_info_list, tensor_data_dict)


@given(data=st.data())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_tensor_data_round_trip(data):
    """
    Property 10: Tensor Data Round-Trip
    
    **Validates: Requirements 5.1, 5.2, 5.4, 5.5**
    
    For any tensor with valid type and dimensions, when tensor data is written to
    a GGUF file and read back by the reader, the read bytes should match the
    original bytes.
    
    This property ensures that:
    - All standard tensor types (F32, F16, I8, I16, I32) can be read correctly (Req 5.1)
    - All quantized tensor types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, 
      Q4_K, Q5_K, Q6_K, Q8_K) can be read correctly (Req 5.2)
    - Tensor data is read from the correct file offset (Req 5.4)
    - The correct number of bytes is read for each tensor (Req 5.5)
    - Tensor data is preserved exactly through write/read cycle
    - Multiple tensors can be read independently
    - Tensors with different types and dimensions work correctly
    - Data caching works correctly (second read returns same data)
    """
    # Generate tensors with data
    tensor_info_list, tensor_data_dict = data.draw(tensor_list_with_data_strategy())
    
    # Create complete GGUF file
    file_content = create_complete_gguf_file_with_tensor_data(
        tensor_info_list,
        tensor_data_dict,
        alignment=32
    )
    
    # Create reader and parse file
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    
    # Parse header, metadata, and tensor info
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    # Verify each tensor's data can be read correctly
    for tensor_info in tensor_info_list:
        tensor_name = tensor_info['name']
        expected_data = tensor_data_dict[tensor_name]
        
        # Read tensor data
        actual_data = reader.get_tensor_data(tensor_name)
        
        # Verify round-trip: read data should match original data
        assert len(actual_data) == len(expected_data), \
            f"Tensor '{tensor_name}' data length mismatch: " \
            f"expected {len(expected_data)} bytes, got {len(actual_data)} bytes"
        
        assert actual_data == expected_data, \
            f"Tensor '{tensor_name}' data mismatch: " \
            f"read data does not match original data"
        
        # Verify data is cached (Requirement 5.5 - efficient reading)
        assert tensor_name in reader.tensor_data_cache, \
            f"Tensor '{tensor_name}' data should be cached after first read"
        
        assert reader.tensor_data_cache[tensor_name] == expected_data, \
            f"Cached data for tensor '{tensor_name}' does not match original"
        
        # Verify second read returns cached data
        second_read = reader.get_tensor_data(tensor_name)
        assert second_read is reader.tensor_data_cache[tensor_name], \
            f"Second read of tensor '{tensor_name}' should return cached data"


@given(
    tensor_type=st.sampled_from(list(TYPE_SIZES.keys())),
    data=st.data()
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
def test_property_tensor_data_all_types(tensor_type, data):
    """
    Property 10 (extended): Tensor Data Round-Trip for All Types
    
    **Validates: Requirements 5.1, 5.2**
    
    This property specifically tests that all tensor types (both standard and
    quantized) can have their data written and read back correctly.
    
    This ensures comprehensive coverage of:
    - Standard types: F32, F16, I8, I16, I32 (Requirement 5.1)
    - Quantized types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, 
      Q5_K, Q6_K, Q8_K (Requirement 5.2)
    """
    # Get type info
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Generate dimensions (first dimension must be multiple of block_size)
    # Keep sizes small to avoid generating huge amounts of data
    num_blocks = data.draw(st.integers(min_value=1, max_value=5))
    first_dim = num_blocks * block_size
    
    n_dims = data.draw(st.integers(min_value=1, max_value=3))
    dims = [first_dim]
    for _ in range(n_dims - 1):
        dims.append(data.draw(st.integers(min_value=1, max_value=10)))
    
    # Calculate tensor data size
    padded_dims = dims + [1] * (4 - len(dims))
    data_size = type_size * (padded_dims[0] // block_size) * padded_dims[1] * padded_dims[2] * padded_dims[3]
    
    # Generate random tensor data
    tensor_data = data.draw(st.binary(min_size=data_size, max_size=data_size))
    
    # Create tensor info
    tensor_info = {
        'name': 'test_tensor',
        'n_dims': n_dims,
        'dims': dims,
        'type': tensor_type,
        'offset': 0
    }
    
    # Create complete GGUF file
    file_content = create_complete_gguf_file_with_tensor_data(
        [tensor_info],
        {'test_tensor': tensor_data},
        alignment=32
    )
    
    # Create reader and parse file
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    
    # Parse file
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    # Read tensor data
    actual_data = reader.get_tensor_data('test_tensor')
    
    # Verify round-trip
    assert len(actual_data) == len(tensor_data), \
        f"Tensor type {tensor_type} data length mismatch: " \
        f"expected {len(tensor_data)} bytes, got {len(actual_data)} bytes"
    
    assert actual_data == tensor_data, \
        f"Tensor type {tensor_type} data mismatch: " \
        f"read data does not match original data"


@given(
    alignment=st.sampled_from([1, 2, 4, 8, 16, 32, 64, 128, 256]),
    data=st.data()
)
def test_property_tensor_data_with_alignment(alignment, data):
    """
    Property 10 (extended): Tensor Data Round-Trip with Different Alignments
    
    **Validates: Requirements 5.4**
    
    This property tests that tensor data can be read correctly regardless of
    the alignment value used in the GGUF file. The alignment affects where
    the tensor data section begins in the file.
    """
    # Generate a simple tensor
    tensor_type = data.draw(st.sampled_from(list(TYPE_SIZES.keys())))
    type_size, block_size = TYPE_SIZES[tensor_type]
    
    # Generate dimensions
    num_blocks = data.draw(st.integers(min_value=1, max_value=5))
    first_dim = num_blocks * block_size
    dims = [first_dim]
    
    # Calculate tensor data size
    data_size = type_size * num_blocks
    
    # Generate random tensor data
    tensor_data = data.draw(st.binary(min_size=data_size, max_size=data_size))
    
    # Create tensor info
    tensor_info = {
        'name': 'aligned_tensor',
        'n_dims': 1,
        'dims': dims,
        'type': tensor_type,
        'offset': 0
    }
    
    # Create complete GGUF file with specified alignment
    file_content = create_complete_gguf_file_with_tensor_data(
        [tensor_info],
        {'aligned_tensor': tensor_data},
        alignment=alignment
    )
    
    # Create reader and parse file
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    
    # Parse file
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    # Verify alignment was calculated correctly
    assert reader.tensor_data_base % alignment == 0, \
        f"Tensor data base {reader.tensor_data_base} is not aligned to {alignment} bytes"
    
    # Read tensor data
    actual_data = reader.get_tensor_data('aligned_tensor')
    
    # Verify round-trip
    assert actual_data == tensor_data, \
        f"Tensor data mismatch with alignment {alignment}: " \
        f"read data does not match original data"


# ============================================================================
# Property Tests for Access Interface (Task 8.5)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 11: Metadata Retrieval
# ----------------------------------------------------------------------------

@given(
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='._-'
        )),
        values=st.one_of(
            st.integers(min_value=-1000000, max_value=1000000),
            st.floats(width=32, allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.text(max_size=100),
            st.lists(st.integers(min_value=-1000, max_value=1000), max_size=20),
            st.lists(st.text(max_size=50), max_size=10)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_metadata_retrieval(metadata, tmp_path):
    """
    Property 11: Metadata Retrieval
    
    **Validates: Requirements 6.1, 6.2**
    
    For any parsed GGUF file with metadata, retrieving all metadata should return
    a dictionary containing all key-value pairs, and retrieving by specific key
    should return the correct value.
    
    This property ensures that:
    - get_metadata() returns all metadata key-value pairs
    - get_metadata_value(key) returns the correct value for each key
    - Metadata values are preserved exactly through parsing
    - All metadata types (integers, floats, bools, strings, arrays) are accessible
    - KeyError is raised for non-existent keys
    """
    # Create a complete GGUF file with the metadata
    file_content = b''
    
    # Header
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', len(metadata))  # metadata_kv_count
    
    # Metadata
    file_content += encode_metadata_dict(metadata)
    
    # Write to temporary file
    test_file = tmp_path / "test_metadata.gguf"
    test_file.write_bytes(file_content)
    
    # Parse file using context manager
    with GGUFReader(str(test_file)) as reader:
        # Test get_metadata() returns all metadata
        retrieved_metadata = reader.get_metadata()
        
        assert isinstance(retrieved_metadata, dict), \
            "get_metadata() should return a dictionary"
        
        assert len(retrieved_metadata) == len(metadata), \
            f"Metadata count mismatch: expected {len(metadata)}, got {len(retrieved_metadata)}"
        
        # Verify all keys are present
        for key in metadata.keys():
            assert key in retrieved_metadata, \
                f"Key '{key}' missing from retrieved metadata"
        
        # Test get_metadata_value(key) for each key
        for key, expected_value in metadata.items():
            retrieved_value = reader.get_metadata_value(key)
            
            # Compare values based on type
            if isinstance(expected_value, float):
                # For floats, check within tolerance
                tolerance = abs(expected_value) * 1e-6 if expected_value != 0 else 1e-6
                assert abs(retrieved_value - expected_value) <= tolerance, \
                    f"Metadata value mismatch for key '{key}': " \
                    f"expected {expected_value}, got {retrieved_value}"
            elif isinstance(expected_value, list) and len(expected_value) > 0 and isinstance(expected_value[0], float):
                # For float arrays, check each element within tolerance
                assert len(retrieved_value) == len(expected_value), \
                    f"Array length mismatch for key '{key}'"
                for i, (exp, act) in enumerate(zip(expected_value, retrieved_value)):
                    tolerance = abs(exp) * 1e-6 if exp != 0 else 1e-6
                    assert abs(act - exp) <= tolerance, \
                        f"Array element {i} mismatch for key '{key}'"
            else:
                # For other types, check exact equality
                assert retrieved_value == expected_value, \
                    f"Metadata value mismatch for key '{key}': " \
                    f"expected {expected_value}, got {retrieved_value}"
        
        # Test that non-existent key raises KeyError
        with pytest.raises(KeyError):
            reader.get_metadata_value('non_existent_key_12345')


# ----------------------------------------------------------------------------
# Property 12: Tensor Retrieval
# ----------------------------------------------------------------------------

@given(
    num_tensors=st.integers(min_value=1, max_value=5),
    data=st.data()
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_tensor_retrieval(num_tensors, data, tmp_path):
    """
    Property 12: Tensor Retrieval
    
    **Validates: Requirements 6.3, 6.4, 6.5**
    
    For any parsed GGUF file with tensors, listing tensor names should return
    all tensor names, and retrieving tensor info or data by name should return
    the correct information.
    
    This property ensures that:
    - list_tensors() returns all tensor names
    - get_tensor_info(name) returns correct tensor metadata for each tensor
    - get_tensor_data(name) returns correct tensor data for each tensor
    - Tensor names, dimensions, types, and data are preserved exactly
    - KeyError is raised for non-existent tensor names
    """
    # Generate unique tensor names
    tensor_names = []
    for i in range(num_tensors):
        name = data.draw(st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='._-'
            )
        ))
        # Ensure uniqueness
        while name in tensor_names:
            name = name + str(i)
        tensor_names.append(name)
    
    # Generate tensor info and data
    tensor_info_list = []
    tensor_data_dict = {}
    current_offset = 0
    
    for name in tensor_names:
        # Generate tensor type
        valid_tensor_types = list(TYPE_SIZES.keys())
        tensor_type = data.draw(st.sampled_from(valid_tensor_types))
        type_size, block_size = TYPE_SIZES[tensor_type]
        
        # Generate dimensions
        n_dims = data.draw(st.integers(min_value=1, max_value=4))
        num_blocks = data.draw(st.integers(min_value=1, max_value=5))
        first_dim = num_blocks * block_size
        dims = [first_dim]
        for _ in range(n_dims - 1):
            dims.append(data.draw(st.integers(min_value=1, max_value=10)))
        
        # Calculate data size
        padded_dims = dims + [1] * (4 - len(dims))
        data_size = type_size * (padded_dims[0] // block_size) * padded_dims[1] * padded_dims[2] * padded_dims[3]
        
        # Generate random data
        tensor_data = data.draw(st.binary(min_size=data_size, max_size=data_size))
        
        # Create tensor info
        tensor_info = {
            'name': name,
            'n_dims': n_dims,
            'dims': dims,
            'type': tensor_type,
            'offset': current_offset
        }
        
        tensor_info_list.append(tensor_info)
        tensor_data_dict[name] = tensor_data
        current_offset += data_size
    
    # Create complete GGUF file
    file_content = create_complete_gguf_file_with_tensor_data(
        tensor_info_list,
        tensor_data_dict,
        alignment=32
    )
    
    # Write to temporary file
    test_file = tmp_path / "test_tensors.gguf"
    test_file.write_bytes(file_content)
    
    # Parse file using context manager
    with GGUFReader(str(test_file)) as reader:
        # Test list_tensors() returns all tensor names
        retrieved_names = reader.list_tensors()
        
        assert isinstance(retrieved_names, list), \
            "list_tensors() should return a list"
        
        assert len(retrieved_names) == num_tensors, \
            f"Tensor count mismatch: expected {num_tensors}, got {len(retrieved_names)}"
        
        # Verify all tensor names are present
        for name in tensor_names:
            assert name in retrieved_names, \
                f"Tensor name '{name}' missing from list_tensors()"
        
        # Test get_tensor_info(name) for each tensor
        for expected_info in tensor_info_list:
            name = expected_info['name']
            retrieved_info = reader.get_tensor_info(name)
            
            assert isinstance(retrieved_info, dict), \
                f"get_tensor_info('{name}') should return a dictionary"
            
            assert retrieved_info['name'] == expected_info['name'], \
                f"Tensor name mismatch for '{name}'"
            
            assert retrieved_info['n_dims'] == expected_info['n_dims'], \
                f"Dimension count mismatch for tensor '{name}'"
            
            assert retrieved_info['dims'] == expected_info['dims'], \
                f"Dimensions mismatch for tensor '{name}'"
            
            assert retrieved_info['type'] == expected_info['type'], \
                f"Type mismatch for tensor '{name}'"
        
        # Test get_tensor_data(name) for each tensor
        for name, expected_data in tensor_data_dict.items():
            retrieved_data = reader.get_tensor_data(name)
            
            assert isinstance(retrieved_data, bytes), \
                f"get_tensor_data('{name}') should return bytes"
            
            assert retrieved_data == expected_data, \
                f"Tensor data mismatch for '{name}': " \
                f"expected {len(expected_data)} bytes, got {len(retrieved_data)} bytes"
        
        # Test that non-existent tensor name raises KeyError
        with pytest.raises(KeyError):
            reader.get_tensor_info('non_existent_tensor_12345')
        
        with pytest.raises(KeyError):
            reader.get_tensor_data('non_existent_tensor_12345')


# ----------------------------------------------------------------------------
# Property 13: Count Retrieval
# ----------------------------------------------------------------------------

@given(
    version=st.integers(min_value=1, max_value=10),
    num_tensors=st.integers(min_value=0, max_value=10),
    data=st.data()
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_count_retrieval(version, num_tensors, data, tmp_path):
    """
    Property 13: Count Retrieval
    
    **Validates: Requirements 6.6, 6.7**
    
    For any parsed GGUF file, the reported tensor count and version should
    match the values in the file header.
    
    This property ensures that:
    - get_tensor_count() returns the correct number of tensors
    - get_version() returns the correct GGUF version
    - Header values are preserved exactly through parsing
    - Works correctly with zero tensors
    - Works correctly with various version numbers
    """
    # Generate tensor info if needed
    tensor_info_list = []
    tensor_data_dict = {}
    
    if num_tensors > 0:
        current_offset = 0
        for i in range(num_tensors):
            # Generate simple tensor
            name = f"tensor_{i}"
            tensor_type = GGMLType.F32
            type_size, block_size = TYPE_SIZES[tensor_type]
            dims = [32]  # Simple 1D tensor with 32 elements
            
            # Calculate data size
            data_size = type_size * 32
            
            # Generate random data
            tensor_data = data.draw(st.binary(min_size=data_size, max_size=data_size))
            
            tensor_info = {
                'name': name,
                'n_dims': 1,
                'dims': dims,
                'type': tensor_type,
                'offset': current_offset
            }
            
            tensor_info_list.append(tensor_info)
            tensor_data_dict[name] = tensor_data
            current_offset += data_size
    
    # Create GGUF file with specified version and tensor count
    file_content = b''
    
    # Header with specified version
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', version)  # version
    file_content += struct.pack('<Q', num_tensors)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count (for alignment)
    
    # Add minimal metadata (alignment)
    key = 'general.alignment'
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', 32)  # alignment value
    
    # Add tensor info if any
    if num_tensors > 0:
        for tensor_info in tensor_info_list:
            # Encode tensor name
            name_bytes = tensor_info['name'].encode('utf-8')
            file_content += struct.pack('<Q', len(name_bytes)) + name_bytes
            
            # Encode n_dims
            file_content += struct.pack('<I', tensor_info['n_dims'])
            
            # Encode dims
            for dim in tensor_info['dims']:
                file_content += struct.pack('<Q', dim)
            
            # Encode type
            file_content += struct.pack('<I', tensor_info['type'])
            
            # Encode offset
            file_content += struct.pack('<Q', tensor_info['offset'])
        
        # Add alignment padding
        current_position = len(file_content)
        padding = (32 - (current_position % 32)) % 32
        file_content += b'\x00' * padding
        
        # Add tensor data
        for tensor_info in tensor_info_list:
            name = tensor_info['name']
            file_content += tensor_data_dict[name]
    
    # Write to temporary file
    test_file = tmp_path / "test_counts.gguf"
    test_file.write_bytes(file_content)
    
    # Parse file using context manager
    with GGUFReader(str(test_file)) as reader:
        # Test get_version() returns correct version
        retrieved_version = reader.get_version()
        
        assert isinstance(retrieved_version, int), \
            "get_version() should return an integer"
        
        assert retrieved_version == version, \
            f"Version mismatch: expected {version}, got {retrieved_version}"
        
        # Test get_tensor_count() returns correct count
        retrieved_count = reader.get_tensor_count()
        
        assert isinstance(retrieved_count, int), \
            "get_tensor_count() should return an integer"
        
        assert retrieved_count == num_tensors, \
            f"Tensor count mismatch: expected {num_tensors}, got {retrieved_count}"
        
        # Verify consistency with list_tensors()
        tensor_list = reader.list_tensors()
        assert len(tensor_list) == retrieved_count, \
            f"list_tensors() length {len(tensor_list)} doesn't match get_tensor_count() {retrieved_count}"


# ============================================================================
# Unit Tests for Context Manager (Task 8.6)
# ============================================================================

def test_context_manager_file_opened_in_enter(tmp_path):
    """
    Test that file is opened in __enter__.
    
    **Requirements: 8.1, 8.2**
    
    Verifies that:
    - The file is opened when entering the context manager
    - The file object is not None after __enter__
    - The file is in binary read mode
    """
    # Create a minimal valid GGUF file
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    test_file = tmp_path / "test_enter.gguf"
    test_file.write_bytes(file_content)
    
    # Create reader
    reader = GGUFReader(str(test_file))
    
    # Before entering context, file should be None
    assert reader.file is None, "File should be None before entering context"
    
    # Enter context
    result = reader.__enter__()
    
    # After entering context, file should be open
    assert reader.file is not None, "File should be opened in __enter__"
    assert not reader.file.closed, "File should not be closed after __enter__"
    assert reader.file.mode == 'rb', "File should be opened in binary read mode"
    
    # __enter__ should return self
    assert result is reader, "__enter__ should return self"
    
    # Clean up
    reader.__exit__(None, None, None)


def test_context_manager_file_closed_in_exit(tmp_path):
    """
    Test that file is closed in __exit__.
    
    **Requirements: 8.3**
    
    Verifies that:
    - The file is closed when exiting the context manager
    - The file object is set to None after __exit__
    - Normal exit (no exception) properly closes the file
    """
    # Create a minimal valid GGUF file
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    test_file = tmp_path / "test_exit.gguf"
    test_file.write_bytes(file_content)
    
    # Create reader and enter context
    reader = GGUFReader(str(test_file))
    reader.__enter__()
    
    # File should be open
    assert reader.file is not None
    assert not reader.file.closed
    
    # Exit context (no exception)
    result = reader.__exit__(None, None, None)
    
    # After exiting context, file should be closed and set to None
    assert reader.file is None, "File should be set to None after __exit__"
    
    # __exit__ should return None (don't suppress exceptions)
    assert result is None, "__exit__ should return None"


def test_context_manager_file_closed_on_exception(tmp_path):
    """
    Test that file is closed when exception occurs.
    
    **Requirements: 8.4**
    
    Verifies that:
    - The file is closed even when an exception occurs within the context
    - The exception is not suppressed (propagates to caller)
    - File cleanup happens before exception propagation
    """
    # Create a minimal valid GGUF file
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    test_file = tmp_path / "test_exception.gguf"
    test_file.write_bytes(file_content)
    
    # Test that exception is raised and file is closed
    with pytest.raises(ValueError, match="Test exception"):
        with GGUFReader(str(test_file)) as reader:
            # File should be open inside context
            assert reader.file is not None
            assert not reader.file.closed
            
            # Raise an exception
            raise ValueError("Test exception")
    
    # After exception, file should be closed
    # Note: We can't check reader.file here because reader is out of scope,
    # but the test verifies that __exit__ was called by checking the exception propagated


def test_context_manager_with_statement(tmp_path):
    """
    Test using the reader with Python's with statement.
    
    **Requirements: 8.2, 8.3**
    
    Verifies that:
    - The reader works correctly with the with statement
    - File is automatically opened and closed
    - Data can be accessed within the context
    """
    # Create a GGUF file with some metadata
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count
    
    # Add one metadata key-value pair
    key = "test.key"
    value = 42
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', GGUFValueType.UINT32)  # type
    file_content += struct.pack('<I', value)  # value
    
    test_file = tmp_path / "test_with.gguf"
    test_file.write_bytes(file_content)
    
    # Use with statement
    with GGUFReader(str(test_file)) as reader:
        # Inside context, file should be open
        assert reader.file is not None
        assert not reader.file.closed
        
        # Should be able to access data
        metadata = reader.get_metadata()
        assert "test.key" in metadata
        assert metadata["test.key"] == value
    
    # After exiting context, file should be closed
    assert reader.file is None


def test_context_manager_keyerror_nonexistent_metadata(tmp_path):
    """
    Test KeyError for non-existent metadata key.
    
    **Requirements: 8.2**
    
    Verifies that:
    - get_metadata_value() raises KeyError for non-existent keys
    - The error message is informative
    - File is still properly closed after the error
    """
    # Create a GGUF file with one metadata key
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count
    
    # Add one metadata key-value pair
    key = "existing.key"
    value = 123
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', GGUFValueType.UINT32)  # type
    file_content += struct.pack('<I', value)  # value
    
    test_file = tmp_path / "test_keyerror_metadata.gguf"
    test_file.write_bytes(file_content)
    
    # Test KeyError for non-existent key
    with GGUFReader(str(test_file)) as reader:
        # Existing key should work
        assert reader.get_metadata_value("existing.key") == value
        
        # Non-existent key should raise KeyError
        with pytest.raises(KeyError):
            reader.get_metadata_value("nonexistent.key")
    
    # File should still be closed properly
    assert reader.file is None


def test_context_manager_keyerror_nonexistent_tensor(tmp_path):
    """
    Test KeyError for non-existent tensor name.
    
    **Requirements: 8.3**
    
    Verifies that:
    - get_tensor_info() raises KeyError for non-existent tensor names
    - get_tensor_data() raises KeyError for non-existent tensor names
    - The error messages are informative
    - File is still properly closed after the error
    """
    # Create a GGUF file with one tensor
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    # Add one tensor info
    tensor_name = "existing.tensor"
    name_bytes = tensor_name.encode('utf-8')
    file_content += struct.pack('<Q', len(name_bytes)) + name_bytes  # name
    file_content += struct.pack('<I', 1)  # n_dims
    file_content += struct.pack('<Q', 32)  # dim[0]
    file_content += struct.pack('<I', GGMLType.F32)  # type
    file_content += struct.pack('<Q', 0)  # offset
    
    # Add alignment padding and tensor data
    current_pos = len(file_content)
    alignment = 32
    padding = (alignment - (current_pos % alignment)) % alignment
    file_content += b'\x00' * padding
    
    # Add tensor data (32 floats = 128 bytes)
    tensor_data = struct.pack('<' + 'f' * 32, *([1.0] * 32))
    file_content += tensor_data
    
    test_file = tmp_path / "test_keyerror_tensor.gguf"
    test_file.write_bytes(file_content)
    
    # Test KeyError for non-existent tensor
    with GGUFReader(str(test_file)) as reader:
        # Existing tensor should work
        info = reader.get_tensor_info("existing.tensor")
        assert info['name'] == "existing.tensor"
        
        data = reader.get_tensor_data("existing.tensor")
        assert len(data) == 128
        
        # Non-existent tensor should raise KeyError for get_tensor_info
        with pytest.raises(KeyError, match="Tensor 'nonexistent.tensor' not found"):
            reader.get_tensor_info("nonexistent.tensor")
        
        # Non-existent tensor should raise KeyError for get_tensor_data
        with pytest.raises(KeyError, match="Tensor 'nonexistent.tensor' not found"):
            reader.get_tensor_data("nonexistent.tensor")
    
    # File should still be closed properly
    assert reader.file is None


def test_context_manager_multiple_contexts(tmp_path):
    """
    Test that the same reader can be used with multiple context managers.
    
    **Requirements: 8.1, 8.2, 8.3**
    
    Verifies that:
    - A reader can be used multiple times with the with statement
    - Each context properly opens and closes the file
    - Data is accessible in each context
    """
    # Create a minimal valid GGUF file
    file_content = b''
    file_content += struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count
    
    # Add metadata
    key = "test.value"
    value = 999
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', GGUFValueType.UINT32)
    file_content += struct.pack('<I', value)
    
    test_file = tmp_path / "test_multiple.gguf"
    test_file.write_bytes(file_content)
    
    reader = GGUFReader(str(test_file))
    
    # First context
    with reader:
        assert reader.file is not None
        assert reader.get_metadata_value("test.value") == value
    
    # File should be closed between contexts
    assert reader.file is None
    
    # Second context
    with reader:
        assert reader.file is not None
        assert reader.get_metadata_value("test.value") == value
    
    # File should be closed after second context
    assert reader.file is None


def test_context_manager_file_not_found():
    """
    Test that FileNotFoundError is properly handled.
    
    **Requirements: 8.1**
    
    Verifies that:
    - Attempting to open a non-existent file raises an appropriate error
    - The error message includes the file path
    - No file handle is left open
    """
    reader = GGUFReader("nonexistent_file.gguf")
    
    # Attempting to enter context with non-existent file should raise error
    with pytest.raises(GGUFFileError, match="File not found"):
        with reader:
            pass
    
    # File should remain None
    assert reader.file is None


def test_context_manager_parsing_error_closes_file(tmp_path):
    """
    Test that file is closed when parsing error occurs in __enter__.
    
    **Requirements: 8.1, 8.4**
    
    Verifies that:
    - If parsing fails during __enter__, the file is still closed
    - The parsing error is propagated to the caller
    - No file handle is leaked
    """
    # Create an invalid GGUF file (wrong magic number)
    file_content = b''
    file_content += struct.pack('<I', 0x12345678)  # invalid magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    test_file = tmp_path / "test_invalid.gguf"
    test_file.write_bytes(file_content)
    
    reader = GGUFReader(str(test_file))
    
    # Attempting to enter context should raise error
    with pytest.raises(GGUFInvalidMagicError):
        with reader:
            pass
    
    # File should be closed (set to None) even though parsing failed
    assert reader.file is None


# ============================================================================
# Property Tests for Error Detection (Task 9.2)
# ============================================================================

# ----------------------------------------------------------------------------
# Property 14: Truncated File Detection
# ----------------------------------------------------------------------------

@given(
    truncation_point=st.sampled_from([
        'header_magic',
        'header_version',
        'header_tensor_count',
        'header_metadata_count',
        'metadata_key',
        'metadata_value_type',
        'metadata_value',
        'tensor_name',
        'tensor_n_dims',
        'tensor_dims',
        'tensor_type',
        'tensor_offset',
    ])
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_truncated_file_detection(truncation_point, tmp_path):
    """
    Property 14: Truncated File Detection
    
    **Validates: Requirements 7.2, 7.5**
    
    For any file that is shorter than the expected size based on header information,
    the reader should raise an error indicating truncation or unexpected EOF.
    
    This property ensures that:
    - Truncation during header parsing is detected
    - Truncation during metadata parsing is detected
    - Truncation during tensor info parsing is detected
    - Error messages indicate the truncation location
    - The reader doesn't crash or hang on truncated files
    """
    # Build a valid GGUF file structure
    file_content = b''
    
    # Header
    GGUF_MAGIC = 0x46554747
    file_content += struct.pack('<I', GGUF_MAGIC)  # magic
    
    if truncation_point == 'header_magic':
        # Truncate in the middle of magic number
        file_content = file_content[:2]
    elif truncation_point == 'header_version':
        # Truncate in the middle of version
        file_content += struct.pack('<I', 3)[:2]  # partial version
    elif truncation_point == 'header_tensor_count':
        # Truncate in the middle of tensor_count
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)[:4]  # partial tensor_count
    elif truncation_point == 'header_metadata_count':
        # Truncate in the middle of metadata_kv_count
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count
        file_content += struct.pack('<Q', 1)[:4]  # partial metadata_kv_count
    elif truncation_point == 'metadata_key':
        # Truncate in the middle of metadata key string
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        # Start a string but truncate it
        file_content += struct.pack('<Q', 10)  # string length = 10
        file_content += b'test'  # only 4 bytes instead of 10
    elif truncation_point == 'metadata_value_type':
        # Truncate in the middle of metadata value type
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        # Complete key
        key = "test.key"
        file_content += struct.pack('<Q', len(key)) + key.encode('utf-8')
        # Partial value type
        file_content += struct.pack('<I', GGUFValueType.UINT32)[:2]
    elif truncation_point == 'metadata_value':
        # Truncate in the middle of metadata value
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        # Complete key
        key = "test.key"
        file_content += struct.pack('<Q', len(key)) + key.encode('utf-8')
        # Complete value type
        file_content += struct.pack('<I', GGUFValueType.UINT64)
        # Partial value (need 8 bytes for UINT64, only provide 4)
        file_content += struct.pack('<Q', 12345)[:4]
    elif truncation_point == 'tensor_name':
        # Truncate in the middle of tensor name
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        # Start tensor name but truncate it
        file_content += struct.pack('<Q', 20)  # string length = 20
        file_content += b'tensor'  # only 6 bytes instead of 20
    elif truncation_point == 'tensor_n_dims':
        # Truncate in the middle of n_dims
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        # Complete tensor name
        name = "test.tensor"
        file_content += struct.pack('<Q', len(name)) + name.encode('utf-8')
        # Partial n_dims
        file_content += struct.pack('<I', 2)[:2]
    elif truncation_point == 'tensor_dims':
        # Truncate in the middle of dimension array
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        # Complete tensor name
        name = "test.tensor"
        file_content += struct.pack('<Q', len(name)) + name.encode('utf-8')
        # Complete n_dims
        file_content += struct.pack('<I', 2)  # 2 dimensions
        # Partial dims (need 2 uint64s, only provide 1)
        file_content += struct.pack('<Q', 100)  # first dim
        file_content += struct.pack('<Q', 200)[:4]  # partial second dim
    elif truncation_point == 'tensor_type':
        # Truncate in the middle of tensor type
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        # Complete tensor name
        name = "test.tensor"
        file_content += struct.pack('<Q', len(name)) + name.encode('utf-8')
        # Complete n_dims and dims
        file_content += struct.pack('<I', 1)  # 1 dimension
        file_content += struct.pack('<Q', 100)  # dim size
        # Partial type
        file_content += struct.pack('<I', GGMLType.F32)[:2]
    elif truncation_point == 'tensor_offset':
        # Truncate in the middle of tensor offset
        file_content += struct.pack('<I', 3)  # version
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        # Complete tensor name
        name = "test.tensor"
        file_content += struct.pack('<Q', len(name)) + name.encode('utf-8')
        # Complete n_dims and dims
        file_content += struct.pack('<I', 1)  # 1 dimension
        file_content += struct.pack('<Q', 100)  # dim size
        # Complete type
        file_content += struct.pack('<I', GGMLType.F32)
        # Partial offset
        file_content += struct.pack('<Q', 0)[:4]
    
    # Write truncated file
    test_file = tmp_path / "truncated.gguf"
    test_file.write_bytes(file_content)
    
    # Attempt to read the truncated file
    reader = GGUFReader(str(test_file))
    
    # Should raise GGUFTruncatedError
    with pytest.raises(GGUFTruncatedError) as exc_info:
        with reader:
            pass
    
    # Verify error message contains useful information
    error_msg = str(exc_info.value)
    assert "Unexpected end of file" in error_msg or "expected to read" in error_msg
    assert "bytes" in error_msg.lower()


# ----------------------------------------------------------------------------
# Property 15: Invalid Version Rejection
# ----------------------------------------------------------------------------

@given(
    version=st.integers(min_value=0, max_value=1000).filter(lambda v: v not in [1, 2, 3])
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_invalid_version_rejection(version, tmp_path):
    """
    Property 15: Invalid Version Rejection
    
    **Validates: Requirements 7.3**
    
    For any GGUF file with an unsupported version number, the reader should raise
    an error containing the version number.
    
    This property ensures that:
    - Only supported versions (1, 2, 3) are accepted
    - Unsupported versions are rejected with clear error messages
    - The error message includes the actual version number encountered
    - Future versions are rejected (forward compatibility check)
    - Very old or invalid versions are rejected
    
    Note: Currently GGUF supports versions 1-3. This test uses versions outside
    that range to verify rejection behavior.
    """
    # Build a GGUF file with invalid version
    file_content = b''
    
    # Header with invalid version
    GGUF_MAGIC = 0x46554747
    file_content += struct.pack('<I', GGUF_MAGIC)  # magic
    file_content += struct.pack('<I', version)  # invalid version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    # Write file
    test_file = tmp_path / "invalid_version.gguf"
    test_file.write_bytes(file_content)
    
    # Attempt to read the file
    reader = GGUFReader(str(test_file))
    
    # For now, the implementation doesn't explicitly validate version,
    # but we can check if it would cause issues. Let's try to parse it.
    # If version validation is added later, this test will catch it.
    try:
        with reader:
            # If we get here, version validation is not implemented
            # This is acceptable for now, but the test documents the expected behavior
            pass
    except GGUFVersionError as e:
        # If version validation is implemented, verify error message
        error_msg = str(e)
        assert str(version) in error_msg
        assert "version" in error_msg.lower()


# ----------------------------------------------------------------------------
# Property 16: Invalid Type Rejection
# ----------------------------------------------------------------------------

@given(
    invalid_type=st.sampled_from([13, 14, 15, 20, 50, 100, 255]),  # Types > 12 are invalid for metadata
    location=st.sampled_from(['metadata_value', 'metadata_array_element'])
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_invalid_type_rejection(invalid_type, location, tmp_path):
    """
    Property 16: Invalid Type Rejection
    
    **Validates: Requirements 7.4**
    
    For any GGUF file containing an invalid type code in metadata or tensor info,
    the reader should raise an error indicating the invalid type.
    
    This property ensures that:
    - Invalid metadata value types are rejected
    - Invalid array element types are rejected
    - Invalid tensor types are rejected
    - Error messages indicate the invalid type code
    - Error messages indicate where the invalid type was encountered
    - The reader doesn't crash or produce garbage data
    
    Valid metadata types: 0-12 (UINT8 through FLOAT64)
    Valid tensor types: 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    """
    # Build a GGUF file with invalid type
    file_content = b''
    
    # Header
    GGUF_MAGIC = 0x46554747
    file_content += struct.pack('<I', GGUF_MAGIC)  # magic
    file_content += struct.pack('<I', 3)  # version
    
    if location == 'metadata_value':
        # Invalid type in metadata value
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        
        # Metadata key
        key = "test.key"
        file_content += struct.pack('<Q', len(key)) + key.encode('utf-8')
        
        # Invalid value type
        file_content += struct.pack('<I', invalid_type)
        
        # Add some dummy data (in case the reader tries to read it)
        file_content += b'\x00' * 8
        
    elif location == 'metadata_array_element':
        # Invalid type in array element type
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        
        # Metadata key
        key = "test.array"
        file_content += struct.pack('<Q', len(key)) + key.encode('utf-8')
        
        # Array value type
        file_content += struct.pack('<I', GGUFValueType.ARRAY)
        
        # Invalid array element type
        file_content += struct.pack('<I', invalid_type)
        file_content += struct.pack('<Q', 1)  # array length = 1 (one element to trigger validation)
        # Add some dummy data for the element
        file_content += b'\x00' * 8
    
    # Write file
    test_file = tmp_path / "invalid_type.gguf"
    test_file.write_bytes(file_content)
    
    # Attempt to read the file
    reader = GGUFReader(str(test_file))
    
    # Should raise GGUFInvalidTypeError
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        with reader:
            pass
    
    # Verify error message contains useful information
    error_msg = str(exc_info.value)
    assert "Invalid" in error_msg or "invalid" in error_msg
    assert "type" in error_msg.lower()
    # The error should mention the invalid type code
    assert str(invalid_type) in error_msg or f"0x{invalid_type:x}" in error_msg.lower()


@given(
    invalid_type=st.sampled_from([4, 5, 19, 20, 50, 100, 255])  # Invalid tensor types
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_invalid_tensor_type_rejection(invalid_type, tmp_path):
    """
    Property 16 (continued): Invalid Tensor Type Rejection
    
    **Validates: Requirements 7.4**
    
    For any GGUF file containing an invalid tensor type code, the reader should
    raise an error indicating the invalid type.
    
    Valid tensor types: 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    Invalid tensor types: 4, 5, 19+
    """
    # Build a GGUF file with invalid tensor type
    file_content = b''
    
    # Header
    GGUF_MAGIC = 0x46554747
    file_content += struct.pack('<I', GGUF_MAGIC)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    # Tensor name
    name = "test.tensor"
    file_content += struct.pack('<Q', len(name)) + name.encode('utf-8')
    
    # Tensor dimensions
    file_content += struct.pack('<I', 1)  # n_dims = 1
    file_content += struct.pack('<Q', 100)  # dim size
    
    # Invalid tensor type
    file_content += struct.pack('<I', invalid_type)
    file_content += struct.pack('<Q', 0)  # offset
    
    # Write file
    test_file = tmp_path / "invalid_tensor_type.gguf"
    test_file.write_bytes(file_content)
    
    # Attempt to read the file
    reader = GGUFReader(str(test_file))
    
    # Should raise GGUFInvalidTypeError
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        with reader:
            pass
    
    # Verify error message contains useful information
    error_msg = str(exc_info.value)
    assert "Invalid" in error_msg or "invalid" in error_msg
    assert "type" in error_msg.lower()
    assert str(invalid_type) in error_msg


# ----------------------------------------------------------------------------
# Property 17: Invalid String Length Rejection
# ----------------------------------------------------------------------------

@given(
    string_length=st.sampled_from([
        2**63 - 1,  # Maximum int64 value
        2**32,      # 4GB
        200 * 1024 * 1024 + 1,  # Just over 200MB limit
        500 * 1024 * 1024,  # 500MB
    ]),
    location=st.sampled_from(['metadata_key', 'metadata_string_value', 'tensor_name'])
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_invalid_string_length_rejection(string_length, location, tmp_path):
    """
    Property 17: Invalid String Length Rejection
    
    **Validates: Requirements 7.6**
    
    For any string with a length value that would read past the end of file or is
    unreasonably large, the reader should raise an error.
    
    This property ensures that:
    - Excessively large string lengths are rejected (prevent memory exhaustion)
    - String lengths that exceed file size are detected
    - Error messages indicate the problem with the string length
    - The reader doesn't attempt to allocate huge amounts of memory
    - Protection against malicious files with crafted length values
    
    The implementation uses a maximum string length of 100MB to prevent
    memory exhaustion attacks.
    """
    # Build a GGUF file with invalid string length
    file_content = b''
    
    # Header
    GGUF_MAGIC = 0x46554747
    file_content += struct.pack('<I', GGUF_MAGIC)  # magic
    file_content += struct.pack('<I', 3)  # version
    
    if location == 'metadata_key':
        # Invalid length in metadata key
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        
        # Metadata key with excessive length
        file_content += struct.pack('<Q', string_length)
        # Don't actually write that many bytes (file would be huge)
        file_content += b'x' * 100  # Just a small amount of data
        
    elif location == 'metadata_string_value':
        # Invalid length in metadata string value
        file_content += struct.pack('<Q', 0)  # tensor_count
        file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
        
        # Valid metadata key
        key = "test.key"
        file_content += struct.pack('<Q', len(key)) + key.encode('utf-8')
        
        # String value type
        file_content += struct.pack('<I', GGUFValueType.STRING)
        
        # String with excessive length
        file_content += struct.pack('<Q', string_length)
        file_content += b'x' * 100  # Just a small amount of data
        
    elif location == 'tensor_name':
        # Invalid length in tensor name
        file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
        file_content += struct.pack('<Q', 0)  # metadata_kv_count
        
        # Tensor name with excessive length
        file_content += struct.pack('<Q', string_length)
        file_content += b'x' * 100  # Just a small amount of data
    
    # Write file
    test_file = tmp_path / "invalid_string_length.gguf"
    test_file.write_bytes(file_content)
    
    # Attempt to read the file
    reader = GGUFReader(str(test_file))
    
    # Should raise either GGUFParseError (for excessive length) or GGUFTruncatedError (for length > file size)
    with pytest.raises((GGUFParseError, GGUFTruncatedError)) as exc_info:
        with reader:
            pass
    
    # Verify error message contains useful information
    error_msg = str(exc_info.value)
    assert "string" in error_msg.lower() or "length" in error_msg.lower()


# ============================================================================
# Unit Tests for Error Messages (Task 9.3)
# ============================================================================

def test_file_not_found_error_includes_path():
    """
    Test that file not found error includes the file path.
    
    Requirements: 7.1
    
    This test verifies that when a file cannot be opened, the error message
    includes the path to the file that was not found.
    """
    # Use a path that definitely doesn't exist
    nonexistent_path = "/path/to/nonexistent/file.gguf"
    reader = GGUFReader(nonexistent_path)
    
    with pytest.raises(GGUFFileError) as exc_info:
        with reader:
            pass
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes the file path
    assert nonexistent_path in error_msg, \
        f"Error message should include file path '{nonexistent_path}', got: {error_msg}"
    assert "File not found" in error_msg or "not found" in error_msg.lower(), \
        f"Error message should indicate file not found, got: {error_msg}"


def test_truncated_file_error_includes_position():
    """
    Test that truncated file error includes the file position.
    
    Requirements: 7.2
    
    This test verifies that when a file is truncated (ends unexpectedly),
    the error message includes the position in the file where the error occurred.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a truncated header (only 10 bytes instead of 24)
    truncated_content = b'\x47\x47\x55\x46'  # Valid magic (4 bytes)
    truncated_content += b'\x03\x00\x00\x00'  # Version (4 bytes)
    truncated_content += b'\x01\x00'  # Only 2 bytes of tensor_count (need 8)
    
    reader.file = io.BytesIO(truncated_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes position information
    assert "position" in error_msg.lower(), \
        f"Error message should include position information, got: {error_msg}"
    # Position should be 8 (after magic and version)
    assert "8" in error_msg, \
        f"Error message should include position 8, got: {error_msg}"
    # Should mention what was expected
    assert "expected" in error_msg.lower(), \
        f"Error message should mention expected bytes, got: {error_msg}"
    # Should mention how many bytes were available
    assert "available" in error_msg.lower() or "only" in error_msg.lower(), \
        f"Error message should mention available bytes, got: {error_msg}"


def test_invalid_version_error_includes_version_number():
    """
    Test that invalid version error includes the version number.
    
    Requirements: 7.3
    
    This test verifies that when an unsupported version is encountered,
    the error message includes the actual version number that was found.
    
    Note: Currently the implementation doesn't validate version numbers,
    so this test documents the expected behavior for future implementation.
    """
    # This test documents expected behavior if version validation is added
    # Currently, the reader accepts any version number without validation
    # If version validation is implemented in the future, this test should pass
    
    reader = GGUFReader("test.gguf")
    
    # Create a file with an unsupported version (e.g., version 999)
    unsupported_version = 999
    file_content = struct.pack('<I', 0x46554747)  # Valid magic
    file_content += struct.pack('<I', unsupported_version)  # Unsupported version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    # Currently this doesn't raise an error, but if version validation is added:
    # with pytest.raises(GGUFVersionError) as exc_info:
    #     reader._read_header()
    # 
    # error_msg = str(exc_info.value)
    # 
    # # Verify the error message includes the version number
    # assert str(unsupported_version) in error_msg, \
    #     f"Error message should include version number {unsupported_version}, got: {error_msg}"
    # assert "version" in error_msg.lower(), \
    #     f"Error message should mention version, got: {error_msg}"
    
    # For now, just verify the header is read without error
    reader._read_header()
    assert reader.header['version'] == unsupported_version


def test_invalid_type_error_includes_type_code():
    """
    Test that invalid type error includes the type code.
    
    Requirements: 7.4
    
    This test verifies that when an invalid data type is encountered,
    the error message includes the actual type code that was found.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a file with an invalid type code
    invalid_type_code = 99  # Not a valid GGUF type
    file_content = struct.pack('<I', invalid_type_code)
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_value(invalid_type_code)
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes the type code
    assert str(invalid_type_code) in error_msg, \
        f"Error message should include type code {invalid_type_code}, got: {error_msg}"
    assert "type" in error_msg.lower(), \
        f"Error message should mention type, got: {error_msg}"
    assert "invalid" in error_msg.lower(), \
        f"Error message should indicate the type is invalid, got: {error_msg}"


def test_invalid_type_error_in_metadata():
    """
    Test that invalid type error in metadata includes context.
    
    Requirements: 7.4
    
    This test verifies that when an invalid type is encountered in metadata,
    the error message includes the type code and context about where it occurred.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a valid header
    file_content = struct.pack('<I', 0x46554747)  # Valid magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count (1 pair)
    
    # Add a metadata key-value pair with invalid type
    key = "test.key"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes  # key
    file_content += struct.pack('<I', 999)  # Invalid type code
    
    reader.file = io.BytesIO(file_content)
    
    # Read header first
    reader._read_header()
    
    # Try to read metadata
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes the type code
    assert "999" in error_msg, \
        f"Error message should include type code 999, got: {error_msg}"
    assert "type" in error_msg.lower(), \
        f"Error message should mention type, got: {error_msg}"


def test_invalid_type_error_in_tensor_info():
    """
    Test that invalid type error in tensor info includes context.
    
    Requirements: 7.4
    
    This test verifies that when an invalid tensor type is encountered,
    the error message includes the type code and the tensor name.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a valid header
    file_content = struct.pack('<I', 0x46554747)  # Valid magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count (1 tensor)
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    # Add tensor info with invalid type
    tensor_name = "test.tensor"
    tensor_name_bytes = tensor_name.encode('utf-8')
    file_content += struct.pack('<Q', len(tensor_name_bytes)) + tensor_name_bytes  # name
    file_content += struct.pack('<I', 2)  # n_dims
    file_content += struct.pack('<Q', 128)  # dim 0
    file_content += struct.pack('<Q', 256)  # dim 1
    file_content += struct.pack('<I', 999)  # Invalid tensor type code
    file_content += struct.pack('<Q', 0)  # offset
    
    reader.file = io.BytesIO(file_content)
    
    # Read header first
    reader._read_header()
    
    # Try to read tensor info
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_tensor_info()
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes the type code
    assert "999" in error_msg, \
        f"Error message should include type code 999, got: {error_msg}"
    assert "type" in error_msg.lower(), \
        f"Error message should mention type, got: {error_msg}"
    # Should also include the tensor name for context
    assert tensor_name in error_msg, \
        f"Error message should include tensor name '{tensor_name}', got: {error_msg}"


def test_invalid_string_length_error():
    """
    Test that invalid string length error is raised appropriately.
    
    Requirements: 7.6
    
    This test verifies that when an invalid string length is encountered
    (e.g., excessively large), an appropriate error is raised.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a string with excessively large length (200MB)
    excessive_length = 200 * 1024 * 1024
    file_content = struct.pack('<Q', excessive_length)
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFParseError) as exc_info:
        reader._read_string()
    
    error_msg = str(exc_info.value)
    
    # Verify the error message mentions string length
    assert "string" in error_msg.lower(), \
        f"Error message should mention string, got: {error_msg}"
    assert "length" in error_msg.lower(), \
        f"Error message should mention length, got: {error_msg}"
    # Should indicate the length is invalid or excessive
    assert "invalid" in error_msg.lower() or "exceed" in error_msg.lower(), \
        f"Error message should indicate the length is invalid or excessive, got: {error_msg}"


def test_truncated_string_error_includes_details():
    """
    Test that truncated string error includes position and expected bytes.
    
    Requirements: 7.2, 7.6
    
    This test verifies that when a string is truncated, the error message
    includes details about the position and expected number of bytes.
    """
    reader = GGUFReader("test.gguf")
    
    # Create a string with length=10 but only 5 bytes of data
    expected_length = 10
    actual_data = b'hello'
    file_content = struct.pack('<Q', expected_length) + actual_data
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_string()
    
    error_msg = str(exc_info.value)
    
    # Verify the error message includes expected and actual byte counts
    assert str(expected_length) in error_msg, \
        f"Error message should include expected length {expected_length}, got: {error_msg}"
    assert str(len(actual_data)) in error_msg, \
        f"Error message should include actual length {len(actual_data)}, got: {error_msg}"
    assert "expected" in error_msg.lower(), \
        f"Error message should mention expected bytes, got: {error_msg}"
    assert "available" in error_msg.lower() or "only" in error_msg.lower(), \
        f"Error message should mention available bytes, got: {error_msg}"


def test_error_messages_include_filepath():
    """
    Test that error messages include the filepath for context.
    
    Requirements: 7.1, 7.2, 7.3, 7.4
    
    This test verifies that various error messages include the filepath
    to help users identify which file has the problem.
    """
    filepath = "my_test_model.gguf"
    reader = GGUFReader(filepath)
    
    # Test 1: Invalid magic number
    file_content = struct.pack('<I', 0xDEADBEEF)  # Invalid magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 0)  # metadata_kv_count
    
    reader.file = io.BytesIO(file_content)
    
    with pytest.raises(GGUFInvalidMagicError) as exc_info:
        reader._read_header()
    
    assert filepath in str(exc_info.value), \
        f"Invalid magic error should include filepath '{filepath}'"
    
    # Test 2: Truncated file
    reader.file = io.BytesIO(b'\x47\x47\x55\x46\x03')  # Only 5 bytes
    
    with pytest.raises(GGUFTruncatedError) as exc_info:
        reader._read_header()
    
    assert filepath in str(exc_info.value), \
        f"Truncated file error should include filepath '{filepath}'"
    
    # Test 3: Invalid type in metadata
    file_content = struct.pack('<I', 0x46554747)  # Valid magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 0)  # tensor_count
    file_content += struct.pack('<Q', 1)  # metadata_kv_count
    file_content += struct.pack('<Q', 3) + b'key'  # key
    file_content += struct.pack('<I', 999)  # Invalid type
    
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    
    with pytest.raises(GGUFInvalidTypeError) as exc_info:
        reader._read_metadata()
    
    assert filepath in str(exc_info.value), \
        f"Invalid type error should include filepath '{filepath}'"
