"""
Unit tests for tensor data edge cases (Task 7.5).

This module contains unit tests for:
- All standard types (F32, F16, I8, I16, I32)
- All quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
- Tensors with different dimension counts

Requirements: 5.1, 5.2
"""

import struct
import io
import pytest

from gguf_reader import (
    GGUFReader,
    GGMLType,
    TYPE_SIZES,
)


def create_gguf_file_with_tensor(tensor_name, tensor_type, dims, tensor_data, alignment=32):
    """
    Helper function to create a complete GGUF file with a single tensor.
    
    Args:
        tensor_name: Name of the tensor
        tensor_type: GGML type code
        dims: List of dimension sizes
        tensor_data: Raw tensor data bytes
        alignment: Alignment for tensor data section (default 32)
        
    Returns:
        Bytes representing a complete GGUF file
    """
    # Header
    file_content = struct.pack('<I', 0x46554747)  # magic
    file_content += struct.pack('<I', 3)  # version
    file_content += struct.pack('<Q', 1)  # tensor_count = 1
    file_content += struct.pack('<Q', 1)  # metadata_kv_count = 1
    
    # Metadata: general.alignment
    key = "general.alignment"
    key_bytes = key.encode('utf-8')
    file_content += struct.pack('<Q', len(key_bytes)) + key_bytes
    file_content += struct.pack('<I', 4)  # UINT32 type
    file_content += struct.pack('<I', alignment)
    
    # Tensor info
    name_bytes = tensor_name.encode('utf-8')
    file_content += struct.pack('<Q', len(name_bytes)) + name_bytes
    file_content += struct.pack('<I', len(dims))  # n_dims
    for dim in dims:
        file_content += struct.pack('<Q', dim)
    file_content += struct.pack('<I', tensor_type)
    file_content += struct.pack('<Q', 0)  # offset = 0
    
    # Calculate alignment padding
    current_pos = len(file_content)
    padding = (alignment - (current_pos % alignment)) % alignment
    file_content += b'\x00' * padding
    
    # Tensor data
    file_content += tensor_data
    
    return file_content


# ============================================================================
# Test All Standard Types (F32, F16, I8, I16, I32)
# ============================================================================

def test_tensor_data_f32_type():
    """Test reading tensor data for F32 (32-bit float) type."""
    tensor_type = GGMLType.F32
    dims = [4]  # 4 elements
    
    # F32: 4 bytes per element, block size 1
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create some F32 data
    tensor_data = struct.pack('<ffff', 1.0, 2.5, -3.14, 0.0)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_f32', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_f32')
    assert actual_data == tensor_data


def test_tensor_data_f16_type():
    """Test reading tensor data for F16 (16-bit float) type."""
    tensor_type = GGMLType.F16
    dims = [8]  # 8 elements
    
    # F16: 2 bytes per element, block size 1
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create some F16 data (just raw bytes, not actual float16 encoding)
    tensor_data = b'\x00\x3c\x00\x40\x00\x42\x00\x44\x00\x46\x00\x48\x00\x4a\x00\x4c'
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_f16', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_f16')
    assert actual_data == tensor_data


def test_tensor_data_i8_type():
    """Test reading tensor data for I8 (8-bit integer) type."""
    tensor_type = GGMLType.I8
    dims = [16]  # 16 elements
    
    # I8: 1 byte per element, block size 1
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create some I8 data
    tensor_data = struct.pack('<16b', -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 127)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_i8', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_i8')
    assert actual_data == tensor_data


def test_tensor_data_i16_type():
    """Test reading tensor data for I16 (16-bit integer) type."""
    tensor_type = GGMLType.I16
    dims = [8]  # 8 elements
    
    # I16: 2 bytes per element, block size 1
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create some I16 data
    tensor_data = struct.pack('<8h', -32768, -1000, -100, -1, 0, 1, 100, 32767)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_i16', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_i16')
    assert actual_data == tensor_data


def test_tensor_data_i32_type():
    """Test reading tensor data for I32 (32-bit integer) type."""
    tensor_type = GGMLType.I32
    dims = [4]  # 4 elements
    
    # I32: 4 bytes per element, block size 1
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create some I32 data
    tensor_data = struct.pack('<4i', -2147483648, -1000000, 1000000, 2147483647)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_i32', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_i32')
    assert actual_data == tensor_data


# ============================================================================
# Test All Quantized Types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
# ============================================================================

def test_tensor_data_q4_0_type():
    """Test reading tensor data for Q4_0 quantized type."""
    tensor_type = GGMLType.Q4_0
    dims = [64]  # 64 elements = 2 blocks of 32
    
    # Q4_0: 18 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q4_0 data (2 blocks * 18 bytes = 36 bytes)
    tensor_data = b'\x00' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q4_0', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q4_0')
    assert actual_data == tensor_data
    assert len(actual_data) == 36


def test_tensor_data_q4_1_type():
    """Test reading tensor data for Q4_1 quantized type."""
    tensor_type = GGMLType.Q4_1
    dims = [32]  # 32 elements = 1 block
    
    # Q4_1: 20 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q4_1 data (1 block * 20 bytes = 20 bytes)
    tensor_data = b'\x01' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q4_1', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q4_1')
    assert actual_data == tensor_data
    assert len(actual_data) == 20


def test_tensor_data_q5_0_type():
    """Test reading tensor data for Q5_0 quantized type."""
    tensor_type = GGMLType.Q5_0
    dims = [96]  # 96 elements = 3 blocks of 32
    
    # Q5_0: 22 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q5_0 data (3 blocks * 22 bytes = 66 bytes)
    tensor_data = b'\x02' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q5_0', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q5_0')
    assert actual_data == tensor_data
    assert len(actual_data) == 66


def test_tensor_data_q5_1_type():
    """Test reading tensor data for Q5_1 quantized type."""
    tensor_type = GGMLType.Q5_1
    dims = [32]  # 32 elements = 1 block
    
    # Q5_1: 24 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q5_1 data (1 block * 24 bytes = 24 bytes)
    tensor_data = b'\x03' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q5_1', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q5_1')
    assert actual_data == tensor_data
    assert len(actual_data) == 24


def test_tensor_data_q8_0_type():
    """Test reading tensor data for Q8_0 quantized type."""
    tensor_type = GGMLType.Q8_0
    dims = [64]  # 64 elements = 2 blocks of 32
    
    # Q8_0: 34 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q8_0 data (2 blocks * 34 bytes = 68 bytes)
    tensor_data = b'\x04' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q8_0', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q8_0')
    assert actual_data == tensor_data
    assert len(actual_data) == 68


def test_tensor_data_q8_1_type():
    """Test reading tensor data for Q8_1 quantized type."""
    tensor_type = GGMLType.Q8_1
    dims = [32]  # 32 elements = 1 block
    
    # Q8_1: 40 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q8_1 data (1 block * 40 bytes = 40 bytes)
    tensor_data = b'\x05' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q8_1', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q8_1')
    assert actual_data == tensor_data
    assert len(actual_data) == 40


# ============================================================================
# Test K-Quantized Types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
# ============================================================================

def test_tensor_data_q2_k_type():
    """Test reading tensor data for Q2_K quantized type."""
    tensor_type = GGMLType.Q2_K
    dims = [256]  # 256 elements = 1 block
    
    # Q2_K: 82 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q2_K data (1 block * 82 bytes = 82 bytes)
    tensor_data = b'\x06' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q2_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q2_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 82


def test_tensor_data_q3_k_type():
    """Test reading tensor data for Q3_K quantized type."""
    tensor_type = GGMLType.Q3_K
    dims = [512]  # 512 elements = 2 blocks of 256
    
    # Q3_K: 110 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q3_K data (2 blocks * 110 bytes = 220 bytes)
    tensor_data = b'\x07' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q3_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q3_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 220


def test_tensor_data_q4_k_type():
    """Test reading tensor data for Q4_K quantized type."""
    tensor_type = GGMLType.Q4_K
    dims = [256]  # 256 elements = 1 block
    
    # Q4_K: 144 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q4_K data (1 block * 144 bytes = 144 bytes)
    tensor_data = b'\x08' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q4_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q4_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 144


def test_tensor_data_q5_k_type():
    """Test reading tensor data for Q5_K quantized type."""
    tensor_type = GGMLType.Q5_K
    dims = [512]  # 512 elements = 2 blocks of 256
    
    # Q5_K: 176 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q5_K data (2 blocks * 176 bytes = 352 bytes)
    tensor_data = b'\x09' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q5_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q5_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 352


def test_tensor_data_q6_k_type():
    """Test reading tensor data for Q6_K quantized type."""
    tensor_type = GGMLType.Q6_K
    dims = [256]  # 256 elements = 1 block
    
    # Q6_K: 210 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q6_K data (1 block * 210 bytes = 210 bytes)
    tensor_data = b'\x0a' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q6_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q6_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 210


def test_tensor_data_q8_k_type():
    """Test reading tensor data for Q8_K quantized type."""
    tensor_type = GGMLType.Q8_K
    dims = [512]  # 512 elements = 2 blocks of 256
    
    # Q8_K: 292 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = dims[0] // block_size
    data_size = type_size * num_blocks
    
    # Create some Q8_K data (2 blocks * 292 bytes = 584 bytes)
    tensor_data = b'\x0b' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q8_k', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q8_k')
    assert actual_data == tensor_data
    assert len(actual_data) == 584


# ============================================================================
# Test Tensors with Different Dimension Counts
# ============================================================================

def test_tensor_data_1d_tensor():
    """Test reading tensor data for 1-dimensional tensor."""
    tensor_type = GGMLType.F32
    dims = [8]  # 1D: 8 elements
    
    # F32: 4 bytes per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create F32 data
    tensor_data = struct.pack('<8f', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_1d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_1d')
    assert actual_data == tensor_data
    assert len(actual_data) == 32  # 8 elements * 4 bytes


def test_tensor_data_2d_tensor():
    """Test reading tensor data for 2-dimensional tensor."""
    tensor_type = GGMLType.F32
    dims = [4, 3]  # 2D: 4x3 = 12 elements
    
    # F32: 4 bytes per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size) * dims[1]
    
    # Create F32 data for 12 elements
    tensor_data = struct.pack('<12f', *range(12))
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_2d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_2d')
    assert actual_data == tensor_data
    assert len(actual_data) == 48  # 12 elements * 4 bytes


def test_tensor_data_3d_tensor():
    """Test reading tensor data for 3-dimensional tensor."""
    tensor_type = GGMLType.I8
    dims = [4, 3, 2]  # 3D: 4x3x2 = 24 elements
    
    # I8: 1 byte per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size) * dims[1] * dims[2]
    
    # Create I8 data for 24 elements
    tensor_data = struct.pack('<24b', *range(-12, 12))
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_3d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_3d')
    assert actual_data == tensor_data
    assert len(actual_data) == 24  # 24 elements * 1 byte


def test_tensor_data_4d_tensor():
    """Test reading tensor data for 4-dimensional tensor."""
    tensor_type = GGMLType.I16
    dims = [2, 2, 2, 2]  # 4D: 2x2x2x2 = 16 elements
    
    # I16: 2 bytes per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size) * dims[1] * dims[2] * dims[3]
    
    # Create I16 data for 16 elements
    tensor_data = struct.pack('<16h', *range(-8, 8))
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_4d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_4d')
    assert actual_data == tensor_data
    assert len(actual_data) == 32  # 16 elements * 2 bytes


def test_tensor_data_quantized_2d_tensor():
    """Test reading tensor data for 2D quantized tensor."""
    tensor_type = GGMLType.Q4_0
    dims = [64, 4]  # 2D: 64x4 = 256 elements
    
    # Q4_0: 18 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = (dims[0] // block_size) * dims[1]
    data_size = type_size * num_blocks
    
    # Create Q4_0 data (8 blocks * 18 bytes = 144 bytes)
    tensor_data = b'\xaa' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q4_0_2d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q4_0_2d')
    assert actual_data == tensor_data
    assert len(actual_data) == 144


def test_tensor_data_quantized_3d_tensor():
    """Test reading tensor data for 3D quantized tensor."""
    tensor_type = GGMLType.Q8_0
    dims = [32, 2, 3]  # 3D: 32x2x3 = 192 elements
    
    # Q8_0: 34 bytes per 32-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = (dims[0] // block_size) * dims[1] * dims[2]
    data_size = type_size * num_blocks
    
    # Create Q8_0 data (6 blocks * 34 bytes = 204 bytes)
    tensor_data = b'\xbb' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q8_0_3d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q8_0_3d')
    assert actual_data == tensor_data
    assert len(actual_data) == 204


def test_tensor_data_k_quantized_4d_tensor():
    """Test reading tensor data for 4D K-quantized tensor."""
    tensor_type = GGMLType.Q4_K
    dims = [256, 2, 2, 1]  # 4D: 256x2x2x1 = 1024 elements
    
    # Q4_K: 144 bytes per 256-element block
    type_size, block_size = TYPE_SIZES[tensor_type]
    num_blocks = (dims[0] // block_size) * dims[1] * dims[2] * dims[3]
    data_size = type_size * num_blocks
    
    # Create Q4_K data (4 blocks * 144 bytes = 576 bytes)
    tensor_data = b'\xcc' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_q4_k_4d', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_q4_k_4d')
    assert actual_data == tensor_data
    assert len(actual_data) == 576


# ============================================================================
# Edge Cases: Minimum and Maximum Dimensions
# ============================================================================

def test_tensor_data_minimum_size():
    """Test reading tensor data for minimum size tensor (1 element)."""
    tensor_type = GGMLType.F32
    dims = [1]  # 1 element
    
    # F32: 4 bytes per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create F32 data for 1 element
    tensor_data = struct.pack('<f', 42.0)
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_min', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_min')
    assert actual_data == tensor_data
    assert len(actual_data) == 4


def test_tensor_data_large_1d_tensor():
    """Test reading tensor data for large 1D tensor."""
    tensor_type = GGMLType.I8
    dims = [1024]  # 1024 elements
    
    # I8: 1 byte per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size)
    
    # Create I8 data for 1024 elements
    tensor_data = bytes(range(256)) * 4  # Repeat 0-255 four times
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_large', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_large')
    assert actual_data == tensor_data
    assert len(actual_data) == 1024


def test_tensor_data_asymmetric_dimensions():
    """Test reading tensor data for tensor with asymmetric dimensions."""
    tensor_type = GGMLType.F32
    dims = [8, 3, 5, 2]  # 4D: 8x3x5x2 = 240 elements
    
    # F32: 4 bytes per element
    type_size, block_size = TYPE_SIZES[tensor_type]
    data_size = type_size * (dims[0] // block_size) * dims[1] * dims[2] * dims[3]
    
    # Create F32 data for 240 elements
    tensor_data = b'\xdd' * data_size
    assert len(tensor_data) == data_size
    
    file_content = create_gguf_file_with_tensor('test_asym', tensor_type, dims, tensor_data)
    
    reader = GGUFReader("test.gguf")
    reader.file = io.BytesIO(file_content)
    reader._read_header()
    reader._read_metadata()
    reader._read_tensor_info()
    reader._calculate_alignment()
    
    actual_data = reader.get_tensor_data('test_asym')
    assert actual_data == tensor_data
    assert len(actual_data) == 960  # 240 elements * 4 bytes
