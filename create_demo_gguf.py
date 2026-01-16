#!/usr/bin/env python3
"""
Create a demo GGUF file for testing the example_usage.py script.

This script creates a small, valid GGUF file with sample metadata and tensors
that can be used to demonstrate the GGUF Reader functionality.
"""

import struct


def create_demo_gguf_file(filepath: str):
    """
    Create a demo GGUF file with sample metadata and tensors.
    
    Args:
        filepath: Path where the GGUF file should be created
    """
    # Start with the header
    file_content = bytearray()
    
    # Header: magic, version, tensor_count, metadata_kv_count
    file_content += struct.pack('<I', 0x46554747)  # magic: "GGUF"
    file_content += struct.pack('<I', 3)           # version: 3
    file_content += struct.pack('<Q', 3)           # tensor_count: 3 tensors
    file_content += struct.pack('<Q', 8)           # metadata_kv_count: 8 metadata entries
    
    # Metadata entries
    metadata = [
        ('general.architecture', 8, 'llama'),  # STRING
        ('general.name', 8, 'Demo Model v1.0'),  # STRING
        ('general.alignment', 4, 32),  # UINT32
        ('llama.context_length', 4, 2048),  # UINT32
        ('llama.embedding_length', 4, 4096),  # UINT32
        ('llama.block_count', 4, 32),  # UINT32
        ('llama.attention.head_count', 4, 32),  # UINT32
        ('llama.feed_forward_length', 4, 11008),  # UINT32
    ]
    
    for key, value_type, value in metadata:
        # Write key (length-prefixed string)
        key_bytes = key.encode('utf-8')
        file_content += struct.pack('<Q', len(key_bytes))
        file_content += key_bytes
        
        # Write value type
        file_content += struct.pack('<I', value_type)
        
        # Write value
        if value_type == 8:  # STRING
            value_bytes = value.encode('utf-8')
            file_content += struct.pack('<Q', len(value_bytes))
            file_content += value_bytes
        elif value_type == 4:  # UINT32
            file_content += struct.pack('<I', value)
    
    # Tensor information
    tensors = [
        ('token_embd.weight', [4096, 32000], 0),  # F32 type
        ('output.weight', [32000, 4096], 1),      # F16 type
        ('norm.weight', [4096], 0),               # F32 type
    ]
    
    tensor_data_list = []
    current_offset = 0
    
    for tensor_name, dims, tensor_type in tensors:
        # Write tensor name (length-prefixed string)
        name_bytes = tensor_name.encode('utf-8')
        file_content += struct.pack('<Q', len(name_bytes))
        file_content += name_bytes
        
        # Write n_dims
        file_content += struct.pack('<I', len(dims))
        
        # Write dimension sizes
        for dim in dims:
            file_content += struct.pack('<Q', dim)
        
        # Write tensor type
        file_content += struct.pack('<I', tensor_type)
        
        # Write offset
        file_content += struct.pack('<Q', current_offset)
        
        # Calculate tensor data size for offset calculation
        if tensor_type == 0:  # F32
            type_size, block_size = 4, 1
        elif tensor_type == 1:  # F16
            type_size, block_size = 2, 1
        else:
            type_size, block_size = 4, 1
        
        # Calculate size
        total_elements = 1
        for dim in dims:
            total_elements *= dim
        data_size = type_size * (total_elements // block_size)
        
        # Create dummy tensor data (just zeros for demo)
        tensor_data = b'\x00' * data_size
        tensor_data_list.append(tensor_data)
        
        # Update offset for next tensor
        current_offset += data_size
    
    # Calculate alignment padding
    alignment = 32
    current_pos = len(file_content)
    padding = (alignment - (current_pos % alignment)) % alignment
    file_content += b'\x00' * padding
    
    # Append tensor data
    for tensor_data in tensor_data_list:
        file_content += tensor_data
    
    # Write to file
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    print(f"Created demo GGUF file: {filepath}")
    print(f"  File size: {len(file_content):,} bytes")
    print(f"  Metadata entries: {len(metadata)}")
    print(f"  Tensors: {len(tensors)}")


if __name__ == '__main__':
    create_demo_gguf_file('demo_model.gguf')
