#!/usr/bin/env python3
"""
Convert a llama2.c quantized model binary to a C header file for bare metal use.

Usage: python3 bin2header.py <input.bin> <output.h>
Example: python3 bin2header.py stories15Mq.bin model_weights.h
"""

import struct
import sys
import os

ALIGNMENT = 128

def quantized_size(n_elements, group_size):
    """Size in bytes for quantized tensor: int8 values + float32 scales"""
    q_size = n_elements * 1  # int8
    s_size = (n_elements // group_size) * 4  # float32 scales
    return q_size + s_size

def check_alignment(name, offset, size):
    """Check if offset is 128-byte aligned, panic if not"""
    if offset % ALIGNMENT != 0:
        print(f"PANIC: {name} at offset {offset} is NOT {ALIGNMENT}-byte aligned!")
        print(f"       offset % {ALIGNMENT} = {offset % ALIGNMENT}")
        sys.exit(1)
    return offset + size

def bin_to_header(input_path, output_path):
    with open(input_path, 'rb') as f:
        # Read header (256 bytes)
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x616b3432:
            print(f"Error: Bad magic number 0x{magic:08x}, expected 0x616b3432")
            sys.exit(1)

        version = struct.unpack('<i', f.read(4))[0]
        if version != 2:
            print(f"Error: Bad version {version}, expected 2")
            sys.exit(1)

        # Read Config
        dim = struct.unpack('<i', f.read(4))[0]
        hidden_dim = struct.unpack('<i', f.read(4))[0]
        n_layers = struct.unpack('<i', f.read(4))[0]
        n_heads = struct.unpack('<i', f.read(4))[0]
        n_kv_heads = struct.unpack('<i', f.read(4))[0]
        vocab_size = struct.unpack('<i', f.read(4))[0]
        seq_len = struct.unpack('<i', f.read(4))[0]

        # Read flags
        shared_classifier = struct.unpack('<B', f.read(1))[0]
        group_size = struct.unpack('<i', f.read(4))[0]

        # Skip to end of 256-byte header
        f.seek(256)

        # Read the rest as weights
        weights_data = f.read()

    weights_size = len(weights_data)
    head_size = dim // n_heads

    print(f"Model: {input_path}")
    print(f"Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}, seq_len={seq_len}")
    print(f"shared_classifier={shared_classifier}, group_size={group_size}")
    print(f"Weights size: {weights_size} bytes ({weights_size / 1024 / 1024:.2f} MB)")

    # Check alignment of all weight sections
    print(f"\nChecking {ALIGNMENT}-byte alignment of weight sections...")
    offset = 0

    # FP32 rmsnorm weights
    offset = check_alignment("rms_att_weight", offset, n_layers * dim * 4)
    offset = check_alignment("rms_ffn_weight", offset, n_layers * dim * 4)
    offset = check_alignment("rms_final_weight", offset, dim * 4)

    # Quantized weights
    offset = check_alignment("q_tokens", offset, quantized_size(vocab_size * dim, group_size))
    offset = check_alignment("wq", offset, quantized_size(n_layers * dim * (n_heads * head_size), group_size))
    offset = check_alignment("wk", offset, quantized_size(n_layers * dim * (n_kv_heads * head_size), group_size))
    offset = check_alignment("wv", offset, quantized_size(n_layers * dim * (n_kv_heads * head_size), group_size))
    offset = check_alignment("wo", offset, quantized_size(n_layers * (n_heads * head_size) * dim, group_size))
    offset = check_alignment("w1", offset, quantized_size(n_layers * dim * hidden_dim, group_size))
    offset = check_alignment("w2", offset, quantized_size(n_layers * hidden_dim * dim, group_size))
    offset = check_alignment("w3", offset, quantized_size(n_layers * dim * hidden_dim, group_size))

    if not shared_classifier:
        offset = check_alignment("wcls", offset, quantized_size(dim * vocab_size, group_size))

    if offset != weights_size:
        print(f"PANIC: Calculated size {offset} != actual size {weights_size}")
        sys.exit(1)

    print(f"All weight sections are {ALIGNMENT}-byte aligned. ✓")

    # Generate header file
    with open(output_path, 'w') as out:
        out.write("/* Auto-generated model weights header for bare metal llama2.c */\n")
        out.write(f"/* Source: {os.path.basename(input_path)} */\n\n")
        out.write("#ifndef MODEL_WEIGHTS_H\n")
        out.write("#define MODEL_WEIGHTS_H\n\n")
        out.write("#include <stdint.h>\n\n")

        # Config struct instance
        out.write("/* Model configuration */\n")
        out.write("static const Config MODEL_CONFIG = {\n")
        out.write(f"    .dim = {dim},\n")
        out.write(f"    .hidden_dim = {hidden_dim},\n")
        out.write(f"    .n_layers = {n_layers},\n")
        out.write(f"    .n_heads = {n_heads},\n")
        out.write(f"    .n_kv_heads = {n_kv_heads},\n")
        out.write(f"    .vocab_size = {vocab_size},\n")
        out.write(f"    .seq_len = {seq_len}\n")
        out.write("};\n\n")

        # Flags
        out.write(f"static const uint8_t MODEL_SHARED_CLASSIFIER = {shared_classifier};\n")
        out.write(f"static const int MODEL_GROUP_SIZE = {group_size};\n\n")

        # Weights array
        out.write(f"/* Model weights ({weights_size} bytes) */\n")
        out.write(f"static const unsigned char MODEL_WEIGHTS[{weights_size}] __attribute__((aligned(128))) = {{\n")

        # Write in chunks for readability
        BYTES_PER_LINE = 16
        for i in range(0, weights_size, BYTES_PER_LINE):
            chunk = weights_data[i:i+BYTES_PER_LINE]
            hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
            if i + BYTES_PER_LINE < weights_size:
                out.write(f"    {hex_str},\n")
            else:
                out.write(f"    {hex_str}\n")

        out.write("};\n\n")
        out.write("#endif /* MODEL_WEIGHTS_H */\n")

    print(f"Generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.h>")
        print(f"Example: {sys.argv[0]} stories15Mq.bin model_weights.h")
        sys.exit(1)

    bin_to_header(sys.argv[1], sys.argv[2])
