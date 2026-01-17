#!/usr/bin/env python3
"""
Convert a llama2.c tokenizer binary to a C header file for bare metal use.

Usage: python3 tokenizer2header.py <input.bin> <output.h>
Example: python3 tokenizer2header.py tokenizer.bin tokenizer_data.h
"""

import struct
import sys
import os

def tokenizer_to_header(input_path, output_path):
    with open(input_path, 'rb') as f:
        # Read and reformat data to add null terminators after each string
        max_token_length = struct.unpack('<i', f.read(4))[0]

        # Build new data with null terminators
        new_data = bytearray(struct.pack('<i', max_token_length))

        vocab_count = 0
        while True:
            score_bytes = f.read(4)
            if not score_bytes:
                break
            new_data.extend(score_bytes)  # score (float)

            len_bytes = f.read(4)
            length = struct.unpack('<i', len_bytes)[0]
            new_data.extend(len_bytes)  # length (int)

            string_bytes = f.read(length)
            new_data.extend(string_bytes)  # string
            new_data.append(0)  # null terminator
            vocab_count += 1

        data = bytes(new_data)

    size = len(data)
    print(f"Tokenizer: {input_path}")
    print(f"Size: {size} bytes ({size / 1024:.2f} KB) (with null terminators)")
    print(f"Max token length: {max_token_length}")
    print(f"Vocab count: {vocab_count}")

    # Generate header file
    with open(output_path, 'w') as out:
        out.write("/* Auto-generated tokenizer data header for bare metal llama2.c */\n")
        out.write(f"/* Source: {os.path.basename(input_path)} */\n\n")
        out.write("#ifndef TOKENIZER_DATA_H\n")
        out.write("#define TOKENIZER_DATA_H\n\n")
        out.write("#include <stdint.h>\n\n")

        out.write(f"#define TOKENIZER_DATA_SIZE {size}\n\n")

        # Tokenizer data array (aligned to 4 bytes for RISC-V)
        out.write(f"/* Tokenizer data ({size} bytes) */\n")
        out.write(f"static const unsigned char TOKENIZER_DATA[{size}] __attribute__((aligned(128))) = {{\n")

        # Write in chunks for readability
        BYTES_PER_LINE = 16
        for i in range(0, size, BYTES_PER_LINE):
            chunk = data[i:i+BYTES_PER_LINE]
            hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
            if i + BYTES_PER_LINE < size:
                out.write(f"    {hex_str},\n")
            else:
                out.write(f"    {hex_str}\n")

        out.write("};\n\n")
        out.write("#endif /* TOKENIZER_DATA_H */\n")

    print(f"Generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.h>")
        print(f"Example: {sys.argv[0]} tokenizer.bin tokenizer_data.h")
        sys.exit(1)

    tokenizer_to_header(sys.argv[1], sys.argv[2])
