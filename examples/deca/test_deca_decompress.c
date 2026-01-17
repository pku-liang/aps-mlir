#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>
#include <string.h>

__attribute__((alwaysinline))
uint32_t deca_decompress_u1(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(
  ".insn r 0x2B, 0b111, 0x05, %0, %1, %2"  // opcode=0x2B (0101011), funct7=0x05 (0000101)
  : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data structure:
// - values[8]: sparse INT8 values packed in i64 format (64 bytes)
// - bitmask (4 bytes): u32 bitmask indicating which positions are non-zero
// - global_scale (2 bytes): i16 Q8.8 fixed-point scale factor
// Total: 64 + 4 + 2 = 70 bytes (plus padding)
volatile uint8_t input_data[128] __attribute__((aligned(128))) = {0};

// Output buffer: 32 INT16 values (64 bytes)
volatile int16_t output_data[32] __attribute__((aligned(128))) = {0};

// Helper: Set a bit in the bitmask
void set_bit(uint32_t* bitmask, int pos) {
  *bitmask |= (1U << pos);
}

// Helper: Count bits set in bitmask
int count_bits(uint32_t bitmask) {
  int count = 0;
  for (int i = 0; i < 32; i++) {
    if (bitmask & (1U << i)) count++;
  }
  return count;
}

// Helper: Print hex array
void print_hex_array(const char* name, const uint8_t* data, size_t size) {
  printf("%s: ", name);
  for (size_t i = 0; i < size; i++) {
    printf("%02X ", data[i]);
    if ((i + 1) % 16 == 0 && i + 1 < size) printf("\n           ");
  }
  printf("\n");
}

// Helper: Print i16 array
void print_i16_array(const char* name, const int16_t* data, size_t size) {
  printf("%s:\n", name);
  for (size_t i = 0; i < size; i++) {
    printf("%6d ", data[i]);
    if ((i + 1) % 8 == 0) printf("\n");
  }
}

// Helper: Manually verify decompression
void verify_decompression(uint32_t bitmask, const int8_t* sparse_vals,
                          int16_t scale, const int16_t* output) {
  printf("\nManual verification:\n");
  int sparse_idx = 0;
  int errors = 0;

  for (int i = 0; i < 32; i++) {
    int16_t expected;
    if (bitmask & (1U << i)) {
      // Non-zero: dequantize sparse value
      int8_t val = sparse_vals[sparse_idx++];
      int32_t mul_result = (int32_t)val * (int32_t)scale;
      expected = (int16_t)(mul_result >> 8);
    } else {
      // Zero position
      expected = 0;
    }

    if (output[i] != expected) {
      printf("  [%2d] ERROR: expected=%6d, got=%6d\n", i, expected, output[i]);
      errors++;
    }
  }

  if (errors == 0) {
    printf("  ✓ All 32 values correct!\n");
  } else {
    printf("  ✗ Found %d errors!\n", errors);
  }
}

int main(void) {
  printf("DECA Decompression Test - Custom RISC-V Instruction\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // ========================================================================
  // Test Case 1: Simple pattern - first 16 positions set
  // ========================================================================
  printf("========================================\n");
  printf("TEST 1: First 16 positions non-zero\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  // Set sparse values (first 16 INT8 values)
  int8_t sparse_values1[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16};
  memcpy((void*)input_data, sparse_values1, 16);

  // Set bitmask at offset 64: first 16 bits set (0x0000FFFF)
  uint32_t bitmask1 = 0x0000FFFF;
  memcpy((void*)(input_data + 64), &bitmask1, 4);

  // Set global scale at offset 68: 1.0 in Q8.8 format (0x0100)
  int16_t scale1 = 0x0100;
  memcpy((void*)(input_data + 68), &scale1, 2);

  printf("Bitmask: 0x%08X (%d bits set)\n", bitmask1, count_bits(bitmask1));
  printf("Scale: 0x%04X (%.2f)\n", scale1, scale1 / 256.0);
  print_hex_array("Sparse values", input_data, 16);

  // Call custom instruction
  // Call custom instruction
  volatile uint32_t result1 = 0;
  for (int i = 0; i < 10; i++) {
    result1 = deca_decompress_u1((uint32_t)(unsigned long)input_data,
               (uint32_t)(unsigned long)output_data);
  }

  printf("\nReturned rd value: %u\n", result1);
  print_i16_array("Output", output_data, 32);
  verify_decompression(bitmask1, sparse_values1, scale1, output_data);

  // ========================================================================
  // Test Case 2: Alternating pattern
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 2: Alternating positions (0xAAAAAAAA)\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  // Set sparse values (16 values for alternating pattern)
  int8_t sparse_values2[16] = {10, 20, 30, 40, 50, 60, 70, 80,
                                90, 100, 110, 120, -10, -20, -30, -40};
  memcpy((void*)input_data, sparse_values2, 16);

  // Set bitmask: alternating pattern (0xAAAAAAAA)
  uint32_t bitmask2 = 0xAAAAAAAA;
  memcpy((void*)(input_data + 64), &bitmask2, 4);

  // Set global scale: 2.0 in Q8.8 format (0x0200)
  int16_t scale2 = 0x0200;
  memcpy((void*)(input_data + 68), &scale2, 2);

  printf("Bitmask: 0x%08X (%d bits set)\n", bitmask2, count_bits(bitmask2));
  printf("Scale: 0x%04X (%.2f)\n", scale2, scale2 / 256.0);
  print_hex_array("Sparse values", input_data, 16);

  // Call custom instruction
  uint32_t result2 = deca_decompress_u1((uint32_t)(unsigned long)input_data,
                                         (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result2);
  print_i16_array("Output", output_data, 32);
  verify_decompression(bitmask2, sparse_values2, scale2, output_data);

  // ========================================================================
  // Test Case 3: Sparse pattern (only a few bits set)
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 3: Sparse pattern (5 values)\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  // Set sparse values (only 5 non-zero)
  int8_t sparse_values3[5] = {100, -50, 75, -25, 127};
  memcpy((void*)input_data, sparse_values3, 5);

  // Set bitmask: positions 0, 5, 10, 20, 31 set
  uint32_t bitmask3 = 0;
  set_bit(&bitmask3, 0);
  set_bit(&bitmask3, 5);
  set_bit(&bitmask3, 10);
  set_bit(&bitmask3, 20);
  set_bit(&bitmask3, 31);
  memcpy((void*)(input_data + 64), &bitmask3, 4);

  // Set global scale: 0.5 in Q8.8 format (0x0080)
  int16_t scale3 = 0x0080;
  memcpy((void*)(input_data + 68), &scale3, 2);

  printf("Bitmask: 0x%08X (%d bits set)\n", bitmask3, count_bits(bitmask3));
  printf("Scale: 0x%04X (%.2f)\n", scale3, scale3 / 256.0);
  print_hex_array("Sparse values", input_data, 5);

  // Call custom instruction
  uint32_t result3 = deca_decompress_u1((uint32_t)(unsigned long)input_data,
                                         (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result3);
  print_i16_array("Output", output_data, 32);
  verify_decompression(bitmask3, sparse_values3, scale3, output_data);

  // ========================================================================
  // Test Case 4: All zeros
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 4: All zeros (empty bitmask)\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0xFF, 64);  // Fill with non-zero to verify clearing

  uint32_t bitmask4 = 0;
  memcpy((void*)(input_data + 64), &bitmask4, 4);

  int16_t scale4 = 0x0100;
  memcpy((void*)(input_data + 68), &scale4, 2);

  printf("Bitmask: 0x%08X (%d bits set)\n", bitmask4, count_bits(bitmask4));

  // Call custom instruction
  uint32_t result4 = deca_decompress_u1((uint32_t)(unsigned long)input_data,
                                         (uint32_t)(unsigned long)output_data);

  printf("Returned rd value: %u\n", result4);

  int all_zero = 1;
  for (int i = 0; i < 32; i++) {
    if (output_data[i] != 0) {
      all_zero = 0;
      break;
    }
  }
  printf("All outputs zero: %s\n", all_zero ? "✓ PASS" : "✗ FAIL");

  printf("\n========================================\n");
  printf("All tests completed!\n");
  printf("========================================\n");

  return 0;
}
