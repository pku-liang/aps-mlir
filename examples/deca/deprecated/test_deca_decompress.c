#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>
#include <string.h>

__attribute__((always_inline))
uint32_t deca_decompress_u1(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x05, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input: sparse values (64B) + bitmask (4B) + scale (2B)
volatile uint8_t input_data[128] __attribute__((aligned(128))) = {0};

// Output: 32 INT16 values (64 bytes)
volatile int16_t output_data[32] __attribute__((aligned(128))) = {0};

void print_i16_array(const char* name, const int16_t* data, size_t size) {
  printf("%s:\n", name);
  for (size_t i = 0; i < size; i++) {
    printf("%6d ", data[i]);
    if ((i + 1) % 8 == 0) printf("\n");
  }
}

int main(void) {
  printf("DECA Decompression Test\n");
  printf("========================================\n");

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Setup sparse values (first 16 INT8 values)
  int8_t sparse_values[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                               9, 10, 11, 12, 13, 14, 15, 16};
  memcpy((void*)input_data, sparse_values, 16);

  // Bitmask: first 16 bits set (0x0000FFFF)
  uint32_t bitmask = 0x0000FFFF;
  memcpy((void*)(input_data + 64), &bitmask, 4);

  // Scale: 1.0 in Q8.8 format (0x0100 = 256)
  int16_t scale = 0x0100;
  memcpy((void*)(input_data + 68), &scale, 2);

  printf("Bitmask: 0x%08X\n", bitmask);
  printf("Scale: 0x%04X (%.2f in Q8.8)\n\n", scale, scale / 256.0);

  // Call decompress
  printf("Decompressing...\n");
  deca_decompress_u1((uint32_t)(unsigned long)input_data,
                     (uint32_t)(unsigned long)output_data);

  print_i16_array("Decompressed output", output_data, 32);

  printf("\n========================================\n");
  printf("Test Complete!\n");
  printf("========================================\n");

  return 0;
}
