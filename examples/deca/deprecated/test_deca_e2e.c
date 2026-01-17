#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>
#include <string.h>

// ============================================================================
// Custom RISC-V Instructions
// ============================================================================

__attribute__((always_inline))
uint32_t deca_decompress_u1(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x0B, 0b111, 0x05, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

__attribute__((always_inline))
uint32_t gemm_4x4_custom(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x38, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// ============================================================================
// Memory Buffers
// ============================================================================

volatile uint8_t compressed_weights[128] __attribute__((aligned(128))) = {0};
volatile int16_t decompressed_weights[32] __attribute__((aligned(128))) = {0};
volatile int32_t gemm_input[32] __attribute__((aligned(128))) = {0};
volatile int32_t matrix_c[16] __attribute__((aligned(128))) = {0};

// ============================================================================
// Helper Functions
// ============================================================================

void print_matrix(const char* name, const int32_t* mat) {
  printf("%s:\n", name);
  for (int i = 0; i < 4; i++) {
    printf("  ");
    for (int j = 0; j < 4; j++) {
      printf("%8d ", mat[i * 4 + j]);
    }
    printf("\n");
  }
}


// ============================================================================
// Main
// ============================================================================

int main(void) {
  printf("DECA E2E Test: Decompress → GEMM\n");
  printf("========================================\n");

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Setup compressed weights (16 values for 4×4 matrix B)
  int8_t sparse_values[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  memcpy((void*)compressed_weights, sparse_values, 16);

  // Bitmask: first 16 bits set (0x0000FFFF)
  uint32_t bitmask = 0x0000FFFF;
  memcpy((void*)(compressed_weights + 64), &bitmask, 4);

  // Scale: 1.0 in Q8.8 format (256 = 0x0100)
  int16_t scale = 256;
  memcpy((void*)(compressed_weights + 68), &scale, 2);

  // Step 1: Decompress
  printf("Step 1: DECA Decompress\n");
  deca_decompress_u1((uint32_t)(unsigned long)compressed_weights,
                     (uint32_t)(unsigned long)decompressed_weights);
  printf("  Decompressed 32 INT16 weights (Q8.8 format)\n");

  // Step 2: Convert first 16 weights to i32 and setup matrices
  printf("\nStep 2: Prepare GEMM input\n");

  // Matrix A: identity matrix
  int32_t matrix_a[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };

  // Matrix B: from decompressed weights (Q8.8 >> 8 = integer part)
  int32_t matrix_b[16];
  for (int i = 0; i < 16; i++) {
    matrix_b[i] = (int32_t)decompressed_weights[i] >> 8;
  }

  // Combine A and B for GEMM input
  memcpy((void*)gemm_input, matrix_a, 64);
  memcpy((void*)(gemm_input + 16), matrix_b, 64);

  print_matrix("  Matrix A (Identity)", matrix_a);
  print_matrix("  Matrix B (from weights)", matrix_b);

  // Step 3: GEMM
  printf("\nStep 3: GEMM 4×4 (C = A × B)\n");
  gemm_4x4_custom((uint32_t)(unsigned long)gemm_input,
                  (uint32_t)(unsigned long)matrix_c);

  print_matrix("  Matrix C (Result)", (int32_t*)matrix_c);

  printf("\n========================================\n");
  printf("E2E Test Complete!\n");
  printf("========================================\n");

  return 0;
}
