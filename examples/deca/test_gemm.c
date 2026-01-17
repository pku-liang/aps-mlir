#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>
#include <string.h>

__attribute__((alwaysinline))
uint32_t gemm_4x4_custom(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(
  ".insn r 0x2B, 0b111, 0x38, %0, %1, %2"  // opcode=0x2B (0101011), funct7=0x38 (0111000)
  : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data structure:
// - matrix_a[16]: 4×4 matrix A (64 bytes, i32)
// - matrix_b[16]: 4×4 matrix B (64 bytes, i32)
// Total: 128 bytes for input (A followed by B)
volatile int16_t input_data[32] __attribute__((aligned(128))) = {0};

// Output buffer: matrix_c[16]: 4×4 matrix C (64 bytes, i32)
volatile int16_t output_data[16] __attribute__((aligned(128))) = {0};

// Helper: Print matrix (4×4 i32)
void print_matrix(const char* name, const int16_t* mat) {
  printf("%s:\n", name);
  for (int i = 0; i < 4; i++) {
    printf("  ");
    for (int j = 0; j < 4; j++) {
      printf("%8d ", mat[i * 4 + j]);
    }
    printf("\n");
  }
}

// Helper: Reference GEMM implementation
void gemm_reference(const int16_t* A, const int16_t* B, int16_t* C) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int16_t acc = 0;
      for (int k = 0; k < 4; k++) {
        acc += A[i * 4 + k] * B[k * 4 + j];
      }
      C[i * 4 + j] = acc;
    }
  }
}

// Helper: Verify result against reference
void verify_gemm(const int16_t* A, const int16_t* B, const int16_t* C_hw) {
  int16_t C_ref[16];
  gemm_reference(A, B, C_ref);

  printf("\nVerification:\n");
  int errors = 0;
  for (int i = 0; i < 16; i++) {
    if (C_hw[i] != C_ref[i]) {
      printf("  [%2d] ERROR: expected=%8d, got=%8d\n", i, C_ref[i], C_hw[i]);
      errors++;
    }
  }

  if (errors == 0) {
    printf("  ✓ All 16 values correct!\n");
  } else {
    printf("  ✗ Found %d errors!\n", errors);
  }
}

int main(void) {
  printf("GEMM 4×4 Test - Custom RISC-V Instruction\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // ========================================================================
  // Test Case 1: Identity matrices (I × I = I)
  // ========================================================================
  printf("========================================\n");
  printf("TEST 1: Identity × Identity\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  // Set up identity matrices
  int16_t identity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };
  memcpy((void*)input_data, identity, 64);        // Matrix A
  memcpy((void*)(input_data + 16), identity, 64); // Matrix B

  print_matrix("Matrix A (Identity)", input_data);
  print_matrix("Matrix B (Identity)", input_data + 16);

  // Call custom instruction
  volatile uint32_t result1 = 0;
  for (int i = 0; i < 10; i++) {
    result1 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                               (uint32_t)(unsigned long)output_data);
  }

  printf("\nReturned rd value: %u\n", result1);
  print_matrix("Matrix C (Result)", output_data);
  verify_gemm(input_data, input_data + 16, output_data);

  // ========================================================================
  // Test Case 2: Small integer matrices
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 2: Small integer matrices\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  int16_t matrix_a2[16] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  int16_t matrix_b2[16] = {
    16, 15, 14, 13,
    12, 11, 10, 9,
    8, 7, 6, 5,
    4, 3, 2, 1
  };

  memcpy((void*)input_data, matrix_a2, 64);
  memcpy((void*)(input_data + 16), matrix_b2, 64);

  print_matrix("Matrix A", input_data);
  print_matrix("Matrix B", input_data + 16);

  uint32_t result2 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                                      (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result2);
  print_matrix("Matrix C (Result)", output_data);
  verify_gemm(input_data, input_data + 16, output_data);

  // ========================================================================
  // Test Case 3: Matrix with zeros (sparse pattern)
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 3: Sparse matrices (many zeros)\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  int16_t matrix_a3[16] = {
    1, 0, 0, 0,
    0, 2, 0, 0,
    0, 0, 3, 0,
    0, 0, 0, 4
  };

  int16_t matrix_b3[16] = {
    5, 0, 0, 0,
    0, 6, 0, 0,
    0, 0, 7, 0,
    0, 0, 0, 8
  };

  memcpy((void*)input_data, matrix_a3, 64);
  memcpy((void*)(input_data + 16), matrix_b3, 64);

  print_matrix("Matrix A (Diagonal)", input_data);
  print_matrix("Matrix B (Diagonal)", input_data + 16);

  uint32_t result3 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                                      (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result3);
  print_matrix("Matrix C (Result)", output_data);
  verify_gemm(input_data, input_data + 16, output_data);

  // ========================================================================
  // Test Case 4: All zeros
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 4: Zero matrices\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0xFF, 64);  // Fill with non-zero to verify clearing

  printf("Matrix A: all zeros\n");
  printf("Matrix B: all zeros\n");

  uint32_t result4 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                                      (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result4);

  int all_zero = 1;
  for (int i = 0; i < 16; i++) {
    if (output_data[i] != 0) {
      all_zero = 0;
      break;
    }
  }
  printf("All outputs zero: %s\n", all_zero ? "✓ PASS" : "✗ FAIL");

  // ========================================================================
  // Test Case 5: Negative values
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 5: Matrices with negative values\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  int16_t matrix_a5[16] = {
    1, -1, 2, -2,
    3, -3, 4, -4,
    5, -5, 6, -6,
    7, -7, 8, -8
  };

  int16_t matrix_b5[16] = {
    -8, 7, -6, 5,
    -4, 3, -2, 1,
    8, -7, 6, -5,
    4, -3, 2, -1
  };

  memcpy((void*)input_data, matrix_a5, 64);
  memcpy((void*)(input_data + 16), matrix_b5, 64);

  print_matrix("Matrix A", input_data);
  print_matrix("Matrix B", input_data + 16);

  uint32_t result5 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                                      (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result5);
  print_matrix("Matrix C (Result)", output_data);
  verify_gemm(input_data, input_data + 16, output_data);

  // ========================================================================
  // Test Case 6: Large values (check for overflow handling)
  // ========================================================================
  printf("\n========================================\n");
  printf("TEST 6: Large values\n");
  printf("========================================\n");

  memset((void*)input_data, 0, 128);
  memset((void*)output_data, 0, 64);

  int16_t matrix_a6[16] = {
    100, 200, 300, 400,
    500, 600, 700, 800,
    900, 1000, 1100, 1200,
    1300, 1400, 1500, 1600
  };

  int16_t matrix_b6[16] = {
    1600, 1500, 1400, 1300,
    1200, 1100, 1000, 900,
    800, 700, 600, 500,
    400, 300, 200, 100
  };

  memcpy((void*)input_data, matrix_a6, 64);
  memcpy((void*)(input_data + 16), matrix_b6, 64);

  print_matrix("Matrix A", input_data);
  print_matrix("Matrix B", input_data + 16);

  uint32_t result6 = gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                                      (uint32_t)(unsigned long)output_data);

  printf("\nReturned rd value: %u\n", result6);
  print_matrix("Matrix C (Result)", output_data);
  verify_gemm(input_data, input_data + 16, output_data);

  printf("\n========================================\n");
  printf("All tests completed!\n");
  printf("========================================\n");

  return 0;
}
