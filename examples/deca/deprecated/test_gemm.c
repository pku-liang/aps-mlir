#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>
#include <string.h>

__attribute__((always_inline))
uint32_t gemm_4x4_custom(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(".insn r 0x2B, 0b111, 0x38, %0, %1, %2"
               : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input: matrix_a[16] + matrix_b[16] (128 bytes total)
volatile int32_t input_data[32] __attribute__((aligned(128))) = {0};

// Output: matrix_c[16] (64 bytes)
volatile int32_t output_data[16] __attribute__((aligned(128))) = {0};

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

int main(void) {
  printf("GEMM 4×4 Test\n");
  printf("========================================\n");

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Setup matrices
  int32_t matrix_a[16] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  int32_t matrix_b[16] = {
    16, 15, 14, 13,
    12, 11, 10, 9,
    8, 7, 6, 5,
    4, 3, 2, 1
  };

  memcpy((void*)input_data, matrix_a, 64);
  memcpy((void*)(input_data + 16), matrix_b, 64);

  print_matrix("Matrix A", matrix_a);
  print_matrix("Matrix B", matrix_b);

  // Call GEMM
  printf("\nExecuting GEMM: C = A × B\n");
  gemm_4x4_custom((uint32_t)(unsigned long)input_data,
                  (uint32_t)(unsigned long)output_data);

  print_matrix("Matrix C (Result)", (int32_t*)output_data);

  printf("\n========================================\n");
  printf("Test Complete!\n");
  printf("========================================\n");

  return 0;
}
