#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>

__attribute__((alwaysinline))
uint32_t vgemv3d_vv(uint32_t rs1, uint32_t rs2) {
  uint32_t rd = 0;
  asm volatile(
  ".insn r 0x7B, 0b111, 0x31, %0, %1, %2"  // opcode=0x2B (0101011), funct7=0x31 (0110001)
  : "=r"(rd) : "r"(rs1), "r"(rs2));
  return rd;
}

// Input data: 4×4 matrix (16 elements, row-major) + 4D vector
// Memory layout:
// - matrix[16] at offset 0 (64 bytes)
// - vector[4] at offset 64 (16 bytes)
// Using signed integers (i32) for this operation
volatile int32_t input_data[20] __attribute__((aligned(128))) = {
  // 4×4 transformation matrix (row-major)
  // Row 0: [1, 0, 0, 10]
  // Row 1: [0, 1, 0, 20]
  // Row 2: [0, 0, 1, 30]
  // Row 3: [0, 0, 0,  1]
  // This is a simple translation matrix
  1, 0, 0, 10,
  0, 1, 0, 20,
  0, 0, 1, 30,
  0, 0, 0,  1,

  // 4D homogeneous vector (x, y, z, w)
  5, 10, 15, 1
};

// Output buffer for result vector (4 elements)
volatile int32_t output_data[4] __attribute__((aligned(128))) = {0};

int main(void) {
  printf("VGEMV3D Test - 4×4 Matrix-Vector Multiply (Homogeneous Coordinates)\n");
  printf("Input data address: 0x%lx\n", (unsigned long)input_data);
  printf("Output data address: 0x%lx\n", (unsigned long)output_data);

  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Running on: %s\n\n", march);

  // Call custom instruction
  volatile uint32_t result = 0;
  for (int i = 0; i < 10; i++) {
    result = vgemv3d_vv((uint32_t)(unsigned long)input_data,
               (uint32_t)(unsigned long)output_data);
  }

  printf("Returned rd value: %u\n", result);

  printf("\nInput matrix (4×4):\n");
  for (int i = 0; i < 4; i++) {
    printf("  [%4d %4d %4d %4d]\n",
           input_data[i*4+0], input_data[i*4+1],
           input_data[i*4+2], input_data[i*4+3]);
  }

  printf("\nInput vector:\n");
  printf("  [%4d %4d %4d %4d]ᵀ\n",
         input_data[16], input_data[17], input_data[18], input_data[19]);

  printf("\nComputed result vector:\n");
  printf("  [%4d %4d %4d %4d]ᵀ\n",
         output_data[0], output_data[1], output_data[2], output_data[3]);

  // Verify results manually
  printf("\nVerification (manual calculation):\n");
  int32_t expected[4] = {0};
  for (int i = 0; i < 4; i++) {
    expected[i] = 0;
    for (int j = 0; j < 4; j++) {
      expected[i] += input_data[i*4 + j] * input_data[16 + j];
    }
  }

  int all_correct = 1;
  for (int i = 0; i < 4; i++) {
    int match = (expected[i] == output_data[i]);
    printf("result[%d]: expected=%d, got=%d %s\n",
           i, expected[i], output_data[i],
           match ? "✓" : "✗");
    if (!match) all_correct = 0;
  }

  printf("\nInterpretation (translation matrix example):\n");
  printf("Input point: (%d, %d, %d)\n", input_data[16], input_data[17], input_data[18]);
  printf("Transformed point: (%d, %d, %d)\n", output_data[0], output_data[1], output_data[2]);
  printf("(Expected: input + translation offset = (15, 30, 45))\n");

  printf("\nOverall: %s\n", all_correct ? "PASS" : "FAIL");

  return 0;
}
