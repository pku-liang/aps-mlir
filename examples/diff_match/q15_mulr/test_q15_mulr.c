
#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
int32_t q15_mulr(int32_t x, int32_t y) {
    int32_t product = ((int64_t)x * y + 16384) >> 15; // Rounding by adding 0.5 in Q15
    return product;
}

// Test data for Q15 fixed-point multiplication
// Q15 format: 1 sign bit + 15 fractional bits (value / 32768)
volatile int32_t test_inputs[10][2] __attribute__((aligned(128))) = {
    {16384, 16384},  // 0.5 * 0.5 = 0.25 => 8192
    {32767, 32767},  // 0.9999 * 0.9999 ≈ 0.9999 => 32766
    {0, 32767},      // 0 * 0.9999 = 0
    {16384, 32767},  // 0.5 * 0.9999 ≈ 0.5 => 16383
    {8192, 8192},    // 0.25 * 0.25 = 0.0625 => 2048
    {-16384, 16384}, // -0.5 * 0.5 = -0.25 => -8192
    {32767, 16384},  // 0.9999 * 0.5 ≈ 0.5 => 16383
    {-32768, 16384}, // -1.0 * 0.5 = -0.5 => -16384
    {4096, 4096},    // 0.125 * 0.125 = 0.015625 => 512
    {24576, 16384}   // 0.75 * 0.5 = 0.375 => 12288
};

volatile int32_t expected_results[10] = {
    8192, 32766, 0, 16383, 2048, -8192, 16383, -16384, 512, 12288
};

int main(void) {
    // printf("Q15_MULR Test - Q15 Fixed-Point Multiplication with Rounding\n");

    uint64_t marchid = read_csr(marchid);
    const char *march = get_march(marchid);
    // printf("Running on: %s\n\n", march);

    int passed = 0;
    int total = 10;

    for (int i = 0; i < total; i++) {
        asm volatile("fence" ::: "memory");
        int32_t result = q15_mulr(test_inputs[i][0], test_inputs[i][1]);
        asm volatile("fence" ::: "memory");

        // printf("Test %d: q15_mulr(%d, %d) = %d (expected: %d) %s\n",
        //        i, test_inputs[i][0], test_inputs[i][1], result,
        //        expected_results[i],
        //        result == expected_results[i] ? "✓" : "✗");

        if (result == expected_results[i]) {
            passed++;
        }
    }

    // printf("\nPassed: %d/%d\n", passed, total);
    return (passed == total) ? 0 : 1;
}
