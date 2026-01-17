#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>

#pragma megg optimize
uint32_t avg_r(uint32_t a, uint32_t b) {
    uint32_t avg_r = (a + b) >> 1;
    return avg_r;
}

// Test data for average with rounding
volatile uint32_t test_inputs[10][2] __attribute__((aligned(128))) = {
    {10, 20},      // (10 + 20) / 2 = 15
    {0, 100},      // (0 + 100) / 2 = 50
    {50, 50},      // (50 + 50) / 2 = 50
    {100, 0},      // (100 + 0) / 2 = 50
    {1, 1},        // (1 + 1) / 2 = 1
    {7, 9},        // (7 + 9) / 2 = 8
    {255, 255},    // (255 + 255) / 2 = 255
    {1000, 2000},  // (1000 + 2000) / 2 = 1500
    {13, 17},      // (13 + 17) / 2 = 15
    {42, 84}       // (42 + 84) / 2 = 63
};

volatile uint32_t expected_results[10] = {
    15, 50, 50, 50, 1, 8, 255, 1500, 15, 63
};

int main(void) {
    // printf("AVG_R Test - Bitwise Average with Rounding\n");

    // uint64_t marchid = read_csr(marchid);
    // const char *march = get_march(marchid);
    // printf("Running on: %s\n\n", march);

    int passed = 0;
    int total = 10;

    for (int i = 0; i < total; i++) {
        asm volatile("fence" ::: "memory");
        uint32_t result = avg_r(test_inputs[i][0], test_inputs[i][1]);
        asm volatile("fence" ::: "memory");

        // printf("Test %d: avg_r(%u, %u) = %u (expected: %u) %s\n",
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
