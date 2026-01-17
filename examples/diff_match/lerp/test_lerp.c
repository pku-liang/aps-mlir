#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>
// (1-t)a + t b, with t=(a + b) >> 2
#pragma megg optimize
int32_t lerp(int32_t a, int32_t b) {
    int32_t t = (a + b) / 4;
    return (1 - t) * a + t * b;
}


// Test data
volatile int32_t test_inputs[10][2] __attribute__((aligned(128))) = {
    {10, 20},   // a=10, b=20, t=5 => 10 + 5*(20-10) = 10 + 50 = 60
    {0, 100},   // a=0, b=100, t=5 => 0 + 5*100 = 500
    {50, 50},   // a=50, b=50, t=5 => 50 + 5*0 = 50
    {100, 0},   // a=100, b=0, t=5 => 100 + 5*(-100) = -400
    {-10, 10},  // a=-10, b=10, t=5 => -10 + 5*20 = 90
    {5, 15},
    {-20, 30},
    {25, 75},
    {0, 0},
    {42, 84}
};

volatile int32_t expected_results[10] = {
    60, 500, 50, -400, 90, 55, 250, 275, 0, 252
};

int main(void) {
    // printf("LERP Test - Linear Interpolation (t=5)\n");

    uint64_t marchid = read_csr(marchid);
    const char *march = get_march(marchid);
    // printf("Running on: %s\n\n", march);

    int passed = 0;
    int total = 10;

    for (int i = 0; i < total; i++) {
        asm volatile("fence" ::: "memory");
        int32_t result = lerp(test_inputs[i][0], test_inputs[i][1]);
        asm volatile("fence" ::: "memory");

        // printf("Test %d: lerp(%d, %d) = %d (expected: %d) %s\n",
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
