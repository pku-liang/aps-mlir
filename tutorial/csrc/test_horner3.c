#include <stdint.h>
#include <stdio.h>
#include "marchid.h"
#include <riscv-pk/encoding.h>
// ((c3x + c2)x + c1)x + c0
#pragma megg optimize
int32_t horner3(int32_t x) {
    int32_t rd_result = 0;
    int32_t vc0 = 1;
    int32_t vc1 = 2;
    int32_t vc2 = 3;
    int32_t vc3 = 4;
    int32_t r = (((vc3 * x) + vc2) * x + vc1) * x + vc0;
    return r;
}


// Test data for polynomial: 4x³ + 3x² + 2x + 1
volatile int32_t test_inputs[10] __attribute__((aligned(128))) = {
    0,   // 4*0³ + 3*0² + 2*0 + 1 = 1
    1,   // 4*1³ + 3*1² + 2*1 + 1 = 4 + 3 + 2 + 1 = 10
    2,   // 4*8 + 3*4 + 2*2 + 1 = 32 + 12 + 4 + 1 = 49
    3,   // 4*27 + 3*9 + 2*3 + 1 = 108 + 27 + 6 + 1 = 142
    -1,  // 4*(-1) + 3*1 + 2*(-1) + 1 = -4 + 3 - 2 + 1 = -2
    -2,  // 4*(-8) + 3*4 + 2*(-2) + 1 = -32 + 12 - 4 + 1 = -23
    5,
    10,
    -3,
    4
};

volatile int32_t expected_results[10] = {
    1, 10, 49, 142, -2, -23, 586, 4321, -80, 313
};

int main(void) {
    // printf("HORNER3 Test - Cubic Polynomial (4x³ + 3x² + 2x + 1)\n");

    uint64_t marchid = read_csr(marchid);
    const char *march = get_march(marchid);
    // printf("Running on: %s\n\n", march);

    int passed = 0;
    int total = 10;

    for (int i = 0; i < total; i++) {
        asm volatile("fence" ::: "memory");
        int32_t result = horner3(test_inputs[i]);
        asm volatile("fence" ::: "memory");

        // printf("Test %d: horner3(%d) = %d (expected: %d) %s\n",
        //        i, test_inputs[i], result, expected_results[i],
        //        result == expected_results[i] ? "✓" : "✗");

        if (result == expected_results[i]) {
            passed++;
        }
    }

    // printf("\nPassed: %d/%d\n", passed, total);
    return (passed == total) ? 0 : 1;
}
