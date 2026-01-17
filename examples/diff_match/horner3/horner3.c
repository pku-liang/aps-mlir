// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

int32_t horner3(int32_t rs1_value, uint32_t rs2_value) {
    int32_t rd_result = 0;
    int32_t vx = rs1_value;
    int32_t vc0 = 1;
    int32_t vc1 = 2;
    int32_t vc2 = 3;
    int32_t vc3 = 4;
    int32_t x2 = (vx * vx);
    int32_t x3 = (x2 * vx);
    int32_t t1 = (vc1 * vx);
    int32_t t2 = (vc2 * x2);
    int32_t t3 = (vc3 * x3);
    int32_t s1 = (vc0 + t1);
    int32_t s2 = (s1 + t2);
    int32_t r = (s2 + t3);
    rd_result = r;
    return rd_result;
}
