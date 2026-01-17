// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint32_t avg_r(uint32_t rs1_value, uint32_t rs2_value) {
    uint32_t rd_result = 0;
    uint32_t x = rs1_value;
    uint32_t y = rs2_value;
    uint32_t and_xy = (x & y);
    uint32_t xor_xy = (x ^ y);
    uint32_t shift = (xor_xy >> 1);
    uint32_t r = (and_xy + shift);
    rd_result = r;
    return rd_result;
}
