// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

int32_t q15_mulr(int32_t rs1_value, int32_t rs2_value) {
    int32_t rd_result = 0;
    int32_t x = rs1_value;
    int32_t y = rs2_value;
    int64_t x64 = x;
    int64_t y64 = y;
    int64_t prod = (x64 * y64);
    int64_t round = 16384;
    int64_t prod_rounded = (prod + round);
    int64_t r64 = (prod_rounded >> 15);
    int32_t r = r64;
    rd_result = r;
    return rd_result;
}
