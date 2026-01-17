// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

int32_t lerp(int32_t rs1_value, int32_t rs2_value) {
    int32_t rd_result = 0;
    int32_t va = rs1_value;
    int32_t vb = rs2_value;
    int32_t vt = ((va + vb) >> 2);
    int32_t diff = (vb - va);
    int32_t scaled = (diff * vt);
    int32_t r = (va + scaled);
    rd_result = r;
    return rd_result;
}
