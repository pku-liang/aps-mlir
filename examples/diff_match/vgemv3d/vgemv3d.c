// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source
#include <stdint.h>



uint8_t vgemv3d_vv(int32_t *rs1, int32_t *rs2) {
    int32_t acc = 0;

    uint8_t rd_result = 0;
    int32_t * addr = rs1;
    int32_t * out_addr = rs2;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    // version 1 - unrolled outer loop by factor of 2
    for (i = 0; i < 2; ++i) {
        acc = 0;
        uint32_t i_mul_2 = (i * 2);
        uint32_t j;
        for (j = 0; j < 4; ++j) {
            acc = (acc + (rs1[((i_mul_2 * 4) + j)] * rs1[16 + j]));
            uint32_t j_ = (j + 1);
        }
        rs2[i_mul_2] = acc;
        acc = 0;
        for (j = 0; j < 4; ++j) {
            acc = (acc + (rs1[(((i_mul_2 + 1) * 4) + j)] * rs1[16 + j]));
            uint32_t j_ = (j + 1);
        }
        rs2[i_mul_2 + 1] = acc;
    }
    // version2 - tile factor 2
    // for (uint32_t i = 0; i < 4; ++i) {
    //     acc = 0;
    //     // j 方向 tile = 2
    //     for (uint32_t jb = 0; jb < 4; jb += 2) {
    //         uint32_t j_end = (jb + 2 < 4) ? (jb + 2) : 4;
    //         for (uint32_t j = jb; j < j_end; ++j) {
    //             acc += rs1[(i * 4) + j] * rs1[16 + j];
    //         }
    //     }
    //     rs2[i] = acc;
    // }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}


