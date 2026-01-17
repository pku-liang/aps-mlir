// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint8_t deca_decompress_u1(int8_t *rs1, int16_t rs2_value, int16_t *rd) {
    int8_t dense_values[32];
    uint32_t vidx = 0;

    uint8_t rd_result = 0;
    int8_t * base_addr = rs1;
    int16_t global_scale = rs2_value;
    int16_t * out_addr = rd;
    // burst_read lowered via register-backed scratchpad
    // burst_read lowered via register-backed scratchpad
    uint32_t idx;
    for (idx = 0; ((idx + 1) < 32); ++idx) {
        uint32_t byte_idx = (idx / 8);
        uint8_t bit_pos = (idx & 0x7);
        uint8_t mask_byte = rs1[byte_idx];
        uint8_t bit_shifted = (mask_byte >> bit_pos);
        uint8_t is_nonzero = (bit_shifted & 0x1);
        int8_t zero_i8 = 0;
        int8_t sparse_val = (is_nonzero ? rs1[4 + vidx] : zero_i8);
        dense_values[idx] = sparse_val;
        uint32_t inc_val = (is_nonzero ? 1 : 0);
        vidx = (vidx + inc_val);
    }
    for (idx = 0; ((idx + 1) < 32); ++idx) {
        int8_t val = dense_values[idx];
        int32_t mul_result = (val * global_scale);
        int16_t dequant = ((mul_result >> 8) & 0xFFFF);
        rd[idx] = dequant;
    }
    // burst_write lowered via register-backed scratchpad
    rd_result = 0;
    return rd_result;
}
