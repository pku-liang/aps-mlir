// Auto-generated from CADL by cadl-to-c
// DO NOT EDIT - Regenerate from CADL source

#include <stdint.h>

uint32_t vfpsmax_v(uint32_t *rs1, uint32_t *rs2) {
    uint32_t indices[16];
    uint32_t reduction_idxs_in[16];
    uint32_t reduction_idxs_out[16];
    uint32_t reduction_vals_out[16];

    uint32_t rd_result = 0;
    uint32_t * addr = rs1;
    uint32_t vl = 16;
    // burst_read lowered via register-backed scratchpad
    uint32_t i;
    for (i = 0; i < 16; ++i) {
        indices[i] = i;
        reduction_idxs_in[i] = i;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 8; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = rs1[idx1];
        uint32_t val2 = rs1[idx2];
        uint8_t is_sentinel1 = ((val1 == 0) ? 1 : 0);
        uint8_t is_sentinel2 = ((val2 == 0) ? 1 : 0);
        uint32_t max_val = (is_sentinel1 ? val2 : (is_sentinel2 ? val1 : ((val1 > val2) ? val1 : val2)));
        uint32_t reduction_idxs_in_idx2 = reduction_idxs_in[idx2];
        uint32_t reduction_idxs_in_idx1 = reduction_idxs_in[idx1];
        uint32_t max_idx = (is_sentinel1 ? reduction_idxs_in_idx2 : (is_sentinel2 ? reduction_idxs_in_idx1 : ((val1 > val2) ? reduction_idxs_in_idx1 : reduction_idxs_in_idx2)));
        reduction_vals_out[i] = max_val;
        reduction_idxs_out[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 4; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = reduction_vals_out[idx1];
        uint32_t val2 = reduction_vals_out[idx2];
        uint32_t max_val = ((val1 > val2) ? val1 : val2);
        uint32_t reduction_idxs_out_idx1 = reduction_idxs_out[idx1];
        uint32_t reduction_idxs_out_idx2 = reduction_idxs_out[idx2];
        uint32_t max_idx = ((val1 > val2) ? reduction_idxs_out_idx1 : reduction_idxs_out_idx2);
        rs1[i] = max_val;
        reduction_idxs_in[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    for (i = 0; i < 2; ++i) {
        uint32_t idx1 = (i * 2);
        uint32_t idx2 = ((i * 2) + 1);
        uint32_t val1 = rs1[idx1];
        uint32_t val2 = rs1[idx2];
        uint32_t max_val = ((val1 > val2) ? val1 : val2);
        uint32_t reduction_idxs_in_idx1 = reduction_idxs_in[idx1];
        uint32_t reduction_idxs_in_idx2 = reduction_idxs_in[idx2];
        uint32_t max_idx = ((val1 > val2) ? reduction_idxs_in_idx1 : reduction_idxs_in_idx2);
        reduction_vals_out[i] = max_val;
        reduction_idxs_out[i] = max_idx;
        uint32_t i_ = (i + 1);
    }
    uint32_t val1 = reduction_vals_out[0];
    uint32_t val2 = reduction_vals_out[1];
    uint32_t final_max = ((val1 > val2) ? val1 : val2);
    uint32_t reduction_idxs_out_0 = reduction_idxs_out[0];
    uint32_t reduction_idxs_out_1 = reduction_idxs_out[1];
    uint32_t final_idx = ((val1 > val2) ? reduction_idxs_out_0 : reduction_idxs_out_1);
    rd_result = final_idx;
    uint32_t * out_addr = rs2;
    rs2[0] = final_max;
    rs2[1] = final_idx;
    return rd_result;
}
