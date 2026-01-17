const int N = 128;
const int TILE_SIZE = 8;


void gemm_isax(uint32_t offsetA, uint32_t offsetB, uint32_t offsetC, uint32_t offsetD) {
}

void load_A(uint8_t *dram_address, uint32_t block_w, uint32_t offset, uint32_t spm_start_addr) {
    // call isax
    // load (dram_address + offset) to (dram_address + offset + 4 * TILE_SIZE * TILE_SIZE) into SPM.
    // where dram_address correspond to spm_start_addr at scratchpad
};

void load_B(uint8_t *dram_address, uint32_t block_w, uint32_t offset, uint32_t spm_start_addr) {
    // call isax
};

void store_D(uint8_t *dram_address, uint32_t block_w, uint32_t offset, uint32_t spm_start_addr) {
    // call isax
};

void move_D_to_C(uint32_t offsetC, uint32_t offsetD) {
    // call isax
};

void gemm_tiled_8x8(const uint8_t* A, const uint8_t* B, uint8_t* D) {
    for (int i = 0; i < N * N; i+= 4 * TILE_SIZE * TILE_SIZE) {
        load_A((uint8_t*)(A + i), N, i, 0x0);
        load_B((uint8_t*)(B + i), N, i, 0x0);
    }
    
    // 使用8x8 tile进行矩阵乘法
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                if (kk == 0) {
                    gemm_isax(ii * (N / TILE_SIZE) + kk, 
                            kk * (N / TILE_SIZE) + jj, 
                            0x80000000, // don't use C input
                            ii * (N / TILE_SIZE) + jj);
                } else {
                    gemm_isax(ii * (N / TILE_SIZE) + kk, 
                            kk * (N / TILE_SIZE) + jj, 
                            ii * (N / TILE_SIZE) + jj,
                            ii * (N / TILE_SIZE) + jj);
                }
                if (kk == (N - TILE_SIZE)) {
                    move_D_to_C(ii * (N / TILE_SIZE) + jj, ii * (N / TILE_SIZE) + jj);
                }
            }
        }
    }
    for (int i = 0; i < N * N; i+= 4 * TILE_SIZE * TILE_SIZE) {
        store_D((uint32_t*)(A + i), N, i, 0x0);
    }
}