/**
 * GEMM 128x128 with W8A8 (8-bit weights, 8-bit activations)
 *
 * Memory layouts:
 * - DRAM: Standard row-major (128 contiguous elements per row)
 * - SPM (Scratchpad): Tile-organized (8x8 tiles stored contiguously)
 *
 * The DMA instructions handle the conversion:
 * - loadmat_x_8tile: Reads 8 rows with stride from DRAM, writes as tile to SPM
 * - storemat_c_8tile: Reads tile from SPM, writes 8 rows with stride to DRAM
 */

#include <stdint.h>

// ============================================================================
// Custom Instruction Encodings
// ============================================================================

// gemm8x8_i8: opcode=0b0001011, funct7=0b0000000
#define GEMM8X8_I8(rd, rs1, rs2) \
    asm volatile (".insn r 0x0B, 0, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

// loadmat_a_8tile: opcode=0b1111011, funct7=0b0000000
// Loads 8 consecutive 8x8 tiles from row-major DRAM to tiled SPM
#define LOADMAT_A_8TILE(rd, rs1, rs2) \
    asm volatile (".insn r 0x7B, 0, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

// loadmat_b_8tile: opcode=0b1111011, funct7=0b0000001
// Loads 8 consecutive 8x8 tiles from row-major DRAM to tiled SPM
#define LOADMAT_B_8TILE(rd, rs1, rs2) \
    asm volatile (".insn r 0x7B, 0, 1, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

// storemat_c_8tile: opcode=0b1111011, funct7=0b0000010
// Stores 8 consecutive 8x8 tiles from tiled SPM to row-major DRAM
#define STOREMAT_C_8TILE(rd, rs1, rs2) \
    asm volatile (".insn r 0x7B, 0, 2, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2))

// ============================================================================
// Matrix Dimensions
// ============================================================================

#define M 128           // Rows of A and C/D
#define N 128           // Cols of B and C/D
#define K 128           // Cols of A, Rows of B
#define TILE_SIZE 8     // Hardware tile size

#define M_TILES (M / TILE_SIZE)  // 16 tiles in M dimension
#define N_TILES (N / TILE_SIZE)  // 16 tiles in N dimension
#define K_TILES (K / TILE_SIZE)  // 16 tiles in K dimension

// DRAM row strides (matrix width in bytes)
#define A_DRAM_STRIDE K         // A is MxK, stride = K bytes (i8)
#define B_DRAM_STRIDE N         // B is KxN, stride = N bytes (i8)
#define C_DRAM_STRIDE (N * 2)   // C is MxN, stride = N*2 bytes (i16)

// ============================================================================
// Helper Functions for Instruction Encoding
// ============================================================================

/**
 * Pack rs2 for DMA load/store instructions:
 *   rs2[31:16] = spm_addr >> 2 (SPM address in 4-byte units)
 *   rs2[28:24] = block_w >> 6 (DRAM row stride in 64-byte units)
 *
 * @param spm_tile_idx: Tile index in SPM (0, 1, 2, ...)
 * @param tile_bytes: Bytes per tile (64 for i8, 128 for i16)
 * @param dram_row_stride: DRAM row stride in bytes
 */
static inline uint32_t pack_dma_rs2(uint32_t spm_tile_idx, uint32_t tile_bytes, uint32_t dram_row_stride) {
    // SPM address = tile_idx * tile_bytes, then >> 2 for 4-byte units
    uint32_t spm_addr = spm_tile_idx * tile_bytes;
    uint32_t spm_field = (spm_addr >> 2) << 16;  // bits [31:16]

    // block_w in 64-byte units, placed in bits [28:24]
    uint32_t block_w_field = ((dram_row_stride >> 6) & 0x1F) << 24;

    return spm_field | block_w_field;
}

/**
 * Pack rs1 for GEMM instruction:
 *   rs1[31:16] = offsetA (tile index for matrix A in SPM)
 *   rs1[15]    = use_c (1 = accumulate with C, 0 = ignore C)
 *   rs1[14:0]  = offsetB (tile index for matrix B in SPM)
 */
static inline uint32_t pack_gemm_rs1(uint16_t offsetA, uint8_t use_c, uint16_t offsetB) {
    return ((uint32_t)offsetA << 16) | ((uint32_t)(use_c & 1) << 15) | (offsetB & 0x7FFF);
}

/**
 * Pack rs2 for GEMM instruction:
 *   rs2[31:16] = offsetC (input accumulator tile index)
 *   rs2[15:0]  = offsetD (output tile index)
 */
static inline uint32_t pack_gemm_rs2(uint16_t offsetC, uint16_t offsetD) {
    return ((uint32_t)offsetC << 16) | offsetD;
}

// ============================================================================
// DMA Functions - Load from row-major DRAM to tiled SPM
// Each instruction loads/stores 8 tiles at once (8 horizontally-adjacent tiles)
// ============================================================================

#define TILES_PER_DMA 8  // Each DMA instruction handles 8 tiles

/**
 * Load matrix A (MxK) from DRAM to SPM
 * DRAM: row-major, 128 columns (K=128)
 * SPM: 16x16 tiles, each tile is 8x8 = 64 bytes
 * Each instruction loads 8 consecutive tiles along K dimension
 *
 * @param dram_base: Base address of matrix A in DRAM
 */
void load_matrix_a(uint32_t dram_base) {
    uint32_t rs1, rs2, rd;

    for (int i_tile = 0; i_tile < M_TILES; i_tile++) {
        for (int k_tile = 0; k_tile < K_TILES; k_tile += TILES_PER_DMA) {
            // DRAM address of first tile's top-left element in row-major layout
            uint32_t dram_addr = dram_base
                               + (i_tile * TILE_SIZE * A_DRAM_STRIDE)  // row offset
                               + (k_tile * TILE_SIZE * 1);             // col offset (i8 = 1 byte)

            // SPM tile index for first of the 8 tiles
            uint32_t spm_tile_idx = i_tile * K_TILES + k_tile;

            rs1 = dram_addr;
            rs2 = pack_dma_rs2(spm_tile_idx, 64, A_DRAM_STRIDE);  // 64 bytes per i8 tile

            LOADMAT_A_8TILE(rd, rs1, rs2);
        }
    }
}

/**
 * Load matrix B (KxN) from DRAM to SPM
 * DRAM: row-major, 128 columns (N=128)
 * SPM: 16x16 tiles, each tile is 8x8 = 64 bytes
 * Each instruction loads 8 consecutive tiles along N dimension
 *
 * @param dram_base: Base address of matrix B in DRAM
 */
void load_matrix_b(uint32_t dram_base) {
    uint32_t rs1, rs2, rd;

    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        for (int j_tile = 0; j_tile < N_TILES; j_tile += TILES_PER_DMA) {
            uint32_t dram_addr = dram_base
                               + (k_tile * TILE_SIZE * B_DRAM_STRIDE)
                               + (j_tile * TILE_SIZE * 1);

            uint32_t spm_tile_idx = k_tile * N_TILES + j_tile;

            rs1 = dram_addr;
            rs2 = pack_dma_rs2(spm_tile_idx, 64, B_DRAM_STRIDE);

            LOADMAT_B_8TILE(rd, rs1, rs2);
        }
    }
}

/**
 * Store matrix C/D (MxN) from SPM to DRAM
 * SPM: 16x16 tiles, each tile is 8x8 = 128 bytes (i16)
 * DRAM: row-major, 128 columns (N=128), 2 bytes per element
 * Each instruction stores 8 consecutive tiles along N dimension
 *
 * @param dram_base: Base address of output matrix in DRAM
 */
void store_matrix_c(uint32_t dram_base) {
    uint32_t rs1, rs2, rd;

    for (int i_tile = 0; i_tile < M_TILES; i_tile++) {
        for (int j_tile = 0; j_tile < N_TILES; j_tile += TILES_PER_DMA) {
            // For i16 elements: col offset = tile_col * TILE_SIZE * 2
            uint32_t dram_addr = dram_base
                               + (i_tile * TILE_SIZE * C_DRAM_STRIDE)
                               + (j_tile * TILE_SIZE * 2);

            uint32_t spm_tile_idx = i_tile * N_TILES + j_tile;

            rs1 = dram_addr;
            rs2 = pack_dma_rs2(spm_tile_idx, 128, C_DRAM_STRIDE);  // 128 bytes per i16 tile

            STOREMAT_C_8TILE(rd, rs1, rs2);
        }
    }
}

// ============================================================================
// GEMM Computation (operates on tiled SPM data)
// ============================================================================

/**
 * GEMM 128x128: D = A * B (no initial bias)
 * Assumes matA and matB are already loaded to SPM in tiled format.
 * Result is computed in matC SPM.
 */
void compute_gemm(void) {
    uint32_t rs1, rs2, rd;

    // Iterate over output tiles
    for (int i_tile = 0; i_tile < M_TILES; i_tile++) {
        for (int j_tile = 0; j_tile < N_TILES; j_tile++) {
            // Output tile index in SPM
            uint16_t offsetD = i_tile * N_TILES + j_tile;

            // Accumulate over K dimension
            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                // Input tile indices in SPM
                uint16_t offsetA = i_tile * K_TILES + k_tile;
                uint16_t offsetB = k_tile * N_TILES + j_tile;

                // First iteration: use_c=0 (start fresh, ignore matC)
                // Subsequent: use_c=1 (accumulate previous result)
                uint8_t use_c = (k_tile == 0) ? 0 : 1;
                uint16_t offsetC = offsetD;

                rs1 = pack_gemm_rs1(offsetA, use_c, offsetB);
                rs2 = pack_gemm_rs2(offsetC, offsetD);

                GEMM8X8_I8(rd, rs1, rs2);
            }
        }
    }
}

/**
 * GEMM 128x128: D = A * B + C (with initial bias)
 * Assumes matA, matB, and matC (bias) are loaded to SPM.
 */
void compute_gemm_with_bias(void) {
    uint32_t rs1, rs2, rd;

    for (int i_tile = 0; i_tile < M_TILES; i_tile++) {
        for (int j_tile = 0; j_tile < N_TILES; j_tile++) {
            uint16_t offsetD = i_tile * N_TILES + j_tile;

            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                uint16_t offsetA = i_tile * K_TILES + k_tile;
                uint16_t offsetB = k_tile * N_TILES + j_tile;

                // Always use_c=1 to include bias on first iteration
                // and accumulate on subsequent iterations
                uint8_t use_c = 1;
                uint16_t offsetC = offsetD;

                rs1 = pack_gemm_rs1(offsetA, use_c, offsetB);
                rs2 = pack_gemm_rs2(offsetC, offsetD);

                GEMM8X8_I8(rd, rs1, rs2);
            }
        }
    }
}

// ============================================================================
// Global Arrays in DRAM (row-major layout)
// ============================================================================

// Matrix A: 128x128 i8 = 16KB
int8_t A[M * K];

// Matrix B: 128x128 i8 = 16KB
int8_t B[K * N];

// Matrix D (output): 128x128 i16 = 32KB
int16_t D[M * N];

// ============================================================================
// Initialization Functions
// ============================================================================

/**
 * Initialize matrix A with pattern: A[i][j] = (i + j) % 8 + 1
 * Values range from 1 to 8 (all positive)
 */
void init_matrix_a(void) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = (int8_t)((i + j) % 8 + 1);
        }
    }
}

/**
 * Initialize matrix B with pattern: B[i][j] = (i * 2 + j) % 8 + 1
 * Values range from 1 to 8 (all positive)
 */
void init_matrix_b(void) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (int8_t)((i * 2 + j) % 8 + 1);
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Full GEMM 128x128 pipeline:
 * 1. Initialize A and B with test patterns
 * 2. Load A (row-major DRAM) -> matA SPM (tiled)
 * 3. Load B (row-major DRAM) -> matB SPM (tiled)
 * 4. Compute D = A * B in SPM (tiled)
 * 5. Store D (tiled SPM) -> (row-major DRAM)
 */
void gemm_128x128_full(void) {
    // Step 1: Initialize input matrices
    init_matrix_a();
    init_matrix_b();

    // Step 2: Load input matrices from row-major DRAM to tiled SPM
    load_matrix_a((uintptr_t)A);
    load_matrix_b((uintptr_t)B);

    // Step 3: Compute GEMM on tiled SPM data
    compute_gemm();

    // Step 4: Store result from tiled SPM to row-major DRAM
    store_matrix_c((uintptr_t)D);
}

int main(void) {
    // Perform full GEMM: D = A * B
    gemm_128x128_full();

    return 0;
}
