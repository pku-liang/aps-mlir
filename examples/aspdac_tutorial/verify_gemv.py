#!/usr/bin/env python3
"""
Verify GEMM 128x128 result.

This script computes the expected result for D = A * B using the same
initialization patterns as gemm_128x128.c:
- A[i][j] = (i + j) % 8 + 1  (values 1-8)
- B[i][j] = (i * 2 + j) % 8 + 1  (values 1-8)

Output D is i16 (int16).
"""

import numpy as np

M = 128
N = 128
K = 128


def init_matrix_a():
    """Initialize matrix A with pattern: A[i][j] = (i + j) % 8 + 1"""
    A = np.zeros((M, K), dtype=np.int8)
    for i in range(M):
        for j in range(K):
            A[i, j] = (i + j) % 7 + 1
    return A


def init_vec_b():
    """Initialize matrix B with pattern: B[i][j] = (i * 2 + j) % 8 + 1"""
    B = np.zeros((K, 1), dtype=np.int8)
    for i in range(K):
        B[i] = (i * 2) % 7 + 1
    return B


def compute_gemm(A, B):
    """Compute D = A * B with i16 output"""
    # Cast to int16 before multiplication to avoid overflow
    A_i16 = A.astype(np.int16)
    B_i16 = B.astype(np.int16)
    D = np.matmul(A_i16, B_i16)
    return D.astype(np.int16)


def main():
    print("=" * 60)
    print("GEMM 128x128 Verification")
    print("=" * 60)
    np.set_printoptions(threshold=np.inf)
    # Initialize matrices
    A = init_matrix_a()
    B = init_vec_b()

    print(A[:8, :16])
    print(A[:8, 16:32])
    print(B.reshape(-1))

    # Compute GEMM
    D = compute_gemm(A, B)

    print(D.reshape(-1))
    return D


if __name__ == "__main__":
    D = main()
