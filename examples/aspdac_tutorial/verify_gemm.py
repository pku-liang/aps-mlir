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


def init_matrix_b():
    """Initialize matrix B with pattern: B[i][j] = (i * 2 + j) % 8 + 1"""
    B = np.zeros((K, N), dtype=np.int8)
    for i in range(K):
        for j in range(N):
            B[i, j] = (i * 2 + j) % 7 + 1
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

    # Initialize matrices
    A = init_matrix_a()
    B = init_matrix_b()

    print(f"\nMatrix A ({M}x{K}, int8):")
    print(f"  Pattern: A[i][j] = (i + j) % 8 + 1")
    print(f"  Range: [{A.min()}, {A.max()}]")
    print(f"  Top-left 8x8:\n{A[:8, :8]}")
    print(f"  Top-left 8x8 (hex):")
    for row in A[:8, :8]:
        print("    " + " ".join(f"0x{v:02X}" for v in row))

    print(f"\nMatrix B ({K}x{N}, int8):")
    print(f"  Pattern: B[i][j] = (i * 2 + j) % 8 + 1")
    print(f"  Range: [{B.min()}, {B.max()}]")
    print(f"  Top-left 8x8:\n{B[:8, :8]}")
    print(f"  Top-left 8x8 (hex):")
    for row in B[:8, :8]:
        print("    " + " ".join(f"0x{v:02X}" for v in row))

    # Compute GEMM
    D = compute_gemm(A, B)

    print(f"\nMatrix D = A * B ({M}x{N}, int16):")
    print(f"  Range: [{D.min()}, {D.max()}]")
    print(f"  Top-left 8x8:\n{D[:8, :8]}")
    for row in D[:8, :8]:
        print("    " + " ".join(f"0x{v:02X}" for v in row))

    print(f"  Another 8x8:\n{D[48:48+8, 64:64+8]}")

    # print(compute_gemm(A[:8, :8], B[:8, :8]))
    # print(compute_gemm(A[:8, 8:16], B[:8, 8:16]))
    # print()

    # Save expected results to file for comparison
    print("\n" + "=" * 60)
    print("Expected Results (for verification)")
    print("=" * 60)

    # Print some specific values for spot-checking
    test_points = [(0, 0), (0, 127), (127, 0), (127, 127), (64, 64)]
    print("\nSpot check values:")
    for i, j in test_points:
        print(f"  D[{i}][{j}] = {D[i, j]}")

    # Save matrices to binary files
    A.tofile("expected_A.bin")
    B.tofile("expected_B.bin")
    D.tofile("expected_D.bin")
    print("\nSaved binary files:")
    print("  expected_A.bin (int8, row-major)")
    print("  expected_B.bin (int8, row-major)")
    print("  expected_D.bin (int16, row-major)")

    # Also save as text for easy inspection
    np.savetxt("expected_D.txt", D, fmt="%d", delimiter=",")
    print("  expected_D.txt (text format)")

    # Compute checksum for quick verification
    checksum = np.sum(D.astype(np.int64))
    print(f"\nChecksum (sum of all D elements): {checksum}")

    return D


if __name__ == "__main__":
    D = main()
