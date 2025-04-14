import numpy as np
import time 
BASE_SIZE = 2
EPSILON = 1e-6


def split_block(block, M, N):
    halfM, halfN = M // 2, N // 2
    return [
        block[:halfM, :halfN],
        block[:halfM, halfN:],
        block[halfM:, :halfN],
        block[halfM:, halfN:]
    ]


def basis_transformation(A, C):
    halfM, halfN = A[0].shape
    for i in range(halfM):
        for j in range(halfN):
            a0 = A[0][i, j]
            a1 = A[1][i, j]
            a2 = A[2][i, j]
            a3 = A[3][i, j]
            C[0][i, j] = a0
            C[1][i, j] = a1 - a2 + a3
            C[2][i, j] = a3 - a2
            C[3][i, j] = a1 + a3


def inv_basis_transformation(A, C):
    halfM, halfN = A[0].shape
    for i in range(halfM):
        for j in range(halfN):
            a0 = A[0][i, j]
            a1 = A[1][i, j]
            a2 = A[2][i, j]
            a3 = A[3][i, j]
            C[0][i, j] = a0
            C[1][i, j] = a1 - a2
            C[2][i, j] = a3 - a1
            C[3][i, j] = a2 + a3 - a1


def recursive_basis_transform(A, C, M, N):
    if M <= BASE_SIZE or N <= BASE_SIZE:
        basis_transformation(A, C)
        return
    A_sub = [split_block(a, M // 2, N // 2) for a in A]
    C_sub = [split_block(c, M // 2, N // 2) for c in C]
    for i in range(4):
        recursive_basis_transform(A_sub[i], C_sub[i], M // 2, N // 2)
    basis_transformation(C, C)


def inv_recursive_basis_transform(A, C, M, N):
    if M <= BASE_SIZE or N <= BASE_SIZE:
        inv_basis_transformation(A, C)
        return
    A_sub = [split_block(a, M // 2, N // 2) for a in A]
    C_sub = [split_block(c, M // 2, N // 2) for c in C]
    for i in range(4):
        recursive_basis_transform(A_sub[i], C_sub[i], M // 2, N // 2)
    inv_basis_transformation(C, C)


def matrix_add(A, B):
    return A + B


def matrix_sub(A, B):
    return A - B


def ABMultiply(A, B, n):
    if n == 1:
        return A * B
    half = n // 2
    A11, A12, A21, A22 = split_block(A, n, n)
    B11, B12, B21, B22 = split_block(B, n, n)

    S = [np.copy(A22), np.copy(A21), np.copy(A12), np.copy(A11)]
    S.append(matrix_sub(A12, A21))
    S.append(matrix_sub(A12, A11))
    S.append(matrix_sub(A22, A12))

    T = [np.copy(B22), np.copy(B21), np.copy(B12), np.copy(B11)]
    T.append(matrix_sub(B22, B12))
    T.append(matrix_sub(B12, B21))
    T.append(matrix_sub(B12, B11))

    M = [ABMultiply(S[i], T[i], half) for i in range(7)]

    U0 = matrix_add(M[3], M[4])
    U1 = matrix_add(matrix_sub(matrix_add(M[2], M[4]), M[5]), M[6])
    U2 = matrix_add(M[1], M[6])
    U3 = matrix_sub(M[0], M[5])

    top = np.hstack((U0, U1))
    bottom = np.hstack((U2, U3))
    return np.vstack((top, bottom))


def Multiply(A, B):
    n = A.shape[0]
    A_parts = split_block(A, n, n)
    B_parts = split_block(B, n, n)
    Ap = np.zeros_like(A)
    Bp = np.zeros_like(B)
    Ap_parts = split_block(Ap, n, n)
    Bp_parts = split_block(Bp, n, n)

    recursive_basis_transform(A_parts, Ap_parts, n, n)
    recursive_basis_transform(B_parts, Bp_parts, n, n)

    C = ABMultiply(Ap, Bp, n)

    C_parts = split_block(C, n, n)
    Cp = np.copy(C)
    Cp_parts = split_block(Cp, n, n)
    inv_recursive_basis_transform(Cp_parts, C_parts, n, n)

    return C


import numpy as np

EPSILON = 1e-6

def main():
    n = 16  # 维度为 2 的幂
    np.random.seed(42)

    # 随机初始化 A 和 B
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # NumPy @ 运算时间
    start_np = time.perf_counter()
    C_expected = A @ B
    end_np = time.perf_counter()

    # 自定义 Multiply 运算时间
    start_custom = time.perf_counter()
    C_actual = Multiply(A, B)
    end_custom = time.perf_counter()

    # 输出
    print(f"NumPy @ Time:      {end_np - start_np:.6f} seconds")
    print(f"Custom Multiply:   {end_custom - start_custom:.6f} seconds")
    print("Equal (within EPSILON):", np.allclose(C_expected, C_actual, atol=EPSILON))

if __name__ == "__main__":
    main()
