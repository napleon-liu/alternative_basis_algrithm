#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cstdio>
constexpr int BASE_SIZE = 2;

#define BASE_SIZE 2
#define MAX_SIZE 8
#define EPSILON 1e-6

using namespace std;

// 将矩阵块（起始指针 + 跨距）拷贝到扁平数组
void matrix_copy_block(double *src, int stride, double *dst, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dst[i * n + j] = src[i * stride + j];
}

// 将扁平块写回矩阵块（dst 是目标矩阵起点）
void write_block(double *dst, int stride, double *src, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dst[i * stride + j] = src[i * n + j];
}

// 矩阵加法
inline void matrix_add(double *A, double *B, double *C, int strideA, int strideB, int strideC, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i * strideC + j] = A[i * strideA + j] + B[i * strideB + j];
        }
    }
}

// 矩阵减法
inline void matrix_sub(double *A, double *B, double *C, int strideA, int strideB, int strideC, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i * strideC + j] = A[i * strideA + j] - B[i * strideB + j];
        }
    }
}

void print_matrix(double *A, int stride, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%6.2f ", A[i * stride + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void basis_transformation(double *A[4], double *C[4], int strideA, int strideC, int M, int N)
{
    int halfM = M / 2;
    int halfN = N / 2;
    for (int i = 0; i < halfM; ++i)
    {
        for (int j = 0; j < halfN; ++j)
        {
            int idx = i * strideA + j;

            double a0 = A[0][idx];
            double a1 = A[1][idx];
            double a2 = A[2][idx];
            double a3 = A[3][idx];

            C[0][idx] = a0;
            C[1][idx] = a1 - a2 + a3;
            C[2][idx] = a3 - a2;
            C[3][idx] = a1 + a3;
        }
    }
}

void inv_basis_transformation(double *A[4], double *C[4], int strideA, int strideC, int M, int N)
{
    int halfM = M / 2;
    int halfN = N / 2;
    for (int i = 0; i < halfM; ++i)
    {
        for (int j = 0; j < halfN; ++j)
        {
            int idx = i * strideA + j;

            double a0 = A[0][idx];
            double a1 = A[1][idx];
            double a2 = A[2][idx];
            double a3 = A[3][idx];

            C[0][idx] = a0;
            C[1][idx] = a1 - a2;
            C[2][idx] = a3 - a1;
            C[3][idx] = a2 + a3 - a1;
        }
    }
}

void split_block(double *block, double *sub[4], int stride, int M, int N)
{
    int halfM = M / 2;
    int halfN = N / 2;
    sub[0] = block;                          // 左上
    sub[1] = block + halfN;                  // 右上
    sub[2] = block + halfM * stride;         // 左下
    sub[3] = block + halfM * stride + halfN; // 右下
}

void recursive_basis_transform(double *A[4], double *C[4], int strideA, int strideC, int M, int N)
{
    if (M <= BASE_SIZE || N <= BASE_SIZE)
    {
        basis_transformation(A, C, strideA, strideC, M, N);
        return;
    }

    int halfM = M / 2;
    int halfN = N / 2;

    double *A_sub[4][4], *C_sub[4][4];

    for (int i = 0; i < 4; ++i)
    {
        split_block(A[i], A_sub[i], strideA, halfM, halfN);
        split_block(C[i], C_sub[i], strideC, halfM, halfN);
    }

    for (int i = 0; i < 4; ++i)
    {
        recursive_basis_transform(A_sub[i], C_sub[i], strideA, strideC, halfM, halfN);
    }

    basis_transformation(C, C, strideC, strideC, M, N);
}

void inv_recursive_basis_transform(double *A[4], double *C[4], int strideA, int strideC, int M, int N)
{
    if (M <= BASE_SIZE || N <= BASE_SIZE)
    {
        inv_basis_transformation(A, C, strideA, strideC, M, N);
        return;
    }

    int halfM = M / 2;
    int halfN = N / 2;

    double *A_sub[4][4], *C_sub[4][4];

    for (int i = 0; i < 4; ++i)
    {
        split_block(A[i], A_sub[i], strideA, halfM, halfN);
        split_block(C[i], C_sub[i], strideC, halfM, halfN);
    }

    for (int i = 0; i < 4; ++i)
    {
        inv_recursive_basis_transform(A_sub[i], C_sub[i], strideA, strideC, halfM, halfN);
    }

    inv_basis_transformation(C, C, strideC, strideC, M, N);
}

void naive_mul(double *A, double *B, double *C, int strideA, int strideB, int strideC, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            C[i * strideC + j] = 0;
            for (int k = 0; k < n; ++k)
                C[i * strideC + j] += A[i * strideA + k] * B[k * strideB + j];
        }
}

void ABMultiply(double *A, double *B, double *C, int strideA, int strideB, int strideC, int n)
{
    if (n < 2)
    {
        C[0] = A[0] * B[0];
        return;
    }

    int half = n / 2;

    double *A11 = A;
    double *A12 = A + 0 * strideA + half;
    double *A21 = A + half * strideA + 0;
    double *A22 = A + half * strideA + half;

    double *B11 = B;
    double *B12 = B + 0 * strideB + half;
    double *B21 = B + half * strideB + 0;
    double *B22 = B + half * strideB + half;

    double *C11 = C;
    double *C12 = C + 0 * strideC + half;
    double *C21 = C + half * strideC + 0;
    double *C22 = C + half * strideC + half;

    double S[7][half * half], T[7][half * half], M[7][half * half], U[4][half * half];

    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            S[0][i * half + j] = A22[i * strideA + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            S[1][i * half + j] = A21[i * strideA + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            S[2][i * half + j] = A12[i * strideA + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            S[3][i * half + j] = A11[i * strideA + j];
        }
    }
    matrix_sub(A12, A21, S[4], strideA, strideA, half, half);
    matrix_sub(A12, A11, S[5], strideA, strideA, half, half);
    matrix_sub(A22, A12, S[6], strideA, strideA, half, half);
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            T[0][i * half + j] = B22[i * strideB + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            T[1][i * half + j] = B21[i * strideB + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            T[2][i * half + j] = B12[i * strideB + j];
        }
    }
    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            T[3][i * half + j] = B11[i * strideB + j];
        }
    }
    matrix_sub(B22, B12, T[4], strideB, strideB, half, half);
    matrix_sub(B12, B21, T[5], strideB, strideB, half, half);
    matrix_sub(B12, B11, T[6], strideB, strideB, half, half);
    for (int i = 0; i < 7; ++i)
    {
        ABMultiply(S[i], T[i], M[i], half, half, half, half);
    }

    matrix_add(M[3], M[4], U[0], half, half, half, half);
    double tmp1[half * half], tmp2[half * half];
    matrix_add(M[2], M[4], tmp1, half, half, half, half);
    matrix_sub(tmp1, M[5], tmp2, half, half, half, half);
    matrix_add(tmp2, M[6], U[1], half, half, half, half);
    matrix_add(M[1], M[6], U[2], half, half, half, half);
    matrix_sub(M[0], M[5], U[3], half, half, half, half);

    for (int i = 0; i < half; ++i)
    {
        for (int j = 0; j < half; ++j)
        {
            C11[i * strideC + j] = U[0][i * half + j];
            C12[i * strideC + j] = U[1][i * half + j];
            C21[i * strideC + j] = U[2][i * half + j];
            C22[i * strideC + j] = U[3][i * half + j];
        }
    }
}

void Multiply(double *A, double *B, double *C, int strideA, int strideB, int strideC, int n)
{
    if (n < 64)
    {
        naive_mul(A, B, C, strideA, strideB, strideC, n);
        return;
    }

    double *A_parts[4];
    double *B_parts[4];
    double *C_parts[4];
    split_block(A, A_parts, strideA, n, n);
    split_block(B, B_parts, strideB, n, n);
    split_block(C, C_parts, strideC, n, n);

    double Ap[n * n], Bp[n * n];
    double *Ap_parts[4];
    double *Bp_parts[4];
    split_block(Ap, Ap_parts, n, n, n);
    split_block(Bp, Bp_parts, n, n, n);

    recursive_basis_transform(A_parts, Ap_parts, strideA, n, n, n);
    recursive_basis_transform(B_parts, Bp_parts, strideB, n, n, n);

    ABMultiply(Ap, Bp, C, n, n, strideC, n);

    split_block(C, C_parts, strideC, n, n);
    double Cp[n * n];
    double *Cp_parts[4];
    split_block(Cp, Cp_parts, n, n, n);
    std::memcpy(Cp, C, sizeof(Cp));

    inv_recursive_basis_transform(Cp_parts, C_parts, n, strideC, n, n);
}

int compare_matrix(double *A, double *B, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (fabs(A[i] - B[i]) > EPSILON)
        {
            return 0;
        }
    }
    return 1;
}

int main()
{
    int n = 1024;
    double A[n * n], B[n * n], C[n * n];

    for (int i = 0; i < n * n; ++i)
    {
        A[i] = static_cast<double>(i + 1);
        B[i] = static_cast<double>(i + 1);
    }

    auto start1 = std::chrono::high_resolution_clock::now();
    Multiply(A, B, C, n, n, n, n);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

    double C_naive[n * n];
    auto start2 = std::chrono::high_resolution_clock::now();
    naive_mul(A, B, C_naive, n, n, n, n);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

    if (compare_matrix(C, C_naive, n * n))
    {
        std::cout << "Result is correct." << std::endl;
    }

    std::cout << "Multiply function took " << duration1 << " microseconds." << std::endl;
    std::cout << "naive_mul function took " << duration2 << " microseconds." << std::endl;

    return 0;
}