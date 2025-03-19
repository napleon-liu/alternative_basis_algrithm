#pragma once
#include "matrix.h"

// 矩阵加法
inline Matrix matrixAdd(const Matrix &A, const Matrix &B);

// 矩阵减法
inline Matrix matrixSub(const Matrix &A, const Matrix &B);

// 标量乘矩阵
inline Matrix scalarMult(int scalar, const Matrix &A);

// 基例：小矩阵乘法（假设基例为 2x2 矩阵）
inline Matrix baseMult(const Matrix &A, const Matrix &B);

// 分块函数（假设矩阵是 2 的幂次）
inline vector<Matrix> splitMatrix(const Matrix &A);

// 合并分块矩阵
inline Matrix mergeMatrix(const vector<Matrix> &blocks);

// 基变换
inline std::vector<Matrix> psi_opt(std::vector<Matrix> &A);

// 逆基变换
inline std::vector<Matrix> psi_opt_inv(std::vector<Matrix> &A);

// 递归进行基变换
Matrix basis_transformation(const Matrix &A, int depth);

// 递归进行逆基变换
Matrix inverse_basis_transformation(const Matrix &A, int depth);

// 新增辅助函数
Matrix generateRandomMatrix(int size, std::function<int()> generator);

// 打印矩阵
void printMatrix(const Matrix &mat);

// 暴力矩阵乘法（标准三重循环实现），用于对比
Matrix bruteForceMultiply(const Matrix &A, const Matrix &B);

// Strassen 矩阵乘法
Matrix strassenMultiply(const Matrix &A, const Matrix &B);

// 矩阵乘
Matrix wrapMultiply(const Matrix &A, const Matrix &B);