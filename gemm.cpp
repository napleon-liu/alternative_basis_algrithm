#include "gemm.h"
// 矩阵加法
inline Matrix matrixAdd(const Matrix &A, const Matrix &B)
{
    int n = A.size();
    Matrix res(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i][j] = A[i][j] + B[i][j];
    return res;
}

// 矩阵减法
inline Matrix matrixSub(const Matrix &A, const Matrix &B)
{
    int n = A.size();
    Matrix res(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i][j] = A[i][j] - B[i][j];
    return res;
}

// 标量乘矩阵
inline Matrix scalarMult(int scalar, const Matrix &A)
{
    int n = A.size();
    Matrix res(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i][j] = scalar * A[i][j];
    return res;
}

// 基例：小矩阵乘法（假设基例为 2x2 矩阵）
inline Matrix baseMult(const Matrix &A, const Matrix &B)
{
    Matrix res(2, vector<double>(2));
    res[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    res[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    res[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    res[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
    return res;
}

// 分块函数（假设矩阵是 2 的幂次）
inline vector<Matrix> splitMatrix(const Matrix &A)
{
    int n = A.size();
    int newSize = n / 2;
    vector<Matrix> blocks(4);
    for (int i = 0; i < 4; ++i)
        blocks[i] = Matrix(newSize, vector<double>(newSize));

    for (int i = 0; i < newSize; ++i)
    {
        for (int j = 0; j < newSize; ++j)
        {
            blocks[0][i][j] = A[i][j];                     // 左上块
            blocks[1][i][j] = A[i][j + newSize];           // 右上块
            blocks[2][i][j] = A[i + newSize][j];           // 左下块
            blocks[3][i][j] = A[i + newSize][j + newSize]; // 右下块
        }
    }
    return blocks;
}

// 合并分块矩阵
inline Matrix mergeMatrix(const vector<Matrix> &blocks)
{
    int n = blocks[0].size() * 2;
    Matrix res(n, vector<double>(n));
    int newSize = n / 2;

    for (int i = 0; i < newSize; ++i)
    {
        for (int j = 0; j < newSize; ++j)
        {
            res[i][j] = blocks[0][i][j];
            res[i][j + newSize] = blocks[1][i][j];
            res[i + newSize][j] = blocks[2][i][j];
            res[i + newSize][j + newSize] = blocks[3][i][j];
        }
    }
    return res;
}

// 基变换
inline std::vector<Matrix> psi_opt(std::vector<Matrix> &A)
{
    std::vector<Matrix> result(4);
    result[0] = A[0];
    result[1] = matrixAdd(matrixSub(A[1], A[2]), A[3]);
    result[2] = matrixSub(A[3], A[2]);
    result[3] = matrixAdd(A[1], A[3]);
    return result;
}

// 逆基变换
inline std::vector<Matrix> psi_opt_inv(std::vector<Matrix> &A)
{
    std::vector<Matrix> result(4);
    result[0] = A[0];
    result[1] = matrixSub(A[1], A[2]);
    result[2] = matrixSub(A[3], A[1]);
    result[3] = matrixSub(matrixAdd(A[2], A[3]), A[1]);
    return result;
}

// 递归进行基变换
Matrix basis_transformation(const Matrix &A, int depth)
{
    if (depth == 0)
    {
        return A;
    }
    // 分块，分成四块
    std::vector<Matrix> blocks = splitMatrix(A);
    std::vector<Matrix> transformed_blocks;
    for (const auto &block : blocks)
    {
        transformed_blocks.push_back(basis_transformation(block, depth - 1));
    }
    transformed_blocks = psi_opt(transformed_blocks);
    Matrix merged = mergeMatrix(transformed_blocks);
    return merged;
}

// 递归进行逆基变换
Matrix inverse_basis_transformation(const Matrix &A, int depth)
{
    if (depth == 0)
    {
        return A;
    }
    std::vector<Matrix> blocks = splitMatrix(A);
    std::vector<Matrix> transformed_blocks;
    for (const auto &block : blocks)
    {
        transformed_blocks.push_back(inverse_basis_transformation(block, depth - 1));
    }
    transformed_blocks = psi_opt_inv(transformed_blocks);
    Matrix merged = mergeMatrix(transformed_blocks);
    return merged;
}

// 新增辅助函数
Matrix generateRandomMatrix(int size, std::function<int()> generator)
{
    Matrix mat(size, std::vector<double>(size));
    for (auto &row : mat)
    {
        for (auto &elem : row)
        {
            elem = generator();
        }
    }
    return mat;
}

void printMatrix(const Matrix &mat)
{
    for (const auto &row : mat)
    {
        for (const auto &elem : row)
        {
            std::cout << std::setw(6) << elem << " ";
        }
        std::cout << "\n";
    }
}
// 暴力矩阵乘法（标准三重循环实现），用于对比
Matrix bruteForceMultiply(const Matrix &A, const Matrix &B)
{
    int n = A.size();
    Matrix res(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return res;
}

Matrix ABMultiply(const Matrix &A, const Matrix &B)
{
    int n = A.size();
    if (n == 1)
    {
        Matrix res(1, vector<double>(1));
        res[0][0] = A[0][0] * B[0][0];
        return res;
    }

    int new_size = n / 2;

    // 分割矩阵
    std::vector<Matrix> A_blocks = splitMatrix(A);
    std::vector<Matrix> B_blocks = splitMatrix(B);

    Matrix A11 = A_blocks[0];
    Matrix A12 = A_blocks[1];
    Matrix A21 = A_blocks[2];
    Matrix A22 = A_blocks[3];

    Matrix B11 = B_blocks[0];
    Matrix B12 = B_blocks[1];
    Matrix B21 = B_blocks[2];
    Matrix B22 = B_blocks[3];

    // 计算S
    std::vector<Matrix> S(7);
    S[0] = A22;
    S[1] = A21;
    S[2] = A12;
    S[3] = A11;
    S[4] = matrixSub(A12, A21);
    S[5] = matrixSub(A12, A11);
    S[6] = matrixSub(A22, A12);

    // 计算T
    std::vector<Matrix> T(7);
    T[0] = B22;
    T[1] = B21;
    T[2] = B12;
    T[3] = B11;
    T[4] = matrixSub(B22, B12);
    T[5] = matrixSub(B12, B21);
    T[6] = matrixSub(B12, B11);

    // 计算M
    std::vector<Matrix> M(8);
    for (int i = 0; i < 7; i++)
    {
        M[i] = ABMultiply(S[i], T[i]);
    }

    // 计算U
    /*
    U[1]=M[4]+M[5], U[2]=M[3]+M[5]-M[6]+M[7], U[3]=M[2]+M[7], U[4]=M[1]-M[6]
    */
    std::vector<Matrix> U(4);
    U[0] = matrixAdd(M[3], M[4]);
    U[1] = matrixAdd(matrixSub(matrixAdd(M[2], M[4]), M[5]), M[6]);
    U[2] = matrixAdd(M[1], M[6]);
    U[3] = matrixSub(M[0], M[5]);

    Matrix C = mergeMatrix(U);
    return C;
}

Matrix wrapMultiply(const Matrix &A, const Matrix &B)
{
    int dim = A.size();
    int depth = int(sqrt(dim));

    // phi transformation
    Matrix A_phi = basis_transformation(A, depth);
    Matrix B_phi = basis_transformation(B, depth);
    // multiply
    Matrix C_phi = ABMultiply(A_phi, B_phi);
    // psi inverse transformation
    Matrix C = inverse_basis_transformation(C_phi, depth);

    return C;
}
