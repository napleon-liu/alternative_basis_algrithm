#include "gemm.h"

int main()
{
    int n = 4;

    // 生成随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-10, 10);

    int size = 16;
    Matrix A = generateRandomMatrix(size, [&]()
                                    { return distrib(gen); });
    Matrix B = generateRandomMatrix(size, [&]()
                                    { return distrib(gen); });

    // 打印原始矩阵
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B);

    // 进行基变换
    Matrix A_transformed = basis_transformation(A, 2);
    Matrix B_transformed = basis_transformation(B, 2);

    // 打印基变换后的矩阵
    std::cout << "Matrix A after basis transformation:" << std::endl;
    printMatrix(A_transformed);
    std::cout << "Matrix B after basis transformation:" << std::endl;
    printMatrix(B_transformed);

    // 进行逆基变换
    Matrix A_inverse_transformed = inverse_basis_transformation(A_transformed, 4);
    Matrix B_inverse_transformed = inverse_basis_transformation(B_transformed, 4);

    // 打印逆基变换后的矩阵
    std::cout << "Matrix A after inverse basis transformation:" << std::endl;
    printMatrix(A_inverse_transformed);
    std::cout << "Matrix B after inverse basis transformation:" << std::endl;
    printMatrix(B_inverse_transformed);

    // 进行矩阵乘法
    Matrix C = bruteForceMultiply(A, B);
    Matrix C_transformed = bruteForceMultiply(A_transformed, B_transformed);
    Matrix C_Strassen = wrapMultiply(A, B);

    // 打印结果
    std::cout << "Result of matrix multiplication:" << std::endl;
    printMatrix(C);
    // std::cout << "Result of transformed matrix multiplication:" << std::endl;
    // printMatrix(C_transformed);

    // Matrix C_inverse_transformed = inverse_basis_transformation(C_transformed, 2);
    // // 打印逆基变换后的矩阵
    // std::cout << "Matrix C after inverse basis transformation:" << std::endl;
    // printMatrix(C_inverse_transformed);
    std::cout << "Result of Strassen matrix multiplication:" << std::endl;
    printMatrix(C_Strassen);

    return 0;
}
