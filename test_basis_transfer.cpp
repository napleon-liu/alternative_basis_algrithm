#include "gemm.h"

int main()
{
    int n = 4;

    // 生成随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-10, 10);

    int size = 4;
    Matrix A = generateRandomMatrix(size, [&]()
                                    { return distrib(gen); });
    Matrix B = generateRandomMatrix(size, [&]()
                                    { return distrib(gen); });

    // 打印原始矩阵
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B);

    // 进行矩阵乘法
    Matrix C = bruteForceMultiply(A, B);
    Matrix C_Strassen = wrapMultiply(A, B);

    // 打印结果
    std::cout << "Result of matrix multiplication:" << std::endl;
    printMatrix(C);
    std::cout << "Result of AB matrix multiplication:" << std::endl;
    printMatrix(C_Strassen);

    return 0;
}
