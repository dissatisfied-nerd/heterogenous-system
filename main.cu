#include "matrix.cuh"
#include <iostream>

void PrintMatrix(Matrix& result)
{
    for (const auto& row : result) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    Matrix A = {
        {1, 2},
        {4, 5}
    };

    Matrix B = {
        {7, 8},
        {9, 10}
    };

    Matrix res1, res2, res3, res4;

    res1 = CPUMultiply(A, B);
    res2 = ThreadMultiply(A, B);
    CUDAMultiply(A, B, res3);
    res4 = UMAMultiply(A, B);
  
    PrintMatrix(res1);
    PrintMatrix(res2);
    PrintMatrix(res3);
    PrintMatrix(res4);

    return 0;
}