#include "matrix.cuh"

Matrix CPUSingleMultiply(const Matrix& A, const Matrix& B) 
{
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Несовместимые размеры матриц для умножения.");
    }

    size_t rows = A.size();
    size_t inner = A[0].size();
    size_t cols = B[0].size();

    Matrix result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}