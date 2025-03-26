#include "matrix.cuh"

void compute_element(const Matrix& A, const Matrix& B, Matrix& result, size_t row, size_t col) 
{
    double sum = 0;

    for (size_t k = 0; k < A[0].size(); ++k) {
        sum += A[row][k] * B[k][col];
    }

    result[row][col] = sum;
}

Matrix ThreadMultiply(const Matrix& A, const Matrix& B) 
{
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    }

    size_t rows = A.size();
    size_t cols = B[0].size();
    Matrix result(rows, std::vector<double>(cols, 0.0));

    std::vector<std::thread> threads;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            threads.emplace_back(compute_element, std::cref(A), std::cref(B), std::ref(result), i, j);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}