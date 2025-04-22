#include "matrix.cuh"

#include <vector>
#include <thread>
#include <future>
#include <stdexcept>

#include <vector>
#include <thread>
#include <future>
#include <stdexcept>

void compute_block(const Matrix& A, const Matrix& B, Matrix& result, size_t startRow, size_t endRow, size_t cols)
{
    for (size_t i = startRow; i < endRow; ++i) 
    {
        for (size_t j = 0; j < cols; ++j) 
        {
            double sum = 0;
            for (size_t k = 0; k < A[0].size(); ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
}

Matrix CPUMultiMultiply(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    }

    size_t rows = A.size();
    size_t cols = B[0].size();
    Matrix result(rows, std::vector<double>(cols, 0.0));

    size_t numThreads = 8;
    size_t blockSize = rows / numThreads;

    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < numThreads; ++i) {
        size_t startRow = i * blockSize;
        size_t endRow = (i == numThreads - 1) ? rows : startRow + blockSize;
        futures.push_back(std::async(std::launch::async, compute_block, std::cref(A), std::cref(B), std::ref(result), startRow, endRow, cols));
    }

    for (auto& future : futures) {
        future.get();
    }

    return result;
}
