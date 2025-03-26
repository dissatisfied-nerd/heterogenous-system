#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <thread>
#include <stdexcept>

using Matrix = std::vector<std::vector<double>>;

Matrix CPUMultiply(const Matrix& A, const Matrix& B);
Matrix ThreadMultiply(const Matrix& A, const Matrix& B);
Matrix UMAMultiply(const Matrix& A, const Matrix& B);
void CUDAMultiply(const Matrix& A, const Matrix& B, Matrix& res);

#endif
