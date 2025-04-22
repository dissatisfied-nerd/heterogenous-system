#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <thread>
#include <stdexcept>

using Matrix = std::vector<std::vector<double>>;

Matrix CPUSingleMultiply(const Matrix& A, const Matrix& B);
Matrix CPUMultiMultiply(const Matrix& A, const Matrix& B);
void CUDAMultiply(const Matrix& A, const Matrix& B, Matrix& res);
float CUDAMultiplyProfile(const Matrix& A, const Matrix& B, Matrix& res);

#endif
