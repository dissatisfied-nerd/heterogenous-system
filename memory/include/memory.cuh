#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <vector>
#include <stdexcept>

#define BLOCK_SIZE 16
#define CUDA_CHECK(err) { cudaError_t err_ = (err); if(err_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_)); }

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int M, int N, int K);

std::vector<double> matrixMulExplicit(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N);
std::vector<double> matrixMulUnified(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N);
std::vector<double> matrixMulZeroCopy(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N);
std::vector<double> matrixMulPinned(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N);

std::vector<double> flatten(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> unflatten(const std::vector<double>& flatMatrix, int rows, int cols);

#endif