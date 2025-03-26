#include "matrix.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

void multiply_cpu_partial(double* A, double* B, double* C, int N, int startRow, int endRow) 
{
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void multiply_gpu_partial(double* A, double* B, double* C, int N, int startRow) 
{
    int row = startRow + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Matrix UMAMultiply(const Matrix& matA, const Matrix& matB) 
{
    int N = matA.size();
    if (N == 0 || matA[0].size() != N || matB.size() != N || matB[0].size() != N) {
        throw std::invalid_argument("Only square matrices of the same size are supported.");
    }

    double *A, *B, *C;
    cudaMallocManaged(&A, N * N * sizeof(double));
    cudaMallocManaged(&B, N * N * sizeof(double));
    cudaMallocManaged(&C, N * N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = matA[i][j];
            B[i * N + j] = matB[i][j];
        }
    }

    int midRow = N / 2;

    std::thread cpu_thread(multiply_cpu_partial, A, B, C, N, 0, midRow);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N - midRow + 15) / 16);

    multiply_gpu_partial<<<blocks, threads>>>(A, B, C, N, midRow);

    cudaDeviceSynchronize();
    cpu_thread.join();

    Matrix result(N, std::vector<double>(N));
    
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            result[i][j] = C[i * N + j];

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return result;
}