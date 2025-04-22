#include "memory.cuh"

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N) 
    {
        double sum = 0.0;
        
        for(int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row*N + col] = sum;
    }
}