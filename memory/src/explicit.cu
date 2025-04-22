#include "memory.cuh"

std::vector<double> matrixMulExplicit(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) 
{
    if (A.size() != M * K || B.size() != K * N){
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    double *d_A, *d_B, *d_C;
    std::vector<double> C(M*N);
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N *sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N *sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    
    return C;
}