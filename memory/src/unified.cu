#include "memory.cuh"

std::vector<double> matrixMulUnified(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    if(A.size() != M*K || B.size() != K*N)
        throw std::invalid_argument("Invalid matrix dimensions");

    double *u_A, *u_B, *u_C;
    CUDA_CHECK(cudaMallocManaged(&u_A, M*K*sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&u_B, K*N*sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&u_C, M*N*sizeof(double)));

    std::copy(A.begin(), A.end(), u_A);
    std::copy(B.begin(), B.end(), u_B);

    dim3 blocks((N + BLOCK_SIZE-1)/BLOCK_SIZE, (M + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matrixMulKernel<<<blocks, threads>>>(u_A, u_B, u_C, M, N, K);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> result(u_C, u_C + M*N);
    
    cudaFree(u_A); 
    cudaFree(u_B); 
    cudaFree(u_C);
    
    return result;
}