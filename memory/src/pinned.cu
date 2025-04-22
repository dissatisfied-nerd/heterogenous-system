#include "memory.cuh"

std::vector<double> matrixMulPinned(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    if(A.size() != M*K || B.size() != K*N)
        throw std::invalid_argument("Invalid matrix dimensions");

    double *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaHostAlloc(&h_A, M*K*sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, K*N*sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C, M*N*sizeof(double), cudaHostAllocDefault));

    std::copy(A.begin(), A.end(), h_A);
    std::copy(B.begin(), B.end(), h_B);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    dim3 blocks((N + BLOCK_SIZE-1)/BLOCK_SIZE, (M + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(double), cudaMemcpyDeviceToHost));
    
    std::vector<double> result(h_C, h_C + M*N);
    
    cudaFreeHost(h_A); 
    cudaFreeHost(h_B); 
    cudaFreeHost(h_C);
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    
    return result;
}