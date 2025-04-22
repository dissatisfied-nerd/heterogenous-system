#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

const int N = 512;
#define BLOCK_SIZE 16

void profile(const char* label, void (*func)()) 
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    printf("[%s] Time: %.3f ms\n", label, time);
}

__global__ void matMulKernel(float* A, float* B, float* C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) 
    {
        float val = 0;
        
        for (int k = 0; k < N; ++k){
            val += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = val;
    }
}

void run_explicit_memory() 
{
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

void run_unified_memory() 
{
    size_t size = N * N * sizeof(float);
    float *A, *B, *C;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N * N; ++i) 
    { 
        A[i] = 1.0f; 
        B[i] = 2.0f; 
    }

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    matMulKernel<<<blocks, threads>>>(A, B, C, N);

    cudaDeviceSynchronize();
    cudaFree(A); cudaFree(B); cudaFree(C);
}

void run_zero_copy() 
{
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    cudaSetDeviceFlags(cudaDeviceMapHost);

    cudaHostAlloc(&h_A, size, cudaHostAllocMapped);
    cudaHostAlloc(&h_B, size, cudaHostAllocMapped);
    cudaHostAlloc(&h_C, size, cudaHostAllocMapped);

    cudaHostGetDevicePointer(&d_A, h_A, 0);
    cudaHostGetDevicePointer(&d_B, h_B, 0);
    cudaHostGetDevicePointer(&d_C, h_C, 0);
    
    for (int i = 0; i < N * N; ++i) 
    { 
        h_A[i] = 1.0f; 
        h_B[i] = 2.0f; 
    }
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
    
    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
}

void run_pinned_memory() 
{
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    cudaHostAlloc(&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, size, cudaHostAllocDefault);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; ++i) 
    { 
        h_A[i] = 1.0f; 
        h_B[i] = 2.0f; 
    }
    
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
    
    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
}

int main() {
    profile("Explicit Memory", run_explicit_memory);
    profile("Unified Memory", run_unified_memory);
    profile("Zero-Copy", run_zero_copy);
    profile("Pinned Memory", run_pinned_memory);
    return 0;
}
