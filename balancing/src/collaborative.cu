#include "balancing.cuh"

void demonstrateCollaborativeProcessing() 
{
    const int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);
    
    initializeVector(h_A, numElements);
    initializeVector(h_B, numElements);
    
    // CPU часть
    for (int i = 0; i < numElements; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    
    // GPU часть
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    
    bool match = true;
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            match = false;
            break;
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    std::cout << "Collaborative processing: CPU and GPU results " << (match ? "match!" : "differ!") << std::endl;
}