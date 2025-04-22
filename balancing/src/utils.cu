#include "balancing.cuh"

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int numElements) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void initializeVector(float* vec, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
}