#include "balancing.cuh"

void demonstrateTaskPartitioning() {
    const int numElements = 50000;
    const int threshold = 100;
    
    std::vector<float> h_A(numElements), h_B(numElements), h_C(numElements);
    std::vector<bool> isComplex(numElements, false);
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 99);
    for (int i = 0; i < numElements; ++i) {
        if (distribution(generator) < 10) {
            isComplex[i] = true;
        }
    }
    
    for (int i = 0; i < numElements; ++i) {
        if (!isComplex[i]) {
            h_C[i] = h_A[i] + h_B[i];
        }
    }
    
    int complexCount = std::count(isComplex.begin(), isComplex.end(), true);
    if (complexCount > 0) {
        std::vector<float> h_A_complex(complexCount), h_B_complex(complexCount), h_C_complex(complexCount);
        std::vector<int> complexIndices;
        
        for (int i = 0; i < numElements; ++i) {
            if (isComplex[i]) {
                complexIndices.push_back(i);
                h_A_complex.push_back(h_A[i]);
                h_B_complex.push_back(h_B[i]);
            }
        }
        
        size_t complexSize = complexCount * sizeof(float);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, complexSize);
        cudaMalloc(&d_B, complexSize);
        cudaMalloc(&d_C, complexSize);
        
        cudaMemcpy(d_A, h_A_complex.data(), complexSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_complex.data(), complexSize, cudaMemcpyHostToDevice);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (complexCount + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, complexCount);
        
        cudaMemcpy(h_C_complex.data(), d_C, complexSize, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < complexCount; ++i) {
            h_C[complexIndices[i]] = h_C_complex[i];
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    std::cout << "Task partitioning completed with " << complexCount << " complex tasks." << std::endl;
}