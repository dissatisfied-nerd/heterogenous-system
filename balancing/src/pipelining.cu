#include "balancing.cuh"

void demonstratePipelining() {
    const int numStages = 3;
    const int numElements = 10000;
    size_t size = numElements * sizeof(float);
    
    std::vector<std::thread> pipelineThreads;
    std::vector<float*> stageBuffers(numStages + 1);
    
    for (int i = 0; i <= numStages; ++i) {
        cudaMallocHost(&stageBuffers[i], size);
    }
    
    initializeVector(stageBuffers[0], numElements);
    
    for (int stage = 0; stage < numStages; ++stage) {
        pipelineThreads.emplace_back([stage, numElements, size, &stageBuffers]() {
            float *d_input, *d_output;
            cudaMalloc(&d_input, size);
            cudaMalloc(&d_output, size);
            
            cudaMemcpyAsync(d_input, stageBuffers[stage], size, cudaMemcpyHostToDevice);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
            vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_input, d_output, numElements);
            
            cudaMemcpyAsync(stageBuffers[stage+1], d_output, size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            
            cudaFree(d_input);
            cudaFree(d_output);
        });
    }
    
    for (auto& t : pipelineThreads) {
        t.join();
    }
    
    for (auto& buf : stageBuffers) {
        cudaFreeHost(buf);
    }
    
    std::cout << "Pipelining completed with " << numStages << " stages." << std::endl;
}