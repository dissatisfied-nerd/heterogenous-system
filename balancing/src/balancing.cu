#include "balancing.cuh"

void demonstrateDynamicLoadBalancing() 
{
    const int numTasks = 100;
    const int numWorkers = 4;
    
    std::queue<int> taskQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<int> completedTasks(0);
    
    for (int i = 0; i < numTasks; ++i) {
        taskQueue.push(i);
    }
    
    auto worker = [&](int workerId) 
    {
        while (true) 
        {
            int task;
        
            {
                std::unique_lock<std::mutex> lock(queueMutex);
        
                if (taskQueue.empty()) {
                    return;
                }
        
                task = taskQueue.front();
                taskQueue.pop();
            }
            
            size_t size = (1000 + task % 5000) * sizeof(float);
            float *d_A, *d_B, *d_C;
        
            cudaMalloc(&d_A, size);
            cudaMalloc(&d_B, size);
            cudaMalloc(&d_C, size);
            
            int numElements = size / sizeof(float);
            int threadsPerBlock = 256;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        
            vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
            cudaDeviceSynchronize();
            
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            
            completedTasks++;
        }
    };
    
    std::vector<std::thread> workers;
    
    for (int i = 0; i < numWorkers; ++i) {
        workers.emplace_back(worker, i);
    }
    
    for (auto& t : workers) {
        t.join();
    }
    
    std::cout << "Dynamic load balancing completed. Tasks processed: " << completedTasks << std::endl;
}