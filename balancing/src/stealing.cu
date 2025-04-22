#include "balancing.cuh"

void demonstrateWorkStealing() 
{
    const int numWorkers = 4;
    const int initialTasksPerWorker = 10;
    const int totalTasks = numWorkers * initialTasksPerWorker;
    
    std::vector<std::queue<int>> workerQueues(numWorkers);
    std::mutex mutex;
    std::atomic<int> completedTasks(0);
    
    for (int i = 0; i < numWorkers; ++i) {
        for (int j = 0; j < initialTasksPerWorker; ++j) {
            workerQueues[i].push(i * initialTasksPerWorker + j);
        }
    }
    
    auto stealTask = [&](int thiefWorkerId) -> bool 
    {
        std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
        
        if (!lock.owns_lock()) {
            return false;
        }
        
        for (int i = 0; i < numWorkers; ++i) 
        {
            if (i != thiefWorkerId && !workerQueues[i].empty()) 
            {
                int task = workerQueues[i].front();
                workerQueues[i].pop();
                workerQueues[thiefWorkerId].push(task);
            
                return true;
            }
        }
        
        return false;
    };
    
    auto worker = [&](int workerId) 
    {
        while (completedTasks < totalTasks) 
        {
            if (!workerQueues[workerId].empty()) 
            {
                int task;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (workerQueues[workerId].empty()) {
                        continue;
                    }
            
                    task = workerQueues[workerId].front();
                    workerQueues[workerId].pop();
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
            else 
            {
                if (!stealTask(workerId)) {
                    std::this_thread::yield();
                }
            }
        }
    };
    
    std::vector<std::thread> workers;
    for (int i = 0; i < numWorkers; ++i) {
        workers.emplace_back(worker, i);
    }
    
    for (auto& t : workers) {
        t.join();
    }
    
    std::cout << "Work stealing completed. Total tasks processed: " << completedTasks << std::endl;
}