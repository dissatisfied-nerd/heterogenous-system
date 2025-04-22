#ifndef BALANCING_CUH
#define BALANCING_CUH   

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>

void demonstrateOffloading();
void demonstrateCollaborativeProcessing();
void demonstratePipelining();
void demonstrateTaskPartitioning();
void demonstrateDynamicLoadBalancing();
void demonstrateWorkStealing();

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int numElements);
void initializeVector(float* vec, int size);

#endif