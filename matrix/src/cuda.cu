#include "matrix.cuh"

__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

void CUDAMultiply(const Matrix& A, const Matrix& B, Matrix& res)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Несовместимые размеры матриц для умножения.");
    }

    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = rowsB * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);

    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    std::vector<double> flatA(rowsA * colsA);
    std::vector<double> flatB(rowsB * colsB);
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            flatA[i * colsA + j] = A[i][j];
    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            flatB[i * colsB + j] = B[i][j];

    cudaMemcpy(d_A, flatA.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    std::vector<double> flatC(rowsA * colsB);
    cudaMemcpy(flatC.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    res.resize(rowsA, std::vector<double>(colsB));
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j)
            res[i][j] = flatC[i * colsB + j];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

float CUDAMultiplyProfile(const Matrix& A, const Matrix& B, Matrix& res)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Несовместимые размеры матриц для умножения.");
    }

    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = rowsB * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);

    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    std::vector<double> flatA(rowsA * colsA);
    std::vector<double> flatB(rowsB * colsB);
    
    for (int i = 0; i < rowsA; ++i){
        for (int j = 0; j < colsA; ++j){
            flatA[i * colsA + j] = A[i][j];
        }
    }

    for (int i = 0; i < rowsB; ++i){
        for (int j = 0; j < colsB; ++j){
            flatB[i * colsB + j] = B[i][j];
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_A, flatA.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), sizeB, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    std::vector<double> flatC(rowsA * colsB);

    cudaEventRecord(start1);
    cudaMemcpy(flatC.data(), d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1);

    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);

    res.resize(rowsA, std::vector<double>(colsB));
    
    for (int i = 0; i < rowsA; ++i){
        for (int j = 0; j < colsB; ++j){
            res[i][j] = flatC[i * colsB + j];
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return milliseconds + milliseconds1;
}

