#include "mandelbrot.cuh"

#include <cuda_runtime.h>

__device__ int mandelbrot(double real, double imag, int max_iter) {
    double z_real = 0, z_imag = 0;
    int iter = 0;
    while (z_real * z_real + z_imag * z_imag < 4.0 && iter < max_iter) {
        double temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2 * z_real * z_imag + imag;
        z_real = temp;
        ++iter;
    }
    return iter;
}

__global__ void mandelbrot_kernel(int* output, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double real = (x - width / 2.0) * 4.0 / width;
    double imag = (y - height / 2.0) * 4.0 / height;
    output[y * width + x] = mandelbrot(real, imag, max_iter);
}

void mandelbrot_cuda(int*& output_host, int width, int height, int max_iter) {
    int* output_dev;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&output_dev, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(output_dev, width, height, max_iter);

    output_host = new int[width * height];
    cudaMemcpy(output_host, output_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(output_dev);
}
