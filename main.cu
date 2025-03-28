#include "matrix.cuh"
#include "mandelbrot.cuh"

#include <iostream>
#include <fstream>

void PrintMatrix(Matrix& result)
{
    for (const auto& row : result) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void save_to_ppm(const std::vector<int>& data, int width, int height, int max_iter, const std::string& filename) 
{
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; ++i) {
        int iter = data[i];
        unsigned char color = static_cast<unsigned char>(255 * iter / max_iter);
        ofs << color << color << color;
    }
}

void save_to_ppm(const int* data, int width, int height, int max_iter, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; ++i) {
        int iter = data[i];
        unsigned char color = static_cast<unsigned char>(255 * iter / max_iter);
        ofs << color << color << color;
    }
}

int main()
{
    Matrix A = {
        {1, 2},
        {4, 5}
    };

    Matrix B = {
        {7, 8},
        {9, 10}
    };

    Matrix res1, res2, res3, res4;

    res1 = CPUMultiply(A, B);
    res2 = ThreadMultiply(A, B);
    CUDAMultiply(A, B, res3);
    res4 = UMAMultiply(A, B);
  
    PrintMatrix(res1);
    PrintMatrix(res2);
    PrintMatrix(res3);
    PrintMatrix(res4);

    int width = 1920, height = 1080, max_iter = 100000;

    // Однопоточно
    //std::vector<int> output1;
    //mandelbrot_single_thread(output1, width, height, max_iter);
    //save_to_ppm(output1, width, height, max_iter, "cpu.ppm");

    // Многопоточно
    //std::vector<int> output2;
    //mandelbrot_multithread(output2, width, height, max_iter, 8);
    //save_to_ppm(output2, width, height, max_iter, "thread.ppm");

    // CUDA
    int* output_cuda = nullptr;
    mandelbrot_cuda(output_cuda, width, height, max_iter);
    save_to_ppm(output_cuda, width, height, max_iter, "cuda.ppm");
    delete[] output_cuda;

    // CUDA + UVM
    //int* output_uvm = nullptr;
    //mandelbrot_cuda_uvm(output_uvm, width, height, max_iter);
    //save_to_ppm(output_uvm, width, height, max_iter, "uma.ppm");
    //cudaFree(output_uvm);

    return 0;
}