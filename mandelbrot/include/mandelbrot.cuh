#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#include <vector>

void mandelbrot_single_thread(std::vector<int>& output, int width, int height, int max_iter);
void mandelbrot_multithread(std::vector<int>& output, int width, int height, int max_iter, int thread_count);
void mandelbrot_cuda(int*& output_host, int width, int height, int max_iter);
void mandelbrot_cuda_uvm(int*& output, int width, int height, int max_iter);

#endif