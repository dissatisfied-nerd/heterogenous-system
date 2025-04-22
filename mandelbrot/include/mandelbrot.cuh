#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#include <vector>

void CPUSingleMandelbrot(std::vector<int>& output, int width, int height, int max_iter);
void CPUMultiMandelbrot(std::vector<int>& output, int width, int height, int max_iter, int thread_count);
void CUDAMandelbrot(int*& output_host, int width, int height, int max_iter);

#endif