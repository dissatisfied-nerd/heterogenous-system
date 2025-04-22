#include "mandelbrot.cuh"

#include <complex>
#include <vector>
#include <thread>

void mandelbrot_thread_worker(std::vector<int>& output, int width, int height, int max_iter, int y_start, int y_end) 
{
    for (int y = y_start; y < y_end; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            std::complex<double> c((x - width / 2.0) * 4.0 / width, (y - height / 2.0) * 4.0 / height);
            std::complex<double> z = 0;
            int iter = 0;
            
            while (std::abs(z) < 2.0 && iter < max_iter) 
            {
                z = z * z + c;
                ++iter;
            }
            
            output[y * width + x] = iter;
        }
    }
}

void CPUMultiMandelbrot(std::vector<int>& output, int width, int height, int max_iter, int thread_count) 
{
    output.resize(width * height);
    std::vector<std::thread> threads;
    int rows_per_thread = height / thread_count;

    for (int i = 0; i < thread_count; ++i) 
    {
        int y_start = i * rows_per_thread;
        int y_end = (i == thread_count - 1) ? height : y_start + rows_per_thread;
        threads.emplace_back(mandelbrot_thread_worker, std::ref(output), width, height, max_iter, y_start, y_end);
    }

    for (auto& t : threads){
        t.join();
    }
}
