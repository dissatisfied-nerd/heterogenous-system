#include "mandelbrot.cuh"

#include <complex>
#include <vector>

void mandelbrot_single_thread(std::vector<int>& output, int width, int height, int max_iter) {
    output.resize(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::complex<double> c((x - width / 2.0) * 4.0 / width,
                                   (y - height / 2.0) * 4.0 / height);
            std::complex<double> z = 0;
            int iter = 0;
            while (std::abs(z) < 2.0 && iter < max_iter) {
                z = z * z + c;
                ++iter;
            }
            output[y * width + x] = iter;
        }
    }
}
