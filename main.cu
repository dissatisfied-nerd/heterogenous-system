#include "matrix.cuh"
#include "mandelbrot.cuh"
#include "gen.hpp"
#include "debug.hpp"
#include "memory.cuh"

#include <iostream>
#include <fstream>

double epsilon = 1e-9;

void PrintVector(std::vector<double> v)
{
    for (auto &elem : v){
        std::cout << elem << ' ';
    }

    std::cout << '\n';
}

template <typename Func, typename... Args>
long long getExecutionTime(Func&& func, Args&&... args) 
{
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main()
{
/*
    std::ofstream fout("../log/transport.txt");

    if (!fout.is_open())
    {
        std::cerr << "File error" << '\n';
        exit(1);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    for (int i = 1; i < 1000 + 1; ++i)
    {
        Matrix a = GetRandMatrix(i, i);
        Matrix b = GetRandMatrix(i, i);
        Matrix CUDARes;

        float milliseconds = CUDAMultiplyProfile(a, b, CUDARes);

        fout << milliseconds << '\n';

        std::cout << '[' << i << ']' << ' ' << milliseconds << '\n';
    }

    fout.close();
*/

/*
    std::ofstream fout("../log/matrix.txt");

    if (!fout.is_open())
    {
        std::cerr << "File error" << '\n';
        exit(1);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    for (int i = 1; i < 500 + 1; ++i)
    {
        Matrix a = GetRandMatrix(i, i);
        Matrix b = GetRandMatrix(i, i);

        start = std::chrono::high_resolution_clock::now();
        CPUSingleMultiply(a, b);
        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> CPUSingleDuration = end - start;


        start = std::chrono::high_resolution_clock::now();
        CPUMultiMultiply(a, b);
        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> CPUMultiDuration = end - start;

        Matrix CUDAres;

        start = std::chrono::high_resolution_clock::now();
        CUDAMultiply(a, b, CUDAres);
        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> CUDADuration = end - start;

        fout << CPUSingleDuration.count() << ' ' 
             << CPUMultiDuration.count() << ' ' 
             << CUDADuration.count() << '\n'; 
        
        std::cout << '[' << i << ']' << ' '
                  << CPUSingleDuration.count() << ' ' 
                  << CPUMultiDuration.count() << ' ' 
                  << CUDADuration.count() << '\n'; 
    }

    fout.close();
*/

    return 0;
}