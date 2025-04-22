#include <gtest/gtest.h>
#include "gen.hpp"
#include "matrix.cuh"
#include <fstream>

TEST(MatrixTest, CorrectTest)
{
    const double epsilon = 1e-9;

    for (int size = 1; size < 11; ++size)
    {
        Matrix a = GetRandMatrix(size, size);
        Matrix b = GetRandMatrix(size, size);
        Matrix CUDARes;
        
        Matrix CPUSingleRes = CPUSingleMultiply(a, b);
        Matrix CPUMultiRes = CPUMultiMultiply(a, b);
        CUDAMultiply(a, b, CUDARes);
        
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                EXPECT_LE(fabs(CPUSingleRes[i][j] - CPUMultiRes[i][j]), epsilon);
                EXPECT_LE(fabs(CPUMultiRes[i][j] - CUDARes[i][j]), epsilon);
            }
        }
    }
}

TEST(MatrixTest, BenchmarkTest)
{
    const int size = 100;

    Matrix a = GetRandMatrix(size, size);
    Matrix b = GetRandMatrix(size, size);

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

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

    std::cout << CPUSingleDuration.count() << ' ' << CPUMultiDuration.count() << ' ' << CUDADuration.count() << '\n';

    EXPECT_GE(CPUSingleDuration.count(), CPUMultiDuration.count());
    EXPECT_GE(CPUMultiDuration.count(), CUDADuration.count());
}

TEST(MatrixTest, StressTest)
{
    std::ofstream fout("matrix.txt");

    if (!fout.is_open())
    {
        std::cerr << "File error" << '\n';
        exit(1);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    Matrix CUDAres;

    for (int i = 1; i < 1000 + 1; ++i)
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


        start = std::chrono::high_resolution_clock::now();
        float milliseconds = CUDAMultiplyProfile(a, b, CUDAres);
        end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> CUDADuration = end - start;

        fout << CPUSingleDuration.count() << ' ' 
             << CPUMultiDuration.count() << ' ' 
             << CUDADuration.count() << ' ' 
             << milliseconds << '\n';
    }

    fout.close();
}
