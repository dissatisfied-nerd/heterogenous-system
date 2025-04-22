#include <gtest/gtest.h>
#include "gen.hpp"
#include "memory.cuh"

TEST(MemoryTest, CorrectTest)
{
    const double epsilon = 1e-9;

    int m = 20;
    int k = 30;
    int n = 20;

    for (int i = 0; i < 10; ++i)
    {
        Matrix A = GetRandMatrix(n, k);
        Matrix B = GetRandMatrix(k, n);

        std::vector<double> flatA = flatten(A);
        std::vector<double> flatB = flatten(B);

        std::vector<double> resExplicit = matrixMulExplicit(flatA, flatB, m, k, n);
        std::vector<double> resUnified = matrixMulUnified(flatA, flatB, m, k, n);
        std::vector<double> resPinned = matrixMulPinned(flatA, flatB, m, k, n);
        std::vector<double> resZeroCopy = matrixMulZeroCopy(flatA, flatB, m, k, n);

        for (int i = 0; i < resExplicit.size(); ++i)
        {
            EXPECT_LE(fabs(resExplicit[i] - resUnified[i]), epsilon);
            EXPECT_LE(fabs(resExplicit[i] - resPinned[i]), epsilon);
            EXPECT_LE(fabs(resExplicit[i] - resZeroCopy[i]), epsilon);
        }
    }
}