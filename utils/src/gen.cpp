#include "gen.hpp"
#include <iostream>

int GetRandInt(int minNum, int maxNum)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dist(minNum, maxNum);

    return dist(gen);
}

double GetRandDouble(double minNum, double maxNum)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(minNum, maxNum);

    return dist(gen);
}

std::vector<double> GetRandVector(int size)
{
    std::vector<double> res(size);

    for (int i = 0; i < size; ++i){
        res[i] = GetRandDouble(-10, 10);
    }

    return res;
}

Matrix GetRandMatrix(int n, int k)
{
    std::vector<std::vector<double>> res(n, std::vector<double>(k));

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < k; ++j){
            res[i][j] = GetRandDouble(-10, 10);
        }
    }

    return res;
}

Matrix GetRandSparseMatrix(int rows, int cols)
{
    Matrix res(rows, std::vector<double>(cols, 0));
    int meanElements = static_cast<int>(rows * cols * 0.3);

    for (int i = 0; i < meanElements; ++i)
    {
        int row, col;

        do 
        {
            row = GetRandInt(0, rows - 1);
            col = GetRandInt(0, cols - 1);
        }
        while (res[row][col] != 0);

        res[row][col] = GetRandDouble(-10, 10);
    }

    return res;
}

