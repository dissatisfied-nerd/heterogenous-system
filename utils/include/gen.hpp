#ifndef GEN_HPP
#define GEN_HPP

#include <random>

using Matrix = std::vector<std::vector<double>>;

double GetRandDouble(double minNum, double maxNum);
std::vector<double> GetRandVector(int size);
Matrix GetRandMatrix(int n, int k);
Matrix GetRandSparseMatrix(int rows, int cols);

#endif