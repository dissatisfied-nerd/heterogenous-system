#include "memory.cuh"

std::vector<double> flatten(const std::vector<std::vector<double>>& matrix) 
{
    if (matrix.empty()) {
        return {};
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    std::vector<double> result(rows * cols);
    
    for (int i = 0; i < rows; ++i) 
    {
        if (matrix[i].size() != cols){
            throw std::invalid_argument("All rows must have the same length");
        }

        std::copy(matrix[i].begin(), matrix[i].end(), result.begin() + i*cols);
    }
    
    return result;
}

std::vector<std::vector<double>> unflatten(const std::vector<double>& flatMatrix, int rows, int cols) 
{
    if(flatMatrix.size() != rows*cols){
        throw std::invalid_argument("Invalid dimensions for unflatten");
    }

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    
    for(int i = 0; i < rows; ++i){
        std::copy(flatMatrix.begin() + i*cols, flatMatrix.begin() + (i+1)*cols, result[i].begin());
    }

    return result;
}