#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <vector>
#include <iostream>

template <typename T>
void PrintMatrix(const std::vector<std::vector<T>> &v)
{
    for (int i = 0; i < v.size(); ++i)
    {
        for (int j = 0; j < v[0].size(); ++j){
            std::cout << v[i][j] << ' ';
        }

        std::cout << '\n';
    }
}

#endif