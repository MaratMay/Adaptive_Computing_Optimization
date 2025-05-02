#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>

using namespace std;

void matrixVectorMultiply(const vector<vector<double>>& matrix, const vector<double>& vec) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<double> result(rows, 0.0);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result[i] += matrix[i][j] * vec[j];
}

int main() {
    const size_t size = 1000; // Размер матрицы и вектора
    const int iterations = 100; // Количество итераций умножения

    vector<vector<double>> matrix(size, vector<double>(size));
    vector<double> vec(size);

    srand(42);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j)
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        vec[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    #pragma omp parallel for // Потоки создаются один раз, итерации распределяются
    for (int i = 0; i < iterations; ++i)
        matrixVectorMultiply(matrix, vec);

    return 0;
}
