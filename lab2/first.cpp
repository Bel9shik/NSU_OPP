#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

constexpr auto eps = 0.00001;
using namespace std;

vector<double> calculateMatrixVector(const vector<double> &A, const vector<double> &x, size_t size) { //size_t
    vector<double> tmp(size, 0);
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        double num = 0.0;
        for (size_t j = 0; j < size; ++j) {
            num += A[i * size + j] * x[j];
        }
        tmp[i] = num;
    }
    return tmp;
}

double scalarMultiplication(vector<double> &first, vector<double> &second, size_t size) {
    double tmp = 0.0;
#pragma omp parallel for reduction(+:tmp)
    for (size_t i = 0; i < size; ++i) {
        tmp += first[i] * second[i];
    }
    return tmp;
}

vector<double> differenceVectors(vector<double> &first, vector<double> &second, size_t size) {
    vector<double> tmp(size);
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        tmp[i] = (first[i] - second[i]);
    }
    return tmp;
}

vector<double> multiplicationNumVector(vector<double> &vect, double num, size_t size) {
    vector<double> tmp(size);
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        tmp[i] = (vect[i] * num);
    }
    return tmp;
}

double criteriaEndOfCalculating(vector<double> &Ax, vector<double> &b, size_t size) {
    vector<double> tmp = differenceVectors(Ax, b, size);
    double crit = sqrt(scalarMultiplication(tmp, tmp, size)) / sqrt(scalarMultiplication(b, b, size));
    return crit;
}

int main(int argc, char *argv[]) {

    auto startTime = omp_get_wtime();

    size_t N = 5000;//65000
    int count = 0;

    vector<double> A(N * N, 1);

    //initializing the matrix
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            //on diagonal
            if (i == j) {
                A[i * N + j] = 2; //rand() % 3
                break;
            }
        }
    }

    vector<double> x(N);

    vector<double> b(N, N + 1);
    while (true) {
        ++count;
        vector<double> tmpAX = calculateMatrixVector(A, x, N);

        vector<double> y = differenceVectors(tmpAX, b, N);

        vector<double> tmpAY = calculateMatrixVector(A, y, N);
        double scalyAy = scalarMultiplication(y, tmpAY, N);
        double scalAyAy = scalarMultiplication(tmpAY, tmpAY, N);
        if (scalAyAy == 0) scalAyAy = 1;
        double tau = scalyAy / scalAyAy;

        vector<double> tmpTauY = multiplicationNumVector(y, tau, N);

        x = differenceVectors(x, tmpTauY, N);

        double crit = criteriaEndOfCalculating(tmpAX, b, N);

        if (crit < eps) break;
    }

    auto endTime = omp_get_wtime();

    cout << count << " iterations" << endl;
    cout << "Passed: " << endTime - startTime << endl;

    return 0;
}

