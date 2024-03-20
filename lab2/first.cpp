#include <iostream>
#include <vector>
#include <omp.h>

constexpr auto eps = 0.00001;
using namespace std;

double scalarMultiplication(const vector<double> &first, const vector<double> &second) {
    double tmp = 0.0;
    for (size_t i = 0; i < first.size(); ++i) {
        tmp += first[i] * second[i];
    }
    return tmp;
}

vector<double> differenceVectors(const vector<double> &first, const vector<double> &second, size_t size) {
    vector<double> tmp(size);
    for (size_t i = 0; i < size; ++i) {
        tmp[i] = (first[i] - second[i]);
    }
    return tmp;
}

int main() {

    auto startTime = omp_get_wtime();

    size_t N = 75000;//75000
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

    double normaB = scalarMultiplication(b, b);
    double scalyAy;
    double scalAyAy;
    double tau;
    double normaY = 0.0;
    vector<double> Axn(N);
    vector<double> y(N);
    vector<double> Ayn(N);
    vector<double> TauY(N);

    while (true) {
        ++count;
        //Axn
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < N; ++i) {
            double num = 0.0;
            for (size_t j = 0; j < N; ++j) {
                num += A[i * N + j] * x[j];
            }
            Axn[i] = num;
        }
        //y
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < N; ++i) {
            y[i] = (Axn[i] - b[i]);
        }
        //Ayn
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < N; ++i) {
            double num = 0.0;
            for (size_t j = 0; j < N; ++j) {
                num += A[i * N + j] * y[j];
            }
            Ayn[i] = num;
        }

        scalAyAy = 0.0;
        scalyAy = 0.0;
        normaY = 0.0;
        //yAy
#pragma omp parallel for reduction(+:scalyAy) schedule(guided)
        for (size_t i = 0; i < y.size(); ++i) {
            scalyAy += y[i] * Ayn[i];
        }
        //AyAy
#pragma omp parallel for reduction(+:scalAyAy) schedule(guided)
        for (size_t i = 0; i < Ayn.size(); ++i) {
            scalAyAy += Ayn[i] * Ayn[i];
        }
        //normaY
#pragma omp parallel for reduction(+:normaY) schedule(guided)
        for (size_t i = 0; i < y.size(); ++i) {
            normaY += y[i] * y[i];
        }
        if (scalAyAy == 0) scalAyAy = 1;
        tau = scalyAy / scalAyAy;

#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < N; ++i) {
            TauY[i] = (y[i] * tau);
        }

        if ((normaY / normaB) < (eps * eps)) break;
        x = differenceVectors(x, TauY, N);
    }

    auto endTime = omp_get_wtime();

    cout << count << " iterations" << endl;
    cout << "Passed: " << endTime - startTime <<
         endl;

    return 0;
}
