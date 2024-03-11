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

    double normaB = scalarMultiplication(b, b);
    double scalyAy;
    double scalAyAy;
    double tau;
    double normaY;
    vector<double> Axn(N);
    vector<double> y(N);
    vector<double> Ayn(N);
    vector<double> TauY(N);

#pragma omp parallel
    {
        while (true) {
#pragma omp single
#pragma omp atomic
            ++count;
            //Axn
#pragma omp for
            for (size_t i = 0; i < N; ++i) {
                double num = 0.0;
                for (size_t j = 0; j < N; ++j) {
                    num += A[i * N + j] * x[j];
                }
                Axn[i] = num;
            }
            //y
#pragma omp for
            for (size_t i = 0; i < N; ++i) {
                y[i] = (Axn[i] - b[i]);
            }
            //Ayn
#pragma omp for
            for (size_t i = 0; i < N; ++i) {
                double num = 0.0;
                for (size_t j = 0; j < N; ++j) {
                    num += A[i * N + j] * y[j];
                }
                Ayn[i] = num;
            }
#pragma omp single
            {
                scalyAy = scalarMultiplication(y, Ayn);
                scalAyAy = scalarMultiplication(Ayn, Ayn);
                if (scalAyAy == 0) scalAyAy = 1;
                tau = scalyAy / scalAyAy;
                normaY = scalarMultiplication(y, y);
            }

#pragma omp for
            for (size_t i = 0; i < N; ++i) {
                TauY[i] = (y[i] * tau);
            }

            if ((normaY / normaB) < (eps * eps)) break;
#pragma omp single
            x = differenceVectors(x, TauY, N);
        }

    } //end of omp

    auto endTime = omp_get_wtime();

    cout << count << " iterations" << endl;
    cout << "Passed: " << endTime - startTime << endl;

    return 0;
}
