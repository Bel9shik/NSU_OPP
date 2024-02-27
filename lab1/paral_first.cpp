#include <vector>
#include <mpi.h>
#include <iostream>

constexpr auto eps = 0.00001;
constexpr auto N = 5000; // N = 1000

using namespace std;

void calculateMatrixVector(const vector<double> &partOfMatrix, const vector<double> &x, const int countRows, vector<double> &neededVector) {
    for (int i = 0; i < countRows; i++) {
        for (int j = 0; j < x.size(); j++) {
            neededVector[i] += partOfMatrix[i * N + j] * x[j];
        }
    }
}

double scalarMultiplication(const vector<double> &first, const vector<double> &second) {
    double tmp = 0;
    for (int i = 0; i < first.size(); ++i) {
        tmp += first[i] * second[i];
    }
    return tmp;
}

void differenceVectors(const vector<double> &first,const vector<double> &second, vector<double> &neededVector) {
    for (int i = 0; i < neededVector.size(); i++) { //min(first.size(), second.size())
        neededVector[i] = first[i] - second[i];
    }
}

void multiplicationNumVector(const vector<double> &vect,const double num, vector<double> &neededVector) {
    for (int i = 0; i < vect.size(); ++i) {
        neededVector[i] = vect[i] * num;
    }
}

int main(int argc, char **argv) {
    int rank, size;

    int count = 0;

    vector<double> b(N, N + 1);
    vector<double> x(N, 0);

    int errCode;

    if ((errCode = MPI_Init(&argc, &argv)) != 0) {
        return errCode;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto startTime = MPI_Wtime();

    if (size > N) {
        cout << "so much process" << endl;
        return 0;
    }

    vector<int> countMarginsInMatrix(size);
    for (int j = 0; j < size; j++) {
        int indexStartForMatrix = 0;
        for (int i = j; i > -1; --i) {
            if (i == j) continue;
            indexStartForMatrix += ((N / size) + ((N % size) > (i)));
        }
        countMarginsInMatrix[j] = indexStartForMatrix;
    }

    vector<int> counts(size);

    for (int i = 0; i < size; i++) {
        counts[i] = ((N / size) + ((N % size) > i));
    }

    int countRowsInA = counts[rank];

    vector<double> partMatrix(N * countRowsInA, 1);

    int indexStartForMatrix = countMarginsInMatrix[rank];

    // Initializing the matrix
    int tmpIndex = indexStartForMatrix;
    for (int i = 0; i < countRowsInA; ++i) {
        partMatrix[i * N + tmpIndex++] = 2;
    }

    double normaB = scalarMultiplication(b, b);

    while (true) {

        vector<double> partOfAxn(countRowsInA);
        calculateMatrixVector(partMatrix, x, countRowsInA, partOfAxn);

        vector<double> partOfY(countRowsInA);
        differenceVectors(partOfAxn, b, partOfY);

        vector<double> allY (N);

        MPI_Allgatherv(&partOfY[0], partOfY.size(), MPI_DOUBLE, &allY[0], &counts[0], &countMarginsInMatrix[0], MPI_DOUBLE, MPI_COMM_WORLD);

        vector<double> partOfAyn(countRowsInA);
        calculateMatrixVector(partMatrix, allY, countRowsInA, partOfAyn);

        double partOfScalarYnAyn = scalarMultiplication(partOfY, partOfAyn);
        double partOfScalarAynAyn = scalarMultiplication(partOfAyn, partOfAyn);

        double  allOfScalarYnAyn = 0.0;
        double  allOfScalarAynAyn = 0.0;

        MPI_Allreduce(&partOfScalarYnAyn, &allOfScalarYnAyn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&partOfScalarAynAyn, &allOfScalarAynAyn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (allOfScalarAynAyn == 0) {
            allOfScalarAynAyn = 1;
        }

        double tauN = allOfScalarYnAyn / allOfScalarAynAyn;

        vector<double> partOfTauYn(countRowsInA);
        multiplicationNumVector(partOfY, tauN, partOfTauYn);

        vector<double> tmpX(countRowsInA);
        differenceVectors(x, partOfTauYn, tmpX); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        MPI_Allgatherv(&tmpX[0], tmpX.size(), MPI_DOUBLE, &x[0], &counts[0], &countMarginsInMatrix[0], MPI_DOUBLE, MPI_COMM_WORLD);

        double partOfNormaYn = scalarMultiplication(partOfY, partOfY);
        double allOfNormaYn = 0.0;

        MPI_Allreduce(&partOfNormaYn, &allOfNormaYn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if ((allOfNormaYn / normaB) < (eps * eps)) {
            break;
        }

        ++count;

    }
    auto endTime = MPI_Wtime();

    MPI_Finalize();
    if (rank == 0){
        cout << count << " iterations" << endl;
        cout << "time passed: " << endTime - startTime << endl;
    }

    return 0;
}
