#include <vector>
#include <mpi.h>
#include <iostream>

constexpr auto eps = 0.00001;
constexpr auto N = 65000; // N = 1000

using namespace std;

void calculateMatrixVector(const vector<double> &partOfMatrix, const vector<double> &partOfX, const size_t countRows,
                           const int &startIndex, const size_t &endIndex, vector<double> &neededVector) {
    for (size_t i = 0; i < countRows; i++) {
        for (size_t j = startIndex; j < endIndex; j++) {
            neededVector[i] += partOfMatrix[i * N + j] * partOfX[j - startIndex];
        }
    }
}

void calculateMatrixVector(const vector<double> &partOfMatrix, const vector<double> &x, const size_t countRows,
                           vector<double> &neededVector) {
    for (size_t i = 0; i < countRows; i++) {
        for (size_t j = 0; j < x.size(); j++) {
            neededVector[i] += partOfMatrix[i * N + j] * x[j];
        }
    }
}

double scalarMultiplication(const vector<double> &first, const vector<double> &second) {
    double tmp = 0;
    for (size_t i = 0; i < first.size(); ++i) {
        tmp += first[i] * second[i];
    }
    return tmp;
}

void differenceVectors(const vector<double> &first, const vector<double> &second, vector<double> &neededVector) {
    for (size_t i = 0; i < neededVector.size(); i++) { //min(first.size(), second.size())
        neededVector[i] = first[i] - second[i];
    }
}

void multiplicationNumVector(const vector<double> &vect, const double num, vector<double> &neededVector) {
    for (size_t i = 0; i < vect.size(); ++i) {
        neededVector[i] = vect[i] * num;
    }
}

int main(int argc, char **argv) {
    int rank, size;

    int count = 0;

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
    for (size_t j = 0; j < size; j++) {
        int indexStartForMatrix = 0;
        for (int i = j; i > -1; --i) {
            if (i == j) continue;
            indexStartForMatrix += ((N / size) + ((N % size) > (i)));
        }
        countMarginsInMatrix[j] = indexStartForMatrix;
    }

    vector<int> counts(size);

    for (size_t i = 0; i < size; i++) {
        counts[i] = ((N / size) + ((N % size) > i));
    }

    int countRowsInA = counts[rank];
    vector<double> b(countRowsInA, N + 1);

    vector<double> partOfX(countRowsInA, 0);

    vector<double> partMatrix(N * countRowsInA, 1);

    int indexStartForMatrix = countMarginsInMatrix[rank];

    // Initializing the matrix
    int tmpIndex = indexStartForMatrix;
    for (size_t i = 0; i < countRowsInA; ++i) {
        partMatrix[i * N + tmpIndex++] = 2;
    }

    double normaB = scalarMultiplication(b, b);

    while (true) {

        vector<double> partOfAxn(countRowsInA);

        //calculate Axn with ring shift
        for (size_t i = 0; i < size; ++i) {
            int recvRank = (rank + i) % size;
            vector<double> tmpVectorForCalculating(counts[recvRank]);
            MPI_Request req[2];
            MPI_Status st[2];
            MPI_Isend(&partOfX[0], partOfX.size(), MPI_DOUBLE, (rank - i + size) % size, 12345, MPI_COMM_WORLD,
                      &req[0]);
            MPI_Irecv(&tmpVectorForCalculating[0], tmpVectorForCalculating.size(), MPI_DOUBLE, recvRank, 12345,
                      MPI_COMM_WORLD, &req[1]);
            MPI_Waitall(2, req, st);

            int startIndex = countMarginsInMatrix[recvRank];
            int endIndex;
            if (recvRank == size - 1) {
                endIndex = N;
            } else {
                endIndex = countMarginsInMatrix[recvRank + 1];
            }

            calculateMatrixVector(partMatrix, tmpVectorForCalculating, countRowsInA, startIndex, endIndex, partOfAxn);

        }

        vector<double> partOfY(countRowsInA);
        differenceVectors(partOfAxn, b, partOfY);

        vector<double> allY(N);

        MPI_Allgatherv(&partOfY[0], partOfY.size(), MPI_DOUBLE, &allY[0], &counts[0], &countMarginsInMatrix[0],
                       MPI_DOUBLE, MPI_COMM_WORLD);

        vector<double> partOfAyn(countRowsInA);
        calculateMatrixVector(partMatrix, allY, countRowsInA, partOfAyn);

        double partOfScalarYnAyn = scalarMultiplication(partOfY, partOfAyn);
        double partOfScalarAynAyn = scalarMultiplication(partOfAyn, partOfAyn);

        double allOfScalarYnAyn = 0.0;
        double allOfScalarAynAyn = 0.0;

        MPI_Allreduce(&partOfScalarYnAyn, &allOfScalarYnAyn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&partOfScalarAynAyn, &allOfScalarAynAyn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (allOfScalarAynAyn == 0) {
            allOfScalarAynAyn = 1;
        }

        double tauN = allOfScalarYnAyn / allOfScalarAynAyn;

        vector<double> partOfTauYn(countRowsInA);
        multiplicationNumVector(partOfY, tauN, partOfTauYn);

        vector<double> tmpX(countRowsInA);
        differenceVectors(partOfX, partOfTauYn, tmpX);

        partOfX = tmpX;

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
    if (rank == 0) {
        cout << endTime - startTime << " passed" << endl;
        cout << count << " iterations" << endl;
    }
    return 0;
}
