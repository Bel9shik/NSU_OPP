#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;
constexpr int MAIN_PROCESS = 0;

void fillMatrix(double* matrix, int n1, int n2) {
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            matrix[i * n2 + j] = 1;
        }
    }
}

void multMatrix(double* A, double* B, double* C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                C[i * n3 + k] += A[i * n2 + j] * B[j * n3 + k];
            }
        }
    }
}

bool checkCorrect(double* A, double* B, double* C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int k = 0; k < n3; ++k) {
            double currSum = 0;
            for (int j = 0; j < n2; ++j) {
                currSum += A[i * n2 + j] * B[j * n3 + k];
            }
            if (C[i * n3 + k] != currSum) {
                return false;
            }
        }
    }
    return true;
}

bool isNumber(const std::string& str) {
    try {
        std::stod(str);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cout << "Enter ./main n1 n2 n3" << std::endl;
        return 0;
    }


    if (isNumber(argv[1]) && isNumber(argv[2]) && isNumber(argv[3])) {
    } else {
        std::cout << "enter only nums" << std::endl;
    }
    const int n1 = atoi(argv[1]);
    const int n2 = atoi(argv[2]);
    const int n3 = atoi(argv[3]);

    if (argv[1])

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int dimN = 2;
    const int coordX = 0;
    const int coordY = 1;

    int dims[dimN];
    fill(dims, dims + dimN, 0);

    int periods[dimN];
    fill(periods, periods + dimN, 0);

    int coords[dimN];

    MPI_Comm gridComm;
    MPI_Comm columnsComm;
    MPI_Comm rowsComm;

    MPI_Dims_create(size, dimN, dims);
    MPI_Cart_create(MPI_COMM_WORLD, dimN, dims, periods, 1, &gridComm); //
    MPI_Cart_coords(gridComm, rank, dimN, coords);

    if (n1 % dims[coordX] != 0 || n3 % dims[coordY] != 0) {
        if (rank == 0) {
            std::cout << "error" << std::endl;
        }
        return 0;
    }

    MPI_Comm_split(gridComm, coords[coordY], coords[coordX], &columnsComm);
    MPI_Comm_split(gridComm, coords[coordX], coords[coordY], &rowsComm);

    auto* A = (rank == MAIN_PROCESS) ? new double[n1 * n2] : nullptr;
    auto* B = (rank == MAIN_PROCESS) ? new double[n2 * n3] : nullptr;
    auto* C = (rank == MAIN_PROCESS) ? new double[n1 * n3] : nullptr;

    double startTime = 0;

    if (rank == MAIN_PROCESS) {

        srand(time(NULL));

        fillMatrix(A, n1, n2);
        fillMatrix(B, n2, n3);
        startTime = MPI_Wtime();
    }


    int rowPartNum = n1 / dims[coordX];
    int colPartNum = n3 / dims[coordY];

    auto* partA = new double[rowPartNum * n2];
    auto* partB = new double[colPartNum * n2];
    auto* partC = new double[rowPartNum * colPartNum];

    fill(partC, partC + rowPartNum * colPartNum, 0);

    if (coords[coordY] == 0) {
        MPI_Scatter(A, rowPartNum * n2, MPI_DOUBLE, partA, rowPartNum * n2, MPI_DOUBLE, MAIN_PROCESS, columnsComm);
    }

    if (coords[coordX] == 0) {
        MPI_Datatype sendB;
        MPI_Type_vector(n2, colPartNum, n3, MPI_DOUBLE, &sendB);
        MPI_Type_commit(&sendB);

        MPI_Datatype sendBResized;
        MPI_Type_create_resized(sendB, 0, colPartNum * sizeof(double), &sendBResized);
        MPI_Type_commit(&sendBResized);

        MPI_Scatter(B, 1, sendBResized, partB, n2 * colPartNum, MPI_DOUBLE, MAIN_PROCESS, rowsComm);

        MPI_Type_free(&sendB);
        MPI_Type_free(&sendBResized);
    }

    MPI_Bcast(partA, rowPartNum * n2, MPI_DOUBLE, MAIN_PROCESS, rowsComm);
    MPI_Bcast(partB, colPartNum * n2, MPI_DOUBLE, MAIN_PROCESS, columnsComm);

    multMatrix(partA, partB, partC, rowPartNum, n2, colPartNum);

    MPI_Datatype sendC;
    MPI_Type_vector(rowPartNum, colPartNum, n3, MPI_DOUBLE, &sendC);
    MPI_Type_commit(&sendC);

    MPI_Datatype sendCResized;
    MPI_Type_create_resized(sendC, 0, colPartNum * sizeof(double), &sendCResized);
    MPI_Type_commit(&sendCResized);

    int recvCounts[size];
    int displs[size];

    for (int i = 0; i < dims[coordX]; ++i) {
        for (int j = 0; j < dims[coordY]; ++j) {
            recvCounts[i * dims[coordY] + j] = 1;
            displs[i * dims[coordY] + j] = j + i * rowPartNum * dims[coordY];
        }
    }

    MPI_Gatherv(partC, colPartNum * rowPartNum, MPI_DOUBLE, C, recvCounts, displs, sendCResized, MAIN_PROCESS, MPI_COMM_WORLD);

    MPI_Type_free(&sendC);
    MPI_Type_free(&sendCResized);

    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&rowsComm);
    MPI_Comm_free(&columnsComm);

    double  endTime = 0;
    if (rank == 0) {
        endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        if (!checkCorrect(A, B, C, n1, n2, n3)) {
            std::cout << "error" << std::endl;
        } else {
            std::cout << "elapsedTime: " << elapsedTime << " sec" << std::endl;
        }
    }

    delete[] partA;
    delete[] partB;
    delete[] partC;

    if (rank == MAIN_PROCESS) {
        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();
    return 0;
}
