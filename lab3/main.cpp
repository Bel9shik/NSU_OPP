#include <iostream>
#include <chrono>

void printMatrix(const double *matrix, const int rowsCount, const int columnsCount) {
    for (int i = 0; i < rowsCount; ++i) {
        for (int j = 0; j < columnsCount; ++j) {
            std::cout << matrix[i * columnsCount + j] << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printElapsedTime(const double elapsedTime) {
    std::cout << "elapsedTime: " << elapsedTime << " sec" << std::endl;
}

void printInfoAboutMatrix(const std::string matrixName, const int rowsCount, const int columnsCount,
                          const double *matrix) {
    std::cout << std::endl << matrixName <<
              std::endl << "rows: " << rowsCount <<
              std::endl << "columns: " << columnsCount << std::endl;
    printMatrix(matrix, rowsCount, columnsCount);
}


int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cout << "Bad input! Enter matrix size" << std::endl;
    }

    const int n1 = atoi(argv[1]);
    const int n2 = atoi(argv[2]);
    const int n3 = atoi(argv[3]);

    // создание матриц
    double *matrixA = new double[n1 * n2];
    double *matrixB = new double[n2 * n3];
    double *matrixC = new double[n1 * n3];

    // заполнение матрицы А
    srand(time(NULL));
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            matrixA[i * n2 + j] = i + 1;
        }
    }

    // заполнение матрицы B
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n3; ++j) {
            matrixB[i * n3 + j] = i + 1;
        }
    }

//    printMatrix(matrixA, n1, n2);
//    printMatrix(matrixB, n2, n3);

    std::fill(matrixC, matrixC + n1 * n3, 0);

    auto startTime = clock();

    // Вычисление матрицы С
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n3; j++) {
            for (int k = 0; k < n2; k++) {
                matrixC[i * n3 + j] += matrixA[i * n2 + k] * matrixB[k * n3 + j];
            }
        }
    }
    auto endTime = clock();
    auto elapsedTime = (double) (endTime - startTime) / CLOCKS_PER_SEC;

//    printMatrix(matrixC, n1, n3);
    printElapsedTime(elapsedTime);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
