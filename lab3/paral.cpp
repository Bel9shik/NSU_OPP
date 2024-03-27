#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <mpi.h>

using namespace std;

bool isMainProcess(const int processRank) {
    return processRank == 0;
}

void printMatrix(const double *matrix, const int rowsCount, const int columnsCount) {
    for (int i = 0; i < rowsCount; i++) {
        for (int j = 0; j < columnsCount; j++) {
            cout << matrix[i * columnsCount + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printElapsedTime(const double elapsedTime) {
    cout << "elapsedTime: " << elapsedTime << " sec" << endl;
}

bool isProcessInFirstColumnInCartTopology(const int *coords, const int coordYIndex) {
    return coords[coordYIndex] == 0;
}

bool isProcessInFirstRowInCartTopology(const int *coords, const int coordXIndex) {
    return coords[coordXIndex] == 0;
}

void fillMatrix(double *matrix, int rowsCount, int columnCount) {
    for (int i = 0; i < rowsCount; ++i) {
        for (int j = 0; j < columnCount; ++j) {
            matrix[i * columnCount + j] = i + 1;
        }
    }
}

bool isCorrectCalculation(const double *matrixC, int rowsCount, int columnsCount,
                          int rowsCountMatrixB) {
    int sum = 0;
    int sumBase = ((1 + rowsCountMatrixB) * rowsCountMatrixB) / 2;
    for (int i = 0; i < rowsCount; ++i) {
        sum += sumBase;
//        std::cout << sum << std:: endl;
        for (int j = 0; j < columnsCount; ++j) {
            if (matrixC[i * columnsCount + j] != sum) {
                return false;
            }
        }
    }
    std::cout << std::endl;
    return true;
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cout << "Bad input! Enter matrix size" << std::endl;
    }

    const int n1 = atoi(argv[1]);
    const int n2 = atoi(argv[2]);
    const int n3 = atoi(argv[3]);

    MPI_Init(&argc, &argv);

    int totalProcessesNumber;           // число процессов в области связи
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcessesNumber);

    int currentProcessRank;             // номер текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcessRank);

    const int dimensionsNumber = 2;     // количество измерений (2D)
    const int coordXIndex = 0;
    const int coordYIndex = 1;

    int dims[dimensionsNumber];         // массив, содержащий количество процессов в каждом измерении
    fill(dims, dims + dimensionsNumber, 0);

    int periods[dimensionsNumber];      // периодические граничные условия, логический массив
    fill(periods, periods + dimensionsNumber, 0);

    int coords[dimensionsNumber];       // координаты текущего процесса

    MPI_Comm gridComm;                  // новые коммуникаторы
    MPI_Comm columnsComm;
    MPI_Comm rowsComm;

    // MPI_Dims_create создает (выгодное) разделение процессов в декартовой сетке
    MPI_Dims_create(totalProcessesNumber, dimensionsNumber, dims);

    // MPI_Cart_create создаёт новый коммуникатор с заданной декартовой топологией
    MPI_Cart_create(MPI_COMM_WORLD, dimensionsNumber, dims, periods, 1, &gridComm);

    // MPI_Cart_coords определяет координаты процесса по его номеру в данной области
    MPI_Cart_coords(gridComm, currentProcessRank, dimensionsNumber, coords);

    // MPI_Comm_split разделит коммуникатор gridComm на непересекающиеся субкоммуникаторы
    MPI_Comm_split(gridComm, coords[coordYIndex], coords[coordXIndex], &columnsComm);
    MPI_Comm_split(gridComm, coords[coordXIndex], coords[coordYIndex], &rowsComm);

    // создание необходимых матриц
    double *matrixA;
    double *matrixB;
    double *matrixC;

    double startTime;

    if (isMainProcess(currentProcessRank)) {
        matrixA = new double[n1 * n2];
        matrixB = new double[n2 * n3];
        matrixC = new double[n1 * n3];

//        srand(time(NULL));

        // заполнение матриц А и В
//        for (int i = 0; i < n1; i++) {
//            for (int j = 0; j < n2; j++) {
//                matrixA[i * n2 + j] = rand() % 5;
//            }
//        }
        fillMatrix(matrixA, n1, n2);

//        for (int i = 0; i < n2; i++) {
//            for (int j = 0; j < n3; j++) {
//                matrixB[i * n3 + j] = rand() % 5;
//            }
//        }
        fillMatrix(matrixB, n2, n3);

//        printInfoAboutMatrix("matrixA", n1, n2, matrixA);
//        printInfoAboutMatrix("matrixB", n2, n3, matrixB);

        startTime = MPI_Wtime();
    }
    // Размер матрицы должен быть кратен количеству запущенных процессов
    int rowsInOneSegmentNum = n1 / dims[coordXIndex];
    int columnsInOneSegmentNum = n3 / dims[coordYIndex];

    double *matASegment = new double[rowsInOneSegmentNum * n2];
    double *matBSegment = new double[columnsInOneSegmentNum * n2];
    double *matCSegment = new double[rowsInOneSegmentNum * columnsInOneSegmentNum];
    fill(matCSegment, matCSegment + rowsInOneSegmentNum * columnsInOneSegmentNum,0);

    // 1. Распределение матрицы А по горизонтальным полосам вдоль координаты (х, 0)
    if (isProcessInFirstColumnInCartTopology(coords, coordYIndex)) {
        MPI_Scatter(matrixA,
                    rowsInOneSegmentNum * n2,
                    MPI_DOUBLE,
                    matASegment,
                    rowsInOneSegmentNum * n2,
                    MPI_DOUBLE,
                    0,
                    columnsComm);
    }

    // 2. Распределение матрицы В по вертикальным полосам вдоль координаты (0, y)
    if (isProcessInFirstRowInCartTopology(coords, coordXIndex)) {

        // Создание нового типа данных - MPI вектора
        MPI_Datatype segmentToSendInVectorRepresentation;

        MPI_Type_vector(n2,
                        columnsInOneSegmentNum,
                        n3,
                        MPI_DOUBLE,
                        &segmentToSendInVectorRepresentation);

        MPI_Type_commit(&segmentToSendInVectorRepresentation);

        MPI_Datatype segmentToSendInDoubleRepresentation;
        MPI_Type_create_resized(segmentToSendInVectorRepresentation,
                                0,
                                columnsInOneSegmentNum * sizeof(double),
                                &segmentToSendInDoubleRepresentation);
        MPI_Type_commit(&segmentToSendInDoubleRepresentation);

        MPI_Scatter(matrixB,
                    1, // значит, что мы посылаем одну полосу
                    segmentToSendInDoubleRepresentation,
                    matBSegment,
                    n2 * columnsInOneSegmentNum,
                    MPI_DOUBLE,
                    0,
                    rowsComm);

        MPI_Type_free(&segmentToSendInVectorRepresentation);
        MPI_Type_free(&segmentToSendInDoubleRepresentation);
    }

    // 3. Распространение полос матрицы А в измерении y
    MPI_Bcast(matASegment, rowsInOneSegmentNum * n2, MPI_DOUBLE, 0, rowsComm);

    // 4. Распространение полос матрицы В в измерении x
    MPI_Bcast(matBSegment, n2 * columnsInOneSegmentNum, MPI_DOUBLE, 0, columnsComm);

    // 5. Каждый процесс вычисляет одну подматрицу произведения матриц
    for (int i = 0; i < rowsInOneSegmentNum; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < columnsInOneSegmentNum; ++k) {
                matCSegment[i * columnsInOneSegmentNum + k] +=
                        matASegment[i * n2 + j] *
                        matBSegment[j * columnsInOneSegmentNum + k];
            }
        }
    }

    // 6. Сбор по каждому процессу результатов вычислений подматриц матрицы С
    // в одну матрицу по процессу (0,0)
    MPI_Datatype segmentToReceive;

    MPI_Type_vector(rowsInOneSegmentNum,
                    columnsInOneSegmentNum,
                    n3,
                    MPI_DOUBLE,
                    &segmentToReceive);

    MPI_Type_commit(&segmentToReceive);

    MPI_Datatype segmentToReceiveResized;
    MPI_Type_create_resized(segmentToReceive,
                            0,
                            columnsInOneSegmentNum * sizeof(double),
                            &segmentToReceiveResized);
    MPI_Type_commit(&segmentToReceiveResized);

    int recvCounts[totalProcessesNumber];
    int displs[totalProcessesNumber];

    for (int i = 0; i < dims[coordXIndex]; ++i) {
        for (int j = 0; j < dims[coordYIndex]; ++j) {
            recvCounts[i * dims[coordYIndex] + j] = 1;
            displs[i * dims[coordYIndex] + j] = j + i * rowsInOneSegmentNum * dims[coordYIndex];
        }
    }

    MPI_Gatherv(matCSegment, columnsInOneSegmentNum * rowsInOneSegmentNum,
                MPI_DOUBLE, matrixC,
                recvCounts, displs, segmentToReceiveResized, 0, MPI_COMM_WORLD);

    if (isMainProcess(currentProcessRank)) {
        double endTime = MPI_Wtime();
        double totalElapsedTime = endTime - startTime;
        if (isCorrectCalculation(matrixC, n1, n3, n2)) {
//            printInfoAboutMatrix("matrixC", n1, n3, matrixC);
            printElapsedTime(totalElapsedTime);
        } else {
            std::cout << "error" << std::endl;
        }
    }

    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&rowsComm);
    MPI_Comm_free(&columnsComm);

    free(matASegment);
    free(matBSegment);
    free(matCSegment);

    if (isMainProcess(currentProcessRank)) {
        free(matrixA);
        free(matrixB);
        free(matrixC);
    }

    MPI_Finalize();
    return 0;

}
