#include <iostream>
#include <mpi.h>          
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale.h>
#include <windows.h> 
#include <direct.h>

using namespace std;

string intToString(int n) {
    stringstream ss;
    ss << n;
    return ss.str();
}

vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

void saveMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();
    file << n << endl;
    file << fixed << setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<double> flattenMatrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<double> flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = matrix[i][j];
    return flat;
}

vector<vector<double>> unflattenMatrix(const vector<double>& flat, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = flat[i * n + j];
    return matrix;
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;

    if (world_rank == 0) {
        _mkdir("results_mpi");

        ifstream check("results_mpi/results.csv");
        bool file_exists = check.good();
        check.close();

        ofstream resultsFile("results_mpi/results.csv", ios::app);

  
        if (!file_exists) {
            resultsFile << "Size,Processes,Time_seconds\n";
        }
        resultsFile.close();

        std::cout << "========================================" << endl;
        std::cout << "ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ (MPI)" << endl;
        std::cout << "========================================" << endl;
        std::cout << "Запущено процессов: " << world_size << endl;
    }

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];

        if (n % world_size != 0) {
            if (world_rank == 0) {
                cerr << "Ошибка: размер " << n << " не делится на " << world_size << " процессов" << endl;
            }
            continue;
        }

        int rows_per_proc = n / world_size;

        vector<double> flat_A, flat_B, flat_C;
        vector<double> local_A(rows_per_proc * n);
        vector<double> local_C(rows_per_proc * n);

        double elapsed = 0.0;

        if (world_rank == 0) {
            string folder = "results_mpi/data_" + intToString(n);
            _mkdir(folder.c_str());

            std::cout << "\n========== Размер " << n << "x" << n << " ==========" << endl;
            std::cout << "Генерация матриц..." << endl;

            auto A = generateMatrix(n);
            auto B = generateMatrix(n);

            saveMatrix(folder + "/A.txt", A);
            saveMatrix(folder + "/B.txt", B);

            flat_A = flattenMatrix(A);
            flat_B = flattenMatrix(B);
            flat_C.resize(n * n);

            std::cout << "Умножение... (процессов: " << world_size << ")" << endl;
        }

        if (world_rank != 0) {
            flat_B.resize(n * n);
        }
        MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            MPI_Scatter(flat_A.data(), rows_per_proc * n, MPI_DOUBLE,
                local_A.data(), rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }
        else {
            MPI_Scatter(nullptr, rows_per_proc * n, MPI_DOUBLE,
                local_A.data(), rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        for (int i_row = 0; i_row < rows_per_proc; ++i_row) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += local_A[i_row * n + k] * flat_B[k * n + j];
                }
                local_C[i_row * n + j] = sum;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        elapsed = end_time - start_time;

        if (world_rank == 0) {
            MPI_Gather(local_C.data(), rows_per_proc * n, MPI_DOUBLE,
                flat_C.data(), rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }
        else {
            MPI_Gather(local_C.data(), rows_per_proc * n, MPI_DOUBLE,
                nullptr, rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }

        if (world_rank == 0) {
            auto C = unflattenMatrix(flat_C, n);
            string folder = "results_mpi/data_" + intToString(n);
            saveMatrix(folder + "/C.txt", C);

            std::cout << "Время MPI: " << elapsed << " сек" << endl;

            ofstream resultsFile("results_mpi/results.csv", ios::app);
            resultsFile << n << "," << world_size << "," << elapsed << "\n";
            resultsFile.close();
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        std::cout << "\n========================================" << endl;
        std::cout << "РЕЗУЛЬТАТЫ СОХРАНЕНЫ В results_mpi/results.csv" << endl;
        std::cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}