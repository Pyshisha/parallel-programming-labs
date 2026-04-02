#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <locale.h>

using namespace std;
using namespace chrono;

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

#pragma omp parallel for collapse(2)
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

vector<vector<double>> multiplyParallel(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int numThreads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    omp_set_num_threads(numThreads);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

int main() {
    setlocale(LC_ALL, "Russian");
#ifdef _OPENMP
    cout << "OpenMP ВКЛЮЧЕН" << endl;
#else
    cout << "OpenMP НЕ ВКЛЮЧЕН" << endl;
#endif

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;
    int threads[] = { 1, 2, 4, 8 };
    int numThreads = 4;

    system("mkdir results_openmp 2>nul");

    ofstream resultsFile("results_openmp/results.csv");
    resultsFile << "Size,Threads,Time_seconds\n";

    cout << "========================================" << endl;
    cout << "ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ (OPENMP)" << endl;
    cout << "========================================" << endl;
    cout << "Доступно потоков: " << omp_get_max_threads() << endl;

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];

        cout << "\n========== Размер " << n << "x" << n << " ==========" << endl;

        string folder = "results_openmp\\data_" + intToString(n);
        string cmd = "mkdir " + folder + " 2>nul";
        system(cmd.c_str());

        cout << "Генерация матриц..." << endl;
        auto A = generateMatrix(n);
        auto B = generateMatrix(n);

        saveMatrix(folder + "/A.txt", A);
        saveMatrix(folder + "/B.txt", B);

        cout << "Умножение..." << endl;

        double sequentialTime = 0.0;

        for (int t = 0; t < numThreads; ++t) {
            int threadCount = threads[t];

            cout << "  Потоков: " << threadCount << "... ";
            cout.flush();

            double totalTime = 0.0;
            int runs = 3;

            for (int r = 0; r < runs; ++r) {
                auto start = high_resolution_clock::now();
                auto C = multiplyParallel(A, B, threadCount);
                auto end = high_resolution_clock::now();
                totalTime += duration_cast<milliseconds>(end - start).count() / 1000.0;

                if (r == runs - 1) {
                    saveMatrix(folder + "/C_" + intToString(threadCount) + ".txt", C);
                }
            }

            double avgTime = totalTime / runs;

            if (threadCount == 1) {
                sequentialTime = avgTime;
            }


            cout << avgTime << " сек, ";

            resultsFile << n << "," << threadCount << "," << avgTime << "\n";
        }
    }

    resultsFile.close();

    cout << "\n========================================" << endl;
    cout << "РЕЗУЛЬТАТЫ СОХРАНЕНЫ В results_openmp/results.csv" << endl;
    cout << "========================================" << endl;

    return 0;
}
