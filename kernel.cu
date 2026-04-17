#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <direct.h>

using namespace std;
using namespace chrono;

string intToString(int n) {
    stringstream ss;
    ss << n;
    return ss.str();
}

vector<double> generateMatrix(int n) {
    vector<double> matrix(n * n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < n * n; ++i) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

void saveMatrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    file << fixed << setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

double runGPUExperiment(const vector<double>& A, const vector<double>& B,
    vector<double>& C, int n,
    int blockSizeX, int blockSizeY,
    const string& resultFilename) {
    double* d_A, * d_B, * d_C;

    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_B, n * n * sizeof(double));
    cudaMalloc(&d_C, n * n * sizeof(double));

    cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 numBlocks((n + blockSizeX - 1) / blockSizeX,
        (n + blockSizeY - 1) / blockSizeY);

    for (int warmup = 0; warmup < 5; ++warmup) {
        matrixMulKernel <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
    }

    double totalTime = 0.0;
    int runs = 3;

    for (int r = 0; r < runs; ++r) {
        auto start = high_resolution_clock::now();
        matrixMulKernel <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;
        totalTime += elapsed.count();
    }

    double seconds = totalTime / runs;

    cudaMemcpy(C.data(), d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    saveMatrix(resultFilename, C, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return seconds;
}

int main() {
    setlocale(LC_ALL, "Russian");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;

    struct Config {
        int x, y;
        string name;
    };

    Config configs[] = {
        {8, 8, "8x8"},
        {16, 16, "16x16"},
        {32, 32, "32x32"},
        {64, 64, "64x64"},
        {32, 8, "32x8"},
        {64, 4, "64x4"},
        {128, 2, "128x2"},
        {8, 32, "8x32"},
        {4, 64, "4x64"},
        {2, 128, "2x128"}
    };
    int numConfigs = 10;
    system("mkdir results_cuda 2>nul");

    ofstream resultsFile("results_cuda/results.csv");
    resultsFile << "Size,BlockConfig,Time_seconds\n";

    cout << "========================================" << endl;
    cout << "ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ (CUDA)" << endl;
    cout << "========================================" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Максимум потоков в блоке: " << prop.maxThreadsPerBlock << endl;
    cout << endl;

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];

        cout << "РАЗМЕР " << n << "x" << n << endl;

        string folder = "results_cuda\\data_" + intToString(n);  
        _mkdir(folder.c_str()); 

        cout << "Генерация матриц..." << endl;
        auto A = generateMatrix(n);
        auto B = generateMatrix(n);

        saveMatrix(folder + "/A.txt", A, n);
        saveMatrix(folder + "/B.txt", B, n);

        cout << "Вычисление с разными конфигурациями:" << endl;
        vector<double> C_gpu(n * n);

        for (int cfg = 0; cfg < numConfigs; ++cfg) {
            int bx = configs[cfg].x;
            int by = configs[cfg].y;

            if (bx * by > prop.maxThreadsPerBlock) {
                cout << "  " << configs[cfg].name << " - НЕ ПОДДЕРЖИВАЕТСЯ (>"
                    << prop.maxThreadsPerBlock << ")" << endl;
                continue;
            }

            string resultFile = folder + "/C_gpu_" + configs[cfg].name + ".txt";
            double time = runGPUExperiment(A, B, C_gpu, n, bx, by, resultFile);

            cout << "  " << configs[cfg].name << ": " << time << " сек" << endl;
            resultsFile << n << "," << configs[cfg].name << "," << time << "\n";
        }
        saveMatrix(folder + "/C_gpu.txt", C_gpu, n);
    }

    resultsFile.close();

    cout << "========================================" << endl;
    cout << "РЕЗУЛЬТАТЫ СОХРАНЕНЫ В results_cuda/results.csv" << endl;
    cout << "========================================" << endl;

    return 0;
}
