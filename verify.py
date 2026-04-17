import numpy as np
import os


def read_matrix(filename):
    """Чтение матрицы из файла"""
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        matrix = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            matrix.append(row)
        return np.array(matrix)


# Размеры матриц
sizes = [200, 400, 800, 1200, 1600, 2000]

print("=" * 50)
print("ВЕРИФИКАЦИЯ CUDA")
print("=" * 50)

passed = 0
total = 0

for n in sizes:
    folder = f"results_cuda/data_{n}"

    if not os.path.exists(folder):
        print(f"Папка {folder} не найдена")
        continue

    A = read_matrix(f"{folder}/A.txt")
    B = read_matrix(f"{folder}/B.txt")

    C_ref = np.dot(A, B)

    C_gpu = read_matrix(f"{folder}/C_gpu.txt")

    diff = np.max(np.abs(C_ref - C_gpu))
    total += 1

    if diff < 1e-6:
        print(f"{n}x{n} ПРОЙДЕНО (diff={diff:.2e})")
        passed += 1
    else:
        print(f"{n}x{n} НЕ ПРОЙДЕНО (diff={diff:.2e})")
        
print("=" * 50)
print(f"РЕЗУЛЬТАТ: {passed}/{total}")
if passed == total:
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
print("=" * 50)