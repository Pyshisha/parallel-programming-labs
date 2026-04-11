import numpy as np
import os


def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        return np.array([list(map(float, f.readline().split())) for _ in range(n)])


sizes = [200, 400, 800, 1200, 1600, 2000]

print("=" * 50)
print("ВЕРИФИКАЦИЯ MPI")
print("=" * 50)

passed = 0
total = 0

for n in sizes:
    folder = f"x64/Release/results_mpi/data_{n}"

    if not os.path.exists(folder):
        print(f"{folder} не найдена")
        continue

    A = read_matrix(f"{folder}/A.txt")
    B = read_matrix(f"{folder}/B.txt")
    C_ref = np.dot(A, B)

    filename = f"{folder}/C.txt"
    if not os.path.exists(filename):
        print(f"{folder}/C.txt не найден")
        continue

    C_mpi = read_matrix(filename)
    diff = np.max(np.abs(C_mpi - C_ref))
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