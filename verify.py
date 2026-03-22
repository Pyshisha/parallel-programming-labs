import numpy as np
import os


def read_matrix(filename):
    """Читает матрицу из файла"""
    try:
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            matrix = []
            for _ in range(n):
                row = list(map(float, f.readline().split()))
                matrix.append(row)
        return np.array(matrix)
    except Exception as e:
        print(f"Ошибка чтения {filename}: {e}")
        return None


sizes = [200, 400, 800, 1200, 1600, 2000]

print("=" * 50)
print("Результаты")
print("=" * 50)

all_passed = True
passed_count = 0

for n in sizes:
    folder = f"data_{n}"

    if not os.path.exists(folder):
        print(f"Папка {folder} не найдена")
        continue

    A = read_matrix(f"{folder}/A.txt")
    B = read_matrix(f"{folder}/B.txt")
    C_cpp = read_matrix(f"{folder}/C.txt")

    if A is None or B is None or C_cpp is None:
        print(f"Размер {n}x{n}: не удалось прочитать файлы")
        all_passed = False
        continue

    C_ref = np.dot(A, B)

    diff = np.max(np.abs(C_cpp - C_ref))

    if diff < 1e-6:
        print(f"ПРОЙДЕНО Размер {n}x{n}: максимальная разница = {diff:.2e}")
        passed_count += 1
    else:
        print(f"НЕ ПРОЙДЕНО Размер {n}x{n}: максимальная разница = {diff:.2e}")
        all_passed = False

        print(f"   Первый элемент C_cpp[0,0] = {C_cpp[0, 0]:.6f}")
        print(f"   Первый элемент C_ref[0,0] = {C_ref[0, 0]:.6f}")
        print(f"   Первый элемент A[0,0] = {A[0, 0]:.6f}")
        print(f"   Первый элемент B[0,0] = {B[0, 0]:.6f}")

print("=" * 50)
print(f"Верификация: {passed_count}/{len(sizes)} пройдено")
if all_passed:
    print("ВСЕ РЕЗУЛЬТАТЫ ПРОШЛИ ВЕРИФИКАЦИЮ")
else:
    print("НЕ ВСЕ РЕЗУЛЬТАТЫ ПРОШЛИ ВЕРИФИКАЦИЮ")
print("=" * 50)