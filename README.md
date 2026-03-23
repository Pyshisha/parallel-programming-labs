# Лабораторная работа №1: Последовательное умножение матриц

## Автор
- **Студент:** Рыжков М.А.
- **Группа:** 6311
- **Курс:** Параллельное программирование

---

## Цель работы
Разработать программу на языке C++ для умножения двух квадратных матриц, провести эксперименты с различными размерами матриц, выполнить верификацию результатов с помощью Python/NumPy.

---

## Реализация

### Алгоритм умножения
Используется оптимизированный порядок циклов **i-k-j**, который обеспечивает эффективное использование кэш-памяти процессора:

```cpp
vector<vector<double>> multiply(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}
```


### Генерация матриц
Матрицы генерируются случайным образом с использованием генератора mt19937:

```cpp
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
```

### Сохранение матриц
Результаты сохраняются в файлы с высокой точностью (15 знаков после запятой) для корректной верификации:

```cpp
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
```
### Верификация результата 
Скрипт verify.py
```python
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
```
## Описание экспериментов

### Цель экспериментов
Определить зависимость времени выполнения умножения матриц от их размера, а также оценить производительность алгоритма.

### Параметры экспериментов
- **Размеры матриц:** 200, 400, 800, 1200, 1600, 2000
- **Тип данных:** double
- **Количество экспериментов:** 6 (по одному на каждый размер)
- **Измеряемые параметры:** Время выполнения (секунды)


### Методика проведения
1. Для каждого размера n генерируются две случайные матрицы A и B размером n×n
2. Матрицы сохраняются в файлы для последующей верификации
3. Выполняется умножение матриц C = A × B с замером времени
4. Результат C сохраняется в файл
5. Выполняется верификация путем сравнения с эталонным умножением в NumPy

## Результат выполнения программы

![Вывод программы](images/Снимок%20экрана%202026-03-23%20143435.png)

## Результат верификации

![Верификация](images/Снимок%20экрана%202026-03-22%20184603.png)```
## Результаты экспериментов
### Время выполнения

| Размер, n | Время, сек. |
|-------------|-------------|
| 200    | 0.188    |
| 400    | 1.473   |
| 800   | 12.434   |
| 1200    | 42.079   |
| 1600    | 100.061    |
| 2000    | 189.016    |

### График зависимости времени от размера матрицы

![График времени выполнения](images/Снимок%20экрана%202026-03-23%20144550.png)
### Анализ
Время выполнения растет пропорционально O(n³)

## Запуск

### Windows (Visual Studio)
1. Откройте `MatrixMultiplication.sln`
2. Нажмите **Ctrl+F5**

## Запуск верификации

### Windows (командная строка)
```bash
pip install numpy
python verify.py
```

## Выводы

В ходе выполнения лабораторной работы была разработана программа на языке C++ для умножения квадратных матриц. Проведены эксперименты с матрицами размеров 200×200, 400×400, 800×800, 1200×1200, 1600×1600 и 2000×2000. Выполнена автоматизированная верификация результатов с помощью библиотеки NumPy на Python. Все 6 экспериментов успешно прошли верификацию
