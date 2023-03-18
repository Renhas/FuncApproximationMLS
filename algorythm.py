"""Реализация аппроксимации функции с помощью метода наименьших квадратов

Функции:
    coefs_calculate(list, list, int) -> list
    solve_system(list) -> numpy.ndarray
    read_data() -> tuple
    build_poly(list, list, int) -> sympy.Expr
    std_dev(list, list, sympy.Expr) -> float
    main()
"""
import sympy as sm
import numpy as np
from sympy.plotting import plot


def coefs_calculate(x_data: list, y_data: list, degree: int) -> list:
    """ Формирование СЛАУ для нахождения коэффициентов полинома

    :param x_data: список аргументов целевой функции
    :param y_data: список значений целевой функции
    :param degree: степень полинома
    :return: СЛАУ в виде двумерного списка
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data should have same length")
    if degree < 1:
        raise ValueError("degree should be >=1 ")
    coefs = []
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    for row in range(degree+1):
        coefs.append([])
        for column in range(degree+1):
            value = (x_data ** (column + row)).sum()
            coefs[row].append(value)
        x_value = x_data ** row
        coefs[row].append((y_data * x_value).sum())
    return coefs


def solve_system(system: list) -> np.ndarray:
    """Решение СЛАУ

    :param system: СЛАУ в виде двумерного списка
    :return: список коэффициентов в виде numpy.ndarray
    """
    system = np.array(system)
    column = system.shape[1]
    matrix = system[:, 0:column-1]
    vector = system[:, column-1]
    return np.linalg.solve(matrix, vector)


def read_data() -> tuple:
    """Считывание исходных данных из файла
    Формат файла: \n
    x1 x2 x3 ... \n
    y1 y2 y3 ...

    :return: список значений аргументов функции и список
    значений функции, объединённые в кортеж
    """
    with open("Data.txt", "r", encoding="utf-8") as file:
        x_data = [float(i) for i in file.readline().split(" ")]
        y_data = [float(i) for i in file.readline().split(" ")]
        return x_data, y_data


def build_poly(x_data: list, y_data: list, degree: int) -> sm.Expr:
    """Построение аппроксимирующего полинома

    :param x_data: список значений аргументов исходной функции
    :param y_data: список значений исходной функции
    :param degree: степень полинома
    :return: полином в виде sympy.Expr
    """
    coefs_matrix = coefs_calculate(x_data, y_data, degree)
    coefs = solve_system(coefs_matrix)
    x_sym = sm.symbols("x")
    expr = 0 * x_sym
    for deg in range(degree+1):
        expr += coefs[deg] * x_sym ** deg
    return expr


def std_dev(x_data: list, y_data: list, expr: sm.Expr) -> float:
    """Вычисление стандартного отклонения найденного полинома

    :param x_data: список значений аргументов исходной функции
    :param y_data: список значений исходной функции
    :param expr: полином в виде sympy.Expr
    :return: стандартное отклонение
    """
    if len(x_data) != len(y_data):
        raise ValueError("x and y must be the same length")
    if not isinstance(expr, sm.Expr):
        raise TypeError(f"expr must be an expression, but it is {type(expr)}")
    result = 0
    for index, (current_x, current_y) in enumerate(zip(x_data, y_data)):
        func_value = expr.subs(sm.symbols("x"), current_x)
        result += (func_value - current_y)**2
    return result


def main():
    """Считывание исходных данных, построение полиномов 2 и 3 степени, отрисовка графика
    и выбор лучшего полинома исходя из их стандартных отклонений

    :return: None
    """
    x_data, y_data = read_data()
    poly2 = build_poly(x_data, y_data, 2)
    poly3 = build_poly(x_data, y_data, 3)
    print(poly2, poly3, sep="\n")
    palette = plot(poly2, show=False, legend=True,
                   markers=[{'args': [x_data, y_data, "ro"]}],
                   size=(7, 7), label="poly2")
    palette2 = plot(poly3, show=False, legend=True, label="poly3", size=(7, 7))
    palette.append(palette2[0])
    palette.show()

    print("2-degree", std_dev(x_data, y_data, poly2))
    print("3-degree", std_dev(x_data, y_data, poly3))


if __name__ == "__main__":
    main()
