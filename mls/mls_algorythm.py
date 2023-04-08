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
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt


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


def read_data(path: str) -> tuple:
    """Считывание исходных данных из файла
    Формат файла: \n
    x1 x2 x3 ... \n
    y1 y2 y3 ...

    :return: список значений аргументов функции и список
    значений функции, объединённые в кортеж
    """
    with open(path, "r", encoding="utf-8") as file:
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
    for _, (current_x, current_y) in enumerate(zip(x_data, y_data)):
        func_value = expr.subs(sm.symbols("x"), current_x)
        result += (func_value - current_y)**2
    return result


# pylint: disable=too-many-instance-attributes
def plot(x_val: np.ndarray, np_poly2, np_poly3, x_data: list, y_data: list):
    """
    Отрисовка графика двух полиномов и точек
    :param x_val: диапазон аргумента
    :param np_poly2: функция первого полинома
    :param np_poly3: функция второго полинома
    :param x_data: координаты по оси абсцисс исходных точек
    :param y_data: координаты по оси ординат исходных точек
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position(("data", 0.0))
    ax.spines['bottom'].set_position(("data", 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.plot(x_val, np_poly2(x_val), '--', label="Poly2")
    ax.plot(x_val, np_poly3(x_val), label="Poly3")
    ax.plot(x_data, y_data, "ro")

    ax.legend()


def main():
    """Считывание исходных данных, построение полиномов 2 и 3 степени, отрисовка графика
    и выбор лучшего полинома исходя из их стандартных отклонений

    :return: None
    """
    x_data, y_data = read_data("mls_data.txt")
    poly2 = build_poly(x_data, y_data, 2)
    poly3 = build_poly(x_data, y_data, 3)
    print(poly2, poly3, sep="\n")
    print("2-degree", std_dev(x_data, y_data, poly2))
    print("3-degree", std_dev(x_data, y_data, poly3))

    np_poly2 = lambdify(tuple(poly2.atoms(sm.Symbol)), poly2, modules=['numpy'])
    np_poly3 = lambdify(tuple(poly3.atoms(sm.Symbol)), poly3, modules=['numpy'])

    x_val = np.linspace(-10, 10, 200)
    plot(x_val, np_poly2, np_poly3, x_data, y_data)
    plt.savefig("../graphics/mls_all.jpg")
    plt.style.use('dark_background')
    plot(x_val, np_poly2, np_poly3, x_data, y_data)
    plt.savefig("../graphics/mls_all_dark.jpg")

    x_val = np.linspace(0,2.1, 100)
    plt.style.use('default')
    plot(x_val, np_poly2, np_poly3, x_data, y_data)
    plt.savefig("../graphics/mls_points.jpg")
    plt.style.use('dark_background')
    plot(x_val, np_poly2, np_poly3, x_data, y_data)
    plt.savefig("../graphics/mls_points_dark.jpg")


if __name__ == "__main__":
    main()
