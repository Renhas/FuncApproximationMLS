import sympy as sm
import numpy as np
from sympy.plotting import plot


def coefs_calculate(x_data: list, y_data: list, degree: int) -> list:
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
    system = np.array(system)
    row, column = system.shape
    matrix = system[:, 0:column-1]
    vector = system[:, column-1]
    return np.linalg.solve(matrix, vector)


def read_data() -> tuple:
    file = open("Data.txt", "r")
    x_data = [float(i) for i in file.readline().split(" ")]
    y_data = [float(i) for i in file.readline().split(" ")]
    return x_data, y_data


def build_poly(x_data: list, y_data: list, degree: int) -> sm.Expr:
    coefs_matrix = coefs_calculate(x_data, y_data, degree)
    coefs = solve_system(coefs_matrix)
    x = sm.symbols("x")
    expr = 0 * x
    for n in range(degree+1):
        expr += coefs[n] * x**n
    return expr


def std_dev(x_data: list, y_data: list, expr: sm.Expr) -> float:
    if len(x_data) != len(y_data):
        raise ValueError("x and y must be the same length")
    if not isinstance(expr, sm.Expr):
        raise TypeError(f"expr must be an expression, but it is {type(expr)}")
    result = 0
    for i in range(len(x_data)):
        current_x = x_data[i]
        current_y = y_data[i]
        func_value = expr.subs(sm.symbols("x"), current_x)
        result += (func_value - current_y)**2
    return result


def main():
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
