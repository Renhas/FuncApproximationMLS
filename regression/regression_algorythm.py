"""
Модуль, реализующий простейшую линейную регрессию

Функции:
    eps1_calculate(list) - вычисление отклонения
    main() - поиск полинома
"""
import numpy as np
from sympy.plotting import plot
import mls.mls_algorythm as mls


def eps1_calculate(y_list: list) -> float:
    """Функция вычисления величины отклонения значений от их среднего

    :param y_list: список значений
    """
    y_array = np.array(y_list)
    return ((y_array - y_array.mean())**2).sum()


def main():
    """Основная функция поиска полинома и его отрисовки"""
    x_data, y_data = mls.read_data("reg_data.txt")
    degree = 1
    eps1 = eps1_calculate(y_data)
    eps2 = eps1
    poly = 0
    while abs(eps2 - eps1) < 0.1:
        poly = mls.build_poly(x_data, y_data, degree)
        eps2 = mls.std_dev(x_data, y_data, poly)
        degree += 1
    print(f"Poly: {poly}")
    print(f"eps1: {eps1}, eps2: {eps2}, difference: {abs(eps2-eps1)}")
    palette = plot(poly, show=True, legend=False,
                   size=(7, 7), label=f"poly{degree}",
                   markers=[{'args': [x_data, y_data, "ro"]}],
                   xlim=(min(x_data)-0.1, max(x_data)+0.1),
                   ylim=(0, 9))
    palette.save("../graphics/regression.jpg")


if __name__ == "__main__":
    main()
