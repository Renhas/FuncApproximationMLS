"""
Поиск минимума функции одной переменной методом золотого сечения
"""
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
A_VALUE = 4.0
B_VALUE = -0.25
GS_VALUE = 0.618


def function(x_value: Union[float, np.ndarray]) -> float:
    """Искомая функция

    :param x_value: одно или несколько значений
    :return: одно или несколько значений
    """
    return x_value ** 2 + A_VALUE * np.exp(B_VALUE * x_value)


def golden_algorythm(start: float, end: float, eps: float = 0.001) -> float:
    """Метод золотого сечения для нахождения минимума функции

    :param start: нижняя граница отрезка
    :param end: верхняя граница отрезка
    :param eps: точность
    :return: точка минимума
    """
    current_a = start
    current_b = end
    current_left = calculate_golden_left(start, end)
    current_right = calculate_golden_right(start, end)

    while current_b - current_a > eps:
        if function(current_left) < function(current_right):
            current_b = current_right
            current_right = current_left
            current_left = calculate_golden_left(current_a, current_b)
        else:
            current_a = current_left
            current_left = current_right
            current_right = calculate_golden_right(current_a, current_b)

    return (current_b - current_a) * 0.5


def calculate_golden_right(start: float, end: float) -> float:
    """Вычисление правой точки золотого сечения"""
    return start + GS_VALUE * (end - start)


def calculate_golden_left(start: float, end: float) -> float:
    """Вычисление левой точки золотого сечения"""
    return end - GS_VALUE * (end - start)


def main():
    """Основная функция"""
    plt.title(r"График функции $x^2 + ae^{bx}$")
    plt.xlabel("x")
    plt.ylabel("y")

    x_values = np.linspace(-1, 0, 100)
    plt.plot(x_values, function(x_values))

    minimum = golden_algorythm(-1, 0)
    plt.plot(minimum, function(minimum), "ro")
    plt.savefig("../graphics/golden.jpg")
    plt.show()


if __name__ == "__main__":
    main()
