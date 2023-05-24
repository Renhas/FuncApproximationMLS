"""
Алгоритмы приближенного решения системы ДУ

Классы:
    BaseAlgorythm - абстрактный класс, реализующий общую схему алгоритма
    Euler - класс, реализующий метод Эйлера
    RungeKutta - класс, реализующий метод Рунге-Кутта

Функции:
    func1 - первая функция в системе ДУ
    func2 - вторая функция в системе ДУ
    plot_results - построение графика с результатами работы метода
    main - основная функция запуска методов и отрисовки
"""
import abc
import typing
from abc import ABC
import copy
import numpy as np
import matplotlib.pyplot as plt


def func1(_: float, y_vector: list) -> float:
    """Функция f(x, y1, y2) = y2

    :param _: заглушка для общего вида функций
    :param y_vector: вектор (y1, y2)
    :return: значение функции
    """
    return y_vector[1]


def func2(x_value: float, _: list) -> float:
    """Функция f(x, y1, y2) = -0.01 * exp{-0.8*x}

    :param x_value: значение x
    :param _: заглушка для общего вида функции
    :return: значение функции
    """
    return -0.01 * np.e ** (-0.8 * x_value)


# pylint: disable=too-few-public-methods
class BaseAlgorythm(ABC):
    """Абстрактный класс, реализующий общую схему алгоритма

    Методы:
        run - запуска алгоритма
    """
    def __init__(self, functions: tuple):
        """Конструктор класса

        :param functions: список функций, образующих СДУ
        """
        self.__functions = copy.copy(functions)
        self._current_x = 0.
        self._current_y = [0., 0.]
        self._step = 0.

    def run(self, start_points: list, step: float, x_lim: tuple = (0, 1),
            *, action: typing.Callable = None) -> tuple:
        """Запуск метода

        :param start_points: начальные условия СДУ
        :param step: шаг сетки
        :param x_lim: диапазон сетки
        :param action: опциональное действие в начале каждой итерации
        :return: значение итоговой точки
        """
        self._current_x = x_lim[0]
        self._current_y = np.array(start_points).astype("float")
        self._step = step
        while self._current_x < x_lim[1]:
            if action is not None:
                action(self._current_x, tuple(self._current_y))
            self.__one_step()
        return self._current_x, self._current_y

    def __one_step(self):
        """Шаг итерации"""
        for index, value in enumerate(self._current_y):
            func = self.__functions[index]
            f_value = self._step_function(func)
            self._current_y[index] = value + self._step * f_value
        self._current_x += self._step

    @abc.abstractmethod
    def _step_function(self, func: typing.Callable) -> float:
        """Функция вычисления шагового множителя

        :param func: функция из СДУ
        :return: множитель
        """


class Euler(BaseAlgorythm):
    """Метод Эйлера приближенного решения СДУ"""
    def _step_function(self, func: typing.Callable) -> float:
        """Реализация множителя для метода Эйлера

        :param func: функция из СДУ
        :return: множитель
        """
        return func(self._current_x, self._current_y)


class RungeKutta(BaseAlgorythm):
    """Метод Рунге-Кутта приближенного решения СДУ"""
    def _step_function(self, func: typing.Callable) -> float:
        """Реализация множителя для метода Рунге-Кутта

        :param func: функция из СДУ
        :return: множитель
        """
        temp_x = copy.copy(self._current_x)
        temp_y = copy.copy(self._current_y)
        value_1 = func(temp_x, temp_y)
        temp_x += self._step / 2
        temp_y = self._current_y + value_1 * self._step / 2
        value_2 = func(temp_x, temp_y)
        temp_y = self._current_y + value_2 * self._step / 2
        value_3 = func(temp_x, temp_y)
        temp_x += self._step / 2
        temp_y = self._current_y + value_3 * self._step
        value_4 = func(temp_x, temp_y)
        f_value = value_1 + 2 * value_2 + 2 * value_3 + value_4
        return f_value / 6


# pylint: disable=no-member
def plot_results(algorythm: BaseAlgorythm, params: list, path: str = ""):
    """Отрисовка результатов работы алгоритма

    :param algorythm: алгоритм
    :param params: список параметров
    :param path: путь для сохранения
    """
    x_values = []
    y_values = []

    def action(current_x, current_y):
        """Сохранение результатов итерации"""
        x_values.append(current_x)
        y_values.append(current_y)

    res = algorythm.run(*params, action=action)
    x_values.append(res[0])
    y_values.append(res[1])
    y_values = np.array(y_values)

    plt.plot(x_values, y_values[:, 0], "ro")
    plt.plot(x_values, y_values[:, 1], "g*")
    plt.legend(["y1", "y2"])
    if path:
        plt.savefig(path)
    plt.show()


def main():
    """Основная функция, запускающая алгоритмы"""
    start_points = (0, 0.5)
    functions = (func1, func2)
    params = [start_points, 0.1, (0, 3)]
    plot_results(Euler(functions), params, path="../graphics/euler_0.1.jpg")
    params[1] = 0.05
    plot_results(Euler(functions), params, path="../graphics/euler_0.05.jpg")
    params[1] = 0.1
    plot_results(RungeKutta(functions), params,
                 path="../graphics/runge_0.1.jpg")
    params[1] = 0.05
    plot_results(RungeKutta(functions), params,
                 path="../graphics/runge_0.05.jpg")


if __name__ == "__main__":
    main()
