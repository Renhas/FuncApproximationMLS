"""Тесты для проверки методов mls_algorythm.py

Классы:
    TestCoefs
    TestSolver
    TestPolyBuilder
    TestStdDev
"""
import pytest
import numpy as np
import sympy as sm
from mls.mls_algorythm import coefs_calculate, solve_system, build_poly, std_dev


class TestCoefs:
    """Тесты для функции формирования СЛАУ

    Методы:
        test_good(list, list, int, list) \n
        test_bad(list, list, int)
    """
    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree", "matrix"), [
            ([1, 2, 3], [1, 1, 1], 1, [[3, 6, 3],
                                       [6, 14, 6]]),
            ([1, 2, 3], [1, 1, 1], 2, [[3, 6, 14, 3],
                                       [6, 14, 36, 6],
                                       [14, 36, 98, 14]]),
            ([1, 2], [2, 5], 1, [[2, 3, 7],
                                 [3, 5, 12]]),
            ([1, 2], [2, 5], 2, [[2, 3, 5, 7],
                                 [3, 5, 9, 12],
                                 [5, 9, 17, 22]])
        ]
    )
    def test_good(self, x_data, y_data, degree, matrix):
        """Проверка на корректность полученных результатов

        :param x_data: список значений аргумента исходной функции
        :param y_data: список значений исходной функции
        :param degree: степень полинома
        :param matrix: корректная СЛАУ
        :return: None
        """
        assert coefs_calculate(x_data, y_data, degree) == matrix

    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree"), [
            ([1, 2], [2], 1),
            ([1], [1, 2], 2),
            ([1], [1], -1),
            ([1, 3, 5], [3, 6, 7], 0)
        ]
    )
    def test_bad(self, x_data, y_data, degree):
        """Проверка обработки некорректных данных

        :param x_data: список значений аргумента исходной функции
        :param y_data: список значений исходной функции
        :param degree: степень полинома
        :return: None
        """
        with pytest.raises(ValueError):
            coefs_calculate(x_data, y_data, degree)


class TestSolver:
    """Тестирование функции решения СЛАУ

    Методы:
        test_good(list, numpy.ndarray)
        test_bad(list)
    """
    @pytest.mark.parametrize(
        ("system", "expected"), [
            ([[1, 1, 1, 6],
              [3, 4, 2, 3],
              [2, 5, 3, 4]], np.array([2.5, -5.75, 9.25])),
            ([[0, 0, 1, 3],
              [1, 2, 3, 5],
              [1, 0, 0, 8]], np.array([8, -6, 3])),
            ([[2, 3, 1],
              [4, 5, 8]], np.array([9.5, -6.0]))
        ]
    )
    def test_good(self, system, expected):
        """Проверка корректности решения СЛАУ

        :param system: СЛАУ в виде двумерного массива
        :param expected: вектор-решение в виде numpy.ndarray
        :return: None
        """
        assert (~(solve_system(system) == expected)).sum() == 0

    @pytest.mark.parametrize(
        "system", [
            ([[1, 2, 3, 2],
              [4, 5, 6, 3],
              [7, 8, 9, 4]]),
            ([[0, 0, 0, 3],
              [3, 5, 8, 1],
              [4, 5, 2, 0]]),
            ([[2, 3, 4, 2],
              [8, 7, 4, 0]]),
            ([[4, 5, 3],
              [6, 9, 2],
              [0, 1, 1]])
        ]
    )
    def test_bad(self, system):
        """Проверка обработки некорректных СЛАУ

        :param system: СЛАУ в виде двумерного массива
        :return: None
        """
        with pytest.raises(np.linalg.LinAlgError):
            solve_system(system)


class TestPolyBuilder:
    """Тестирование функции построения полинома

    Поля:
        x_sym: sympy.symbols - символ икса
    Методы:
        test_good(list, list, int, tuple)
        test_bad(list, list int)
    """
    x_sym = sm.symbols("x")

    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree", "expected"), [
            ([1, 2, 3], [1, 1, 1], 1, (1 * x_sym, 0)),
            ([1, 2], [2, 5], 1, (3.0 * x_sym - 1.0, 0.1)),
            ([1, 5, 6, 7], [3, 5, 7, 9], 3,
             (-0.05 * x_sym ** 3 + 0.9 * x_sym ** 2 - 3.35 * x_sym + 5.5, 0.001))
        ]
    )
    def test_good(self, x_data, y_data, degree, expected):
        """Проверка корректности построенного полинома с указанной точностью

        :param x_data: список значений аргументов исходной функции
        :param y_data: список значений исходной функции
        :param degree: степень полинома
        :param expected: кортеж из выражения sympy.Expr и точности проверки
        :return: None
        """
        func_data = build_poly(x_data, y_data, degree)
        assert (func_data - expected[0]).subs(self.x_sym, 1) <= expected[1]

    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree"), [
            ([1, 2], [2, 5], 2)
        ]
    )
    def test_bad(self, x_data, y_data, degree):
        """Проверка обработки некорректных данных

        :param x_data: список значений аргументов исходной функции
        :param y_data: список значений исходной функции
        :param degree: степень полинома
        :return: None
        """
        with pytest.raises(np.linalg.LinAlgError):
            build_poly(x_data, y_data, degree)


class TestStdDev:
    """Тестирование функции вычисления стандартного отклонения

    Поля:
        x_sym: sympy.symbols - символ икса
    Методы:
        test_good(list, list, sympy.Expr, float)
        test_bad(list, list sympy.Expr, Exception)
    """
    x_sym = sm.symbols("x")

    @pytest.mark.parametrize(
        ("x_data", "y_data", "expr", "expected"), [
            ([1, 2, 3], [1, 1, 1], 2 * x_sym, 35),
            ([1, 2], [5, 0], 3 * x_sym, 40),
            ([0, sm.pi/2, sm.pi], [0, 1, 0], sm.sin(x_sym), 0)
        ]
    )
    def test_good(self, x_data, y_data, expr, expected):
        """ Проверка корректности вычисления стандартного отклонения

        :param x_data: список значений аргументов исходной функции
        :param y_data: список значений исходной функции
        :param expr: функция в виде sympy.Expr
        :param expected: значение стандартного отклонения
        :return: None
        """
        assert std_dev(x_data, y_data, expr) == expected

    @pytest.mark.parametrize(
        ("x_data", "y_data", "expr", "exception"), [
            ([1, 2], [2], 2 * x_sym, ValueError),
            ([2], [5, 4], 3 * x_sym, ValueError),
            ([3, 2], [5, 6], "3*x + 2", TypeError),
            ([3, 5], [2, 3], 1, TypeError)

        ]
    )
    def test_bad(self, x_data, y_data, expr, exception):
        """Проверка обработки некорректных данных

        :param x_data: список значений аргументов исходной функции
        :param y_data: список значений исходной функции
        :param expr: функция в виде sympy.Expr
        :param exception: ошибка, выдаваемая функцией
        :return: None
        """
        with pytest.raises(exception):
            std_dev(x_data, y_data, expr)
