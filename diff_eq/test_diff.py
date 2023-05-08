"""
Тестирование методов приближенного решения СДУ

Функции:
    test_func - тестирование СДУ
    test_algorythm - тестирование алгоритмов
"""
import pytest
from diff_eq.algorythm import func1, func2, Euler, RungeKutta


@pytest.mark.parametrize(
    ("x_val", "y_val", "func", "expected", "acc"), [
        (0, [5, 3, 7], func1, 3, 0),
        (0, [4, 3, 2], func2, -0.01, 0.01),
        pytest.param(0, [5], func1, 0, 0,
                     marks=pytest.mark.xfail(strict=True)),
    ]
)
def test_func(x_val, y_val, func, expected, acc):
    """Тестирование СДУ

    :param x_val: значения x
    :param y_val: вектор значений y
    :param func: функция для тестирования
    :param expected: ожидаемое значение
    :param acc: точность
    """
    assert func(x_val, y_val) == pytest.approx(expected, abs=acc)


@pytest.mark.parametrize(
    ("algorythm", "params", "functions", "expected", "acc"), [
        (Euler, [[0.5], 0.5], (func2,),
         [[0, 0.5, 1], [0.5, 0.495, 0.491]],
         0.001),
        (RungeKutta, [[0], 1], (func2, ),
         [[0, 1], [0, -0.006884]],
         0.001)
    ]
)
def test_algorythm(algorythm, params, functions, expected, acc):
    """Тестирование алгоритма

    :param algorythm: алгоритм
    :param params: параметры алгоритма
    :param functions: СДУ
    :param expected: ожидаемые значения x и y
    :param acc: точность
    """
    x_res = []
    y_res = []

    def steps_check(x_val, y_val):
        """Сохранение промежуточных данных"""
        x_res.append(x_val)
        y_res.append(*y_val)

    res = algorythm(functions).run(*params, action=steps_check)
    x_res.append(res[0])
    y_res.append(*res[1])

    assert x_res == pytest.approx(expected[0], abs=acc)
    assert y_res == pytest.approx(expected[1], abs=acc)
