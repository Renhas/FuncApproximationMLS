"""
Модуль тестирования модуля линейной регрессии

Функции:
    test_eps1(list, float, float) - тестирование вычисления отклонения
"""
import pytest
from regression_algorythm import eps1_calculate


@pytest.mark.parametrize(
    ("y_value", "expected", "accuracy"), [
        ([0, 0, 0], 0, 1),
        ([1, 2, 3, 4], 5, 1),
        ([0, 1, 0], 0.66666, 0.00001)
    ]
)
def test_eps1(y_value: list, expected: float, accuracy: float):
    """Тестирование вычисления отклонения

    :param y_value: список значений
    :param expected: ожидаемый результат
    :param accuracy: точность сравнения
    """
    assert eps1_calculate(y_value) == pytest.approx(expected, abs=accuracy)
