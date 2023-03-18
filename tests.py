import pytest
import numpy as np
from algorythm import *


class TestCoefs:
    @pytest.mark.parametrize(
        ("x", "y", "m", "matrix"), [
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
    def test_good(self, x, y, m, matrix):
        assert coefs_calculate(x, y, m) == matrix

    @pytest.mark.parametrize(
        ("x", "y", "m"), [
            ([1, 2], [2], 1),
            ([1], [1, 2], 2),
            ([1], [1], -1),
            ([1, 3, 5], [3, 6, 7], 0)
        ]
    )
    def test_bad(self, x, y, m):
        with pytest.raises(ValueError):
            coefs_calculate(x, y, m)


class TestSolver:
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
        with pytest.raises(np.linalg.LinAlgError):
            solve_system(system)


class TestPolyBuilder:
    x = sm.symbols("x")

    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree", "expected"), [
            ([1, 2, 3], [1, 1, 1], 1, 1),
            ([1, 2], [2, 5], 1, 3.0000000000000027*x - 1.0000000000000047),
            ([1, 5, 6, 7], [3, 5, 7, 9], 3, -0.04999999999984518*x**3 + 0.8999999999979575*x**2
             - 3.3499999999924457*x + 5.4999999999942775)
        ]
    )
    def test_good(self, x_data, y_data, degree, expected):
        assert build_poly(x_data, y_data, degree) == expected

    @pytest.mark.parametrize(
        ("x_data", "y_data", "degree"), [
            ([1, 2], [2, 5], 2)
        ]
    )
    def test_bad(self, x_data, y_data, degree):
        with pytest.raises(np.linalg.LinAlgError):
            build_poly(x_data, y_data, degree)


class TestStdDev:
    x = sm.symbols("x")

    @pytest.mark.parametrize(
        ("x_data", "y_data", "expr", "expected"), [
            ([1, 2, 3], [1, 1, 1], 2*x, 35),
            ([1, 2], [5, 0], 3*x, 40),
            ([0, sm.pi/2, sm.pi], [0, 1, 0], sm.sin(x), 0)
        ]
    )
    def test_good(self, x_data, y_data, expr, expected):
        assert std_dev(x_data, y_data, expr) == expected

    @pytest.mark.parametrize(
        ("x_data", "y_data", "expr", "exception"), [
            ([1, 2], [2], 2*x, ValueError),
            ([2], [5, 4], 3*x, ValueError),
            ([3, 2], [5, 6], "3*x + 2", TypeError),
            ([3, 5], [2, 3], 1, TypeError)

        ]
    )
    def test_bad(self, x_data, y_data, expr, exception):
        with pytest.raises(exception):
            std_dev(x_data, y_data, expr)

