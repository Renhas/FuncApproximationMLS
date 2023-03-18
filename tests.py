import pytest
from Algorythm import coefs_calculate


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
    
