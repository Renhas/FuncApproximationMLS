import sympy as sm
import numpy as np


def coefs_calculate(x:list, y:list, m:int) -> list:
    if len(x) != len(y):
        raise ValueError("x and y should have same length")
    if m < 1:
        raise ValueError("m should be >=1 ")
    coefs = []
    for row in range(m+1):
        coefs.append([])
        for column in range(m+1):
            new_x = np.array(x)
            new_x = pow(new_x, column + row)
            coefs[row].append(new_x.sum())
        new_x = pow(np.array(x), row)
        new_y = np.array(y)
        coefs[row].append((new_y * new_x).sum())
    return coefs


def solve_system(system):
    S = np.array(system)
    m, n = S.shape
    A = S[:, 0:n-1]
    b = S[:, n-1]
    return np.linalg.solve(A, b)
