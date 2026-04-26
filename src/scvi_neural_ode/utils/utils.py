from typing import Callable, Sequence

import numpy as np


def compfunc1d(func1d: Callable, *arr_list: Sequence[np.ndarray]):
    """
    Composition of functions that take only 2 arguments (e.g. np.union1d)
    across a 1-dimensional list of arrays of arbitrary length.

    Example: If function f only takes 2 arguments, then for arrays A, B, C, D
    compfunc1d(f, A, B, C, D) == f(A, f(B, f(C, D)))
    """
    res = arr_list[0]
    for j in range(1, len(arr_list)):
        res = func1d(res, arr_list[j])

    return res


def column_standardize_array(vals):
    """
    Standardize each column of 2-dimensional array `vals` to the [0, 1] interval.
    """
    if vals.ndim > 2:
        raise NotImplementedError
    minval, maxval = vals.min(0), vals.max(0)
    return (vals - minval) / (maxval - minval)


def row_standardize_array(vals):
    """
    Standardize each row of 2-dimensional array `vals` to the [0, 1] interval.
    """
    if vals.ndim != 2:
        raise NotImplementedError("Array must be 2-dimensional.")
    xmin, xmax = vals.min(1)[:, None], vals.max(1)[:, None]
    return (vals - xmin) / (xmax - xmin)
