#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear algebra utils
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def H(A: np.array) -> np.array:
    """
    Compute the hermitian transpose of a matrix

    Parameters
    ----------
    A : np.array
        matrix to be transposed.

    Returns
    -------
    np.array
        Hermitian transpose of `A`.

    """
    return A.conj().T


def normalize(x: np.array, **kwargs) -> np.array:
    """
    Normalize an array by setting its norm to 1.

    Map the null vector to itself

    Parameters
    ----------
    x : np.array
        array to be normalized.
    **kwargs : TYPE
        np.linalg.norm kwargs.

    Returns
    -------
    np.array
        normalized array, s.t. `np.linalg.norm(normalized_array, **kwargs) == 1`
            or `all(x == 0)`.

    """
    norm = np.linalg.norm(x, **kwargs)
    return x / norm if norm != 0 else x


def greatest_eigenvector(Y: np.array, assume_hermitian: bool = False) -> np.array:
    """
    Eigenvector associated with the eigenvalue of greatest magnitude.

    Parameters
    ----------
    Y : cupy.array | np.array
        matrix.
    assume_hermitian : bool, optional
        whether Y is hermitian. The default is False.

    Returns
    -------
    greatestEigVect : np.array
        eigenvector associated with the eigenvalue of greatest magnitude of Y.

    """
    if cp is None:
        xp = np
    else:
        xp = cp.get_array_module(Y)
    if assume_hermitian:
        eigVal, eigVect = xp.linalg.eigh(Y)
    else:
        eigVal, eigVect = xp.linalg.eig(Y)
    greatestEigVect = eigVect[:, np.argmax(abs(eigVal))]
    return greatestEigVect
    # return scipy.linalg.eigsh(Y, k=1, which="LM")[1][:, 0]
