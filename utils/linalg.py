#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:43:57 2022

@author: louis
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def H(A):
    return A.conj().T


def normalize(x, **kwargs):
    norm = np.linalg.norm(x, **kwargs)
    return x / norm if norm != 0 else x


def greatest_eigenvector(Y, assume_hermitian=False):
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
