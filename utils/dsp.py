#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def power_to_db(x):
    if x == 0:
        return -np.inf
    return 10 * np.log10(x)


def db_to_power(x_db):
    return np.power(10, x_db / 10)


def signal_power_db(signal):
    return power_to_db((np.abs(signal)**2).mean())


def fourier_magnitudes(signal):
    """
    Fourier magnitudes of a signal

    Parameters
    ----------
    signal : numpy.array
        signal.

    Returns
    -------
    numpy.array
        Fourier magnitudes.

    """
    return np.abs(np.fft.fft(signal))


def ser_db(estimated_signal, original_signal):
    """
    Compute the signal-to-error ratio (in db)

    Parameters
    ----------
    estimated_signal : np.array
        estimated signal.
    original_signal : np.array
        original signal.

    Returns
    -------
    int
        Signal-to-error ratio (in db).

    """
    if np.linalg.norm(estimated_signal - original_signal) == 0:
        return np.inf
    return power_to_db(np.linalg.norm(original_signal) ** 2 / np.linalg.norm(estimated_signal - original_signal) ** 2)
