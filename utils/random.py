#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random utils

"""

import numpy as np
from utils.dsp import db_to_power, signal_power_db

rng = np.random.default_rng()


def random_interval(high: int, length: int) -> np.array:
    """
    Generate a random integer range of length `length`, and which maximum is lower than `high`

    Parameters
    ----------
    high : int
        Upper bound of the generated range.
    length : int
        length of the range.

    Returns
    -------
    np.array
        range.

    """
    if high == length:
        return np.asarray(range(high))
    begin = rng.integers(high - length)
    return np.arange(begin, begin + length)


def awgn(original_signal, target_snr_db):
    """
    Adds white gaussian noise whose variance is adjusted to fit a given signal-to-noise ratio

    Parameters
    ----------
    original_signal : np.array
        original signal.
    target_snr_db : float
        target signal-to-noise ratio (in db).

    Returns
    -------
    noisy_signal : np.array
        noisy signal s.t. snr_db(noisy_signal, original_signal) = target_snr_db.

    """
    if target_snr_db == np.inf:
        return original_signal.copy()
    if all(original_signal == 0) or target_snr_db == -np.inf:
        return rng.normal(0, 1, len(original_signal))
    Px_db = signal_power_db(original_signal)
    var = db_to_power(Px_db - target_snr_db)
    std = np.sqrt(var)
    noise = rng.normal(0, std, len(original_signal))
    noisy_signal = noise + original_signal
    return noisy_signal


def awgn_capped(original_signal: np.array, target_snr_db: float, cap=0) -> np.array:
    """
    Adds white gaussian noise whose variance is adjusted to fit a given signal-to-noise ratio
    and which minimum is higher than a given threshold

    Parameters
    ----------
    original_signal : np.array
        original signal.
    target_snr_db : float
        target signal-to-noise ratio (in db).
    cap : TYPE, optional
        minimum of the resulting degraded signal. The default is 0.

    Returns
    -------
    np.array
        noisy signal s.t. `snr_db(noisy_signal, original_signal) = target_snr_db`
        and `min(noisy_signal) >= cap`

    """
    return np.maximum(cap*np.ones_like(original_signal), awgn(original_signal, target_snr_db))
