#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Experiment 3 of the paper
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import utils.signal_extraction
import utils.dsp
import utils.random
from utils.inpainters.alternated_minimization import inpaint as inpaint_am
from utils.inpainters.convex_relaxation import inpaint as inpaint_cr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

AUDIO_DIRECTORY_PATH = './Data/'
FIGURE_PATH = './results/experiment_3.pdf'
MISSING_FRACTION = 0.25
L = 1024
MAGNITUDE_SNRS = [0, 5, 10, 15, 20, 25, 30, 35, 40]
NUM_SIGNALS = 100

signals = utils.signal_extraction.import_audios(
    NUM_SIGNALS, L, AUDIO_DIRECTORY_PATH)

sers_am = np.zeros((NUM_SIGNALS, len(MAGNITUDE_SNRS)))
sers_cr = np.zeros((NUM_SIGNALS, len(MAGNITUDE_SNRS)))
sers_cr_am = np.zeros((NUM_SIGNALS, len(MAGNITUDE_SNRS)))


iterator_signals = tqdm(
    signals, desc='Signal') if tqdm is not None else signals
iterator_magnitude_snrs = tqdm(
    MAGNITUDE_SNRS, desc='Magnitude SNR', leave=True) if tqdm is not None else MAGNITUDE_SNRS

for i, signal in enumerate(iterator_signals):
    exact_fourier_magnitudes = utils.dsp.fourier_magnitudes(signal)
    # Degrade the signal
    degraded_signal, missing_indices = utils.signal_extraction.degrade_signal(
        signal, MISSING_FRACTION)

    for j, magnitude_snr in enumerate(iterator_magnitude_snrs):
        erroneous_fourier_magnitudes = utils.random.awgn_capped(
            exact_fourier_magnitudes, magnitude_snr)

        # inpaint using the 3 methods
        inpainted_signal_am = inpaint_am(
            degraded_signal, erroneous_fourier_magnitudes, missing_indices)
        inpainted_signal_cr = inpaint_cr(
            degraded_signal, erroneous_fourier_magnitudes, missing_indices)
        inpainted_signal_cr_am = inpaint_am(
            inpainted_signal_cr, erroneous_fourier_magnitudes, missing_indices)

        # Evaluate
        sers_am[i, j] = utils.dsp.ser_db(inpainted_signal_am, signal)
        sers_cr[i, j] = utils.dsp.ser_db(inpainted_signal_cr, signal)
        sers_cr_am[i, j] = utils.dsp.ser_db(inpainted_signal_cr_am, signal)

# Plot and save figure
plt.rcParams.update({"font.size": 36,
                     'figure.dpi': 300,
                     'lines.linewidth': 3,
                     'lines.markersize': 16})

plt.figure(figsize=(14, 10))
ax = plt.subplot(111)
ax.plot(MAGNITUDE_SNRS, sers_am.mean(axis=0), label='AM', marker="s")
ax.plot(MAGNITUDE_SNRS, sers_cr.mean(axis=0), label='CR', marker="^")
ax.plot(MAGNITUDE_SNRS, sers_cr_am.mean(axis=0),
        label='CR + AM', marker="o")
ax.set_xlabel('Magnitude SNR (dB)')
ax.set_ylabel('Mean SER (dB)')
plt.legend()

if not os.path.isdir(os.path.dirname(FIGURE_PATH)):
    os.makedirs(os.path.dirname(FIGURE_PATH))
plt.savefig(FIGURE_PATH, bbox_inches='tight')
