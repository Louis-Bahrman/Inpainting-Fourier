#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import utils.signal_extraction
import utils.dsp
from utils.inpainters.alternated_minimization import inpaint as inpaint_am
from utils.inpainters.convex_relaxation import inpaint as inpaint_cr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

AUDIO_DIRECTORY_PATH = './Data/'
FIGURE_PATH = './results/experiment_1.pdf'
MISSING_FRACTIONS = np.arange(0.05, 0.55, 0.05).round(2)
L = 1024
NUM_SIGNALS = 2
SER_PERFECT_RECONSTRUCTION_DB = 20

# Import signals and compute their Fourier Magnitudes
signals = utils.signal_extraction.import_audios(
    NUM_SIGNALS, L, AUDIO_DIRECTORY_PATH)

sers_am = np.zeros((NUM_SIGNALS, len(MISSING_FRACTIONS)))
sers_cr = np.zeros((NUM_SIGNALS, len(MISSING_FRACTIONS)))
sers_cr_am = np.zeros((NUM_SIGNALS, len(MISSING_FRACTIONS)))

iterator_signals = tqdm(signals, desc='Signal') if tqdm is not None else signals
iterator_missing_fraction = tqdm(
    MISSING_FRACTIONS, desc='Missing fraction', leave=True) if tqdm is not None else MISSING_FRACTIONS

for i, signal in enumerate(iterator_signals):
    fourier_magnitudes = utils.dsp.fourier_magnitudes(signal)
    for j, missing_fraction in enumerate(iterator_missing_fraction):
        # Degrade the signal
        degraded_signal, missing_indices = utils.signal_extraction.degrade_signal(
            signal, missing_fraction)

        # inpaint using the 3 methods
        inpainted_signal_am = inpaint_am(
            degraded_signal, fourier_magnitudes, missing_indices)
        inpainted_signal_cr = inpaint_cr(
            degraded_signal, fourier_magnitudes, missing_indices)
        inpainted_signal_cr_am = inpaint_am(
            inpainted_signal_cr, fourier_magnitudes, missing_indices)

        # Evaluate
        sers_am[i, j] = utils.dsp.ser_db(inpainted_signal_am, signal)
        sers_cr[i, j] = utils.dsp.ser_db(inpainted_signal_cr, signal)
        sers_cr_am[i, j] = utils.dsp.ser_db(inpainted_signal_cr_am, signal)


proba_perfect_reconstruction_am = (
    sers_am > SER_PERFECT_RECONSTRUCTION_DB).mean(axis=0)
proba_perfect_reconstruction_cr = (
    sers_cr > SER_PERFECT_RECONSTRUCTION_DB).mean(axis=0)
proba_perfect_reconstruction_cr_am = (
    sers_cr_am > SER_PERFECT_RECONSTRUCTION_DB).mean(axis=0)

# Plot and save figure
plt.rcParams.update({"font.size": 36,
                     'figure.dpi': 300,
                     'lines.linewidth': 3,
                     'lines.markersize': 16})

plt.figure(figsize=(14, 10))
ax = plt.subplot(111)
ax.plot(MISSING_FRACTIONS, proba_perfect_reconstruction_am, label='AM', marker="s")
ax.plot(MISSING_FRACTIONS, proba_perfect_reconstruction_cr, label='CR', marker="^")
ax.plot(MISSING_FRACTIONS, proba_perfect_reconstruction_cr_am,
        label='CR + AM', marker="o")
ax.set_xlabel('Missing fraction')
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylabel('Probability of recovery')
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

plt.legend()

if not os.path.isdir(os.path.dirname(FIGURE_PATH)):
    os.makedirs(os.path.dirname(FIGURE_PATH))
plt.savefig(FIGURE_PATH, bbox_inches='tight')
