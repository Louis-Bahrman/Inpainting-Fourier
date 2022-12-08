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

AUDIO_DIRECTORY_PATH = '../Data/LibriSpeech'
FIGURE_PATH = './results/experiment_1.pdf'
MISSING_FRACTIONS = np.arange(0.05, 0.55, 0.05).round(2)
L = 1024
NUM_SIGNALS = 2
SER_PERFECT_RECONSTRUCTION_DB = 20

# Import signals and compute their Fourier Magnitudes
signals = utils.signal_extraction.import_audios(
    NUM_SIGNALS, L, AUDIO_DIRECTORY_PATH)

fourier_magnitudes = [utils.dsp.fourier_magnitudes(
    signal) for signal in signals]

proba_perfect_reconstruction_am = []
proba_perfect_reconstruction_cr = []
proba_perfect_reconstruction_cr_am = []

for missing_fraction in MISSING_FRACTIONS:
    # Degrade the signals
    degraded_signals_and_missing_indices = [utils.signal_extraction.degrade_signal(
        signal, missing_fraction) for signal in signals]

    # reorganize data into several inpainting problems
    # which consists of triplets (degraded signals, fourier magnitudes, missing indices)
    inpainting_problems = [(ds, fm, mi) for ((ds, mi), fm) in zip(
        degraded_signals_and_missing_indices, fourier_magnitudes)]

    # Run and evaluate the three methods:
    # AM
    inpainted_signals_am = [inpaint_am(*ip) for ip in inpainting_problems]
    sers_am = np.array([utils.dsp.ser_db(es, os)
                        for es, os in zip(inpainted_signals_am, signals)])
    proba_perfect_reconstruction_am.append(
        (sers_am > SER_PERFECT_RECONSTRUCTION_DB).mean())

    # CR
    inpainted_signals_cr = [inpaint_cr(*ip) for ip in inpainting_problems]
    sers_cr = np.array([utils.dsp.ser_db(es, os)
                        for es, os in zip(inpainted_signals_cr, signals)])
    proba_perfect_reconstruction_cr.append(
        (sers_cr > SER_PERFECT_RECONSTRUCTION_DB).mean())

    # CR + AM
    inpainted_signals_cr_am = [inpaint_am(*(init_signal, *ip[1:]), already_initialized=True)
                               for init_signal, ip in zip(inpainted_signals_cr, inpainting_problems)]
    sers_cr_am = np.array([utils.dsp.ser_db(es, os)
                           for es, os in zip(inpainted_signals_cr_am, signals)])
    proba_perfect_reconstruction_cr_am.append(
        (sers_cr_am > SER_PERFECT_RECONSTRUCTION_DB).mean())


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
