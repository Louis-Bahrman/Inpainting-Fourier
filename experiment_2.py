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
FIGURE_PATH = './results/experiment_2.pdf'
MISSING_FRACTIONS = np.arange(0.05, 0.55, 0.05).round(2)
L = [128, 256, 512, 1024, 2048, 4096]
NUM_SIGNALS = 100
SER_PERFECT_RECONSTRUCTION_DB = 20

# Import signals and compute their Fourier Magnitudes
sers_cr_am = np.zeros(len(L), NUM_SIGNALS, len(MISSING_FRACTIONS))

iterator_l = tqdm(L, desc='Total signal length') if tqdm is not None else L
iterator_missing_fraction = tqdm(
    MISSING_FRACTIONS, desc='Missing fraction', leave=True) if tqdm is not None else MISSING_FRACTIONS

for i, l in enumerate(iterator_l):
    # Extract signals
    signals = utils.signal_extraction.import_audios(
        NUM_SIGNALS, l, AUDIO_DIRECTORY_PATH)

    iterator_signals = tqdm(signals) if tqdm is not None else signals
    for j, signal in iterator_signals:

        fourier_magnitudes = utils.dsp.fourier_magnitudes(signal)

        for k, missing_fraction in enumerate(iterator_missing_fraction):
            # Degrade the signal
            degraded_signal, missing_indices = utils.signal_extraction.degrade_signal(
                signal, missing_fraction)

            # inpaint using the CR + AM Method
            inpainted_signal_cr = inpaint_cr(
                degraded_signal, fourier_magnitudes, missing_indices)
            inpainted_signal_cr_am = inpaint_am(
                inpainted_signal_cr, fourier_magnitudes, missing_indices)

            # Evaluate
            sers_cr_am[i, j, k] = utils.dsp.ser_db(
                inpainted_signal_cr_am, signal)

proba_perfect_reconstruction_cr_am = (
    sers_cr_am > SER_PERFECT_RECONSTRUCTION_DB).mean(axis=1)

# Plot figure
plt.rcParams.update({"font.size": 36,
                     'figure.dpi': 300,
                     'lines.linewidth': 3,
                     'lines.markersize': 16})

plt.figure(figsize=(14, 10))
plt.imshow(proba_perfect_reconstruction_cr_am.T, cmap='Blues', origin='lower',
           aspect='auto', extent=[0, len(L), 0, len(missing_fraction)], vmin=0, vmax=1)
# plt.plot((len(N)+1)*[1/2*(len(deletedFractions)-1)],color='tab:red',linewidth=2)
plt.colorbar(format=PercentFormatter(xmax=1.0, decimals=0), ax=plt.gca())
plt.xticks(
    ticks=0.5+np.arange(len(L)),
    labels=L,
    rotation="vertical",
)
plt.xlabel('Total signal length')

plt.yticks(ticks=0.5+np.arange(len(missing_fraction)),
           labels=[f"{missing_fraction:.0%}" for missing_fraction in MISSING_FRACTIONS])
plt.ylabel('Missing fraction')

if not os.path.isdir(os.path.dirname(FIGURE_PATH)):
    os.makedirs(os.path.dirname(FIGURE_PATH))
plt.savefig(FIGURE_PATH, bbox_inches='tight')
