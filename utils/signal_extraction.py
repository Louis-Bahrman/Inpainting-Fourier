#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import soundfile as sf
import utils.dsp
import utils.random
import utils.linalg

def import_audios(num_signals, len_audios, audios_directory, extract_from_different_audio_files=True, silence_power_db=-50, normalize_chunks=True):
    all_files_iterator = glob.iglob(audios_directory + '**/**', recursive=True)
    audio_chunks = []
    while len(audio_chunks) < num_signals:
        # Extract a new audio file
        try:
            tentative_path = next(all_files_iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Not enough audio files in {audios_directory}, \
                    try setting extract_from_different_audios to False"
            ) from exc
        try:
            full_audio_signal, _ = sf.read(tentative_path)
        except sf.LibsndfileError:
            # The file is not an audio, so we skip it
            pass
        else:
            # Divide each audio file in several excerpts of given length
            num_chunks_full_audio_signal = len(full_audio_signal) // len_audios
            if num_chunks_full_audio_signal > 0:
                full_audio_signal = full_audio_signal[utils.random.random_interval(
                    len(full_audio_signal), num_chunks_full_audio_signal * len_audios)]
                chunks_current_file = np.split(
                    full_audio_signal, num_chunks_full_audio_signal)
                # Chunks of too low power are removed
                non_silent_chunks = [
                    chunk for chunk in chunks_current_file
                    if utils.dsp.signal_power_db(chunk) > silence_power_db
                ]
                if extract_from_different_audio_files:
                    # If we need to extract from different files, \
                    # we select only one chunk per audio file
                    selected_chunks = [
                        utils.random.rng.choice(non_silent_chunks)]
                audio_chunks.extend(selected_chunks)
                # We ensure we didn't add too many audios at once
                audio_chunks = audio_chunks[:num_signals]
    # assert len(audio_chunks) == nSignalsWithSameLength
    # assert all(
    #     [signalPower_db(chunk) > silencePower_db for chunk in chunksN]
    # ), f"some signals are only silence (P_x < {silencePower_db})"
    if normalize_chunks:
        audio_chunks = [utils.linalg.normalize(
            chunk) for chunk in audio_chunks]
    return np.asarray(audio_chunks)


def degrade_signal(signal, missing_fraction):
    missing_indices = utils.random.random_interval(
        len(signal), int(round(missing_fraction*len(signal))))
    degraded_signal = signal.copy()
    degraded_signal[missing_indices] = 0
    return degraded_signal, missing_indices
