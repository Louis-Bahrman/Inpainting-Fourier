#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils to extract signals and degrade them

"""
import glob
import numpy as np
import soundfile as sf
import utils.dsp
import utils.random
import utils.linalg


def import_audios(num_signals: int, len_audios: int, audios_directory: str, extract_from_different_audio_files: bool = True, silence_power_db: float = -50, normalize_chunks: bool = True) -> np.array:
    """
    Imports and splits audio files into chunks

    Parameters
    ----------
    num_signals : int
        Number of chunks to be extracted.
    len_audios : int
        length of each chunk to be extracted (in samples).
    audios_directory : str
        Directory where the audio files are stored.
    extract_from_different_audio_files : bool, optional
        Whether to select only one chunk of length `len_audios` per audio file.
        The default is True.
    silence_power_db : float, optional
        Minimal power such that a chunk is selected.
        Signals which power is lower than `silence_power_db` are ignored.
        The default is -50.
    normalize_chunks : bool, optional
        Whether to normalize chunks (set power to 1). The default is True.

    Raises
    ------
    RuntimeError
        If not enough audio files are found in `audios_directory`.

    Returns
    -------
    np.array
        extracted audios chunks.

    """
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


def degrade_signal(signal: np.array, missing_fraction: float) -> (np.array, np.array):
    """
    Returns a degraded copy of a signal in which some contiguous indices are zeroed
    The ratio between the number of contiguous indices which are zeroed `d`
    and the length of the signal `L` is d/L = `missing_fraction`

    Parameters
    ----------
    signal : np.array
        The signal to be degraded.
    missing_fraction : float
        Ratio of the number of contiguous indices to be zeroed
        over the total length of the signal.

    Returns
    -------
    degraded_signal : np.array
        Degraded signal.
    missing_indices : np.array
        contiguous indices which have been zeroed.

    """
    missing_indices = utils.random.random_interval(
        len(signal), int(round(missing_fraction*len(signal))))
    degraded_signal = signal.copy()
    degraded_signal[missing_indices] = 0
    return degraded_signal, missing_indices
