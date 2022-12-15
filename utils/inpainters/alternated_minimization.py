import numpy as np
import utils.dsp

try:
    from tqdm.auto import trange
except ImportError:
    trange = None


def loss(degraded_signal, fourier_magnitudes, missing_indices):
    """
    Loss of the alternated minimization method, normalized by the length of the missing section of the signal.

    Parameters
    ----------
    degraded_signal : np.array
        degraded signal.
    fourier_magnitudes : np.array
        Supposedly known Fourier magnitudes.
    missing_indices : np.array
        indices of the missing samples of `degraded_signal`.

    Returns
    -------
    float
        loss of the inpainting problem.

    """
    return np.linalg.norm(utils.dsp.fourier_magnitudes(degraded_signal) - fourier_magnitudes) ** 2 / len(missing_indices) ** 2


def zero_init(degraded_signal, fourier_magnitudes, missing_indices):
    """
    Initialize a zeroed signal for the inpainting method.

    The output signal corresponds to the degraded signal whose missing indices are set to 0.

    Parameters
    ----------
    degraded_signal : np.array
        degraded signal.
    fourier_magnitudes : np.array
        Supposedly known Fourier magnitudes, provided just for compatibility with other initialization functions.
    missing_indices : np.array
        indices of the missing samples of `degraded_signal`.

    Returns
    -------
    signal_init : np.array
        Signal whose missing indices are set to 0, which can be used as init for the inpaint function.

    """
    signal_init = degraded_signal.copy()
    signal_init[missing_indices] = 0
    return signal_init


def inpaint(degraded_signal, fourier_magnitudes, missing_indices, n_iter_max=1000, improvement_threshold_stop=1e-10, num_last_iterations_stop=5, accumulator_stop=lambda a: np.max(np.abs(np.diff(a))), already_initialized=False, initialization_function=zero_init):
    """
    Inpaint using the alternated minimization method

    Parameters
    ----------
    degraded_signal : np.array
        signal to be inpainted.
    fourier_magnitudes : np.array
        Supposedly known Fourier magnitudes.
    missing_indices : np.array
        indices of the missing samples of `degraded_signal`.
    n_iter_max : int, optional
        Maximal number of iterations. The default is 1000.
    improvement_threshold_stop : float, optional
        Threshold for the improvement of the loss.
        If `accumulator_stop` applied to the last `num_last_iterations_stop` of the loss is greater than `improvement_threshold_stop`, the algorithm stops. The default is 1e-10.
    num_last_iterations_stop : int, optional
        Number of last iterations considered for the early stopping of the procedure (see above). The default is 5.
    accumulator_stop : function, optional
        Accumulator on the loss of the last iterations (see above). The default is lambda a: np.max(np.abs(np.diff(a))) which corresponds to the max variation of the loss over the last iterations.
    already_initialized : bool, optional
        Whether `degraded_signal` has already been initialized. Should be set to True when this algorithm is used in conjunction with the convex relaation method. The default is False.
    initialization_function : function, optional
        Initialization function. The default is zero_init.

    Returns
    -------
    inpainted_signal : np.array
        Inpainted signal.

    """
    if not already_initialized:
        inpainted_signal = initialization_function(
            degraded_signal, fourier_magnitudes, missing_indices)
    else:
        inpainted_signal = degraded_signal.copy()
    last_scores = np.full(num_last_iterations_stop, np.inf)
    iterator = trange(n_iter_max, desc='Alternated Minimization iteration',
                      leave=False) if trange is not None else range(n_iter_max)
    for iteration in iterator:
        # We compute the stop criterion
        last_scores[iteration % num_last_iterations_stop] = loss(
            inpainted_signal, fourier_magnitudes, missing_indices)
        if (iteration > num_last_iterations_stop
                and accumulator_stop(np.roll(last_scores, - (iteration % num_last_iterations_stop))) < improvement_threshold_stop):
            # if hasattr(iterator, 'clear'):
            #     iterator.close()
            break
        # Each iteration
        u = np.exp(1j * np.angle(np.fft.fft(inpainted_signal)))
        inpainted_signal[missing_indices] = np.fft.ifft(
            fourier_magnitudes * u).real[missing_indices]
    return inpainted_signal
