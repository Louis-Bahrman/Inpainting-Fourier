import scipy
import numpy as np
from utils.linalg import H, greatest_eigenvector

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from tqdm.auto import trange
except ImportError:
    trange = None

MIN_L_TO_USE_CUDA = 512
DTYPE_COMPLEX_CUDA = np.csingle


def reliable_indices(missing_indices, total_length):
    return np.array(list(set(range(total_length)) - set(missing_indices)))
    # return np.setdiff1d(range(total_length), missing_indices, assume_unique=True)


def compute_m(degraded_signal, fourier_magnitudes, missing_indices):
    L = len(fourier_magnitudes)
    ri = reliable_indices(missing_indices, L)
    phi = scipy.linalg.dft(L)
    phi_H = 1/L*H(phi)
    phi_v_bar = phi[:, missing_indices]
    phi_v_bar_H = phi_H[missing_indices, :]
    block1 = (phi_v_bar@phi_v_bar_H - np.identity(L)
              ) @ np.diag(fourier_magnitudes)
    phi_v = phi[:, ri]
    x_v = degraded_signal[ri]
    block2 = phi_v @ x_v[:, np.newaxis]
    m = np.block([block1, block2])
    return m


def inpaint(degraded_signal, fourier_magnitudes, missing_indices, nu=0, n_iter=10):
    L = len(fourier_magnitudes)
    use_cuda = cp is not None and len(fourier_magnitudes) > MIN_L_TO_USE_CUDA
    if use_cuda:
        xp = cp
    else:
        xp = np

    # Initialize M and U
    m = compute_m(degraded_signal, fourier_magnitudes, missing_indices)
    M = H(m) @ m
    if use_cuda:
        M = cp.asarray(M, dtype=DTYPE_COMPLEX_CUDA)
    U = xp.identity(len(fourier_magnitudes) + 1, dtype=M.dtype)

    # Iterate until max number of iterations is reached
    outer_iterator = trange(n_iter, desc='Block coordinate descent outer iteration',
                            leave=False) if trange is not None else range(n_iter)
    for _ in outer_iterator:
        for k in range(L+1):
            kc = reliable_indices([k], L+1)
            z = U[kc, :][:, kc] @ M[kc, k]
            gamma = (H(z)@M[kc, k]).item()
            gamma = gamma.real
            if gamma > 0:
                U[kc, k] = -xp.sqrt((1 - nu) / gamma) * z
                U[k, kc] = (U[kc, k]).conj()
            else:
                U[kc, k] = U[k, kc] = 0

    # Post-process U to get the missing part of the signal
    u = greatest_eigenvector(U, assume_hermitian=True)
    u = u[:-1]/(u[-1] + 1e-15)

    # Free memory
    del M, U, z, gamma
    if use_cuda:
        # u should also be on the GPU so we move it to the CPU
        u_np = cp.asnumpy(u)
        del u
        cp.get_default_memory_pool().free_all_blocks()
        u = u_np

    restaured_signal = degraded_signal.copy()
    restaured_signal[missing_indices] = scipy.fft.ifft(
        fourier_magnitudes * u).real[missing_indices]
    return restaured_signal
