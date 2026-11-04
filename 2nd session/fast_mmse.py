"""Exact local MMSE algebra exploiting the CP-covered time-domain FIR sparsity.

No delay/Doppler coupling is truncated. Sparse time-domain inversion and a
Schur complement give the same native data normal-system solution/self-gain.
The regular dense solver remains an accuracy-checked fallback.
"""

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

_dependencies = str(Path(__file__).with_name(".runtime-deps"))
if _dependencies not in sys.path:
    sys.path.insert(0, _dependencies)
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu


@lru_cache(maxsize=4)
def inverse_fourier_basis(length):
    return np.fft.ifft(np.eye(length, dtype=complex), axis=0, norm="ortho")


def time_matrix(channel, block):
    q = np.arange(channel.length)
    rows = np.tile(q, len(channel.delays))
    columns = np.concatenate([(q-delay) % channel.length for delay in channel.delays])
    values = np.concatenate([channel.taps[:, block, j] for j in range(len(channel.delays))])
    return csc_matrix((values, (rows, columns)), shape=(channel.length,)*2)


def certified_full_rank(channel, block):
    """Strict row diagonal dominance is a sufficient nonsingularity certificate."""
    if 0 not in channel.delays:
        return False
    magnitudes = abs(channel.taps[:, block, :])
    main = magnitudes[:, channel.delays.index(0)]
    return bool(np.all(2*main > magnitudes.sum(axis=1)))


def _native_to_time(channel, values):
    if channel.scheme == "AFDM":
        return channel.chirp1[:, None]*np.fft.ifft(channel.chirp2[:, None]*values, axis=0, norm="ortho")
    return np.fft.ifft(values, axis=0, norm="ortho")


def _time_to_native(channel, values):
    if channel.scheme == "AFDM":
        return channel.chirp2.conj()[:, None]*np.fft.fft(channel.chirp1.conj()[:, None]*values, axis=0, norm="ortho")
    return np.fft.fft(values, axis=0, norm="ortho")


def _fourier_diagonal(channel, inverse_time):
    if channel.scheme == "AFDM":
        inverse_time = channel.chirp1.conj()[:, None]*inverse_time*channel.chirp1[None, :]
    q = np.arange(channel.length)
    cyclic_sums = np.sum(inverse_time[q[:, None], (q[:, None]-q[None, :]) % channel.length], axis=0)
    return (np.fft.fft(cyclic_sums)/channel.length).real


def solve_local(problem):
    channel, model, lam = problem.model.channel, problem.model, problem.regularization
    values, gains = np.zeros(problem.data_count, dtype=complex), np.empty(problem.data_count)
    for block in range(problem.grid_shape[1]):
        selected = model.block_data_slice(block)
        data = np.flatnonzero(model.data_mask[:, block])
        if lam == 0 and not certified_full_rank(channel, block):
            a = model.block_matrix(block)
            u, singular, vh = np.linalg.svd(a, full_matrices=False)
            retained = singular > np.finfo(float).eps*max(a.shape)*singular[0]
            values[selected] = vh[retained].conj().T@((u[:, retained].conj().T@problem.observations[:, block])/singular[retained])
            gains[selected] = np.sum(abs(vh[retained])**2, axis=0)
            continue
        if data.size <= channel.length//2:
            a = model.block_matrix(block)
            gram = a.conj().T@a
            factor = cho_factor(gram+lam*np.eye(data.size), lower=True, check_finite=False)
            if lam == 0:
                values[selected] = cho_solve(factor, problem.rhs[selected], check_finite=False)
                gains[selected] = 1.
            else:
                solved = cho_solve(factor, np.column_stack([problem.rhs[selected], gram]), check_finite=False)
                values[selected], gains[selected] = solved[:, 0], np.diag(solved[:, 1:]).real
            continue
        h = time_matrix(channel, block)
        factor = splu((h.conj().T@h+lam*eye(channel.length, format="csc")).tocsc())
        native_rhs = np.zeros((channel.length, 1), dtype=complex)
        native_rhs[data, 0] = problem.rhs[selected]
        full_solution = _time_to_native(channel, factor.solve(_native_to_time(channel, native_rhs)))[:, 0]
        other = np.flatnonzero(~model.data_mask[:, block])
        correction = np.zeros(data.size)
        if other.size:
            basis = inverse_fourier_basis(channel.length)[:, other]
            if channel.scheme == "AFDM":
                basis = channel.chirp1[:, None]*basis*channel.chirp2[other][None, :]
            k_columns = _time_to_native(channel, factor.solve(basis))
            kpp = k_columns[other]
            kpp = (kpp+kpp.conj().T)/2
            chol = cho_factor(kpp, lower=True, check_finite=False)[0]
            w = solve_triangular(chol, k_columns[data].conj().T, lower=True, check_finite=False)
            v = solve_triangular(chol, full_solution[other], lower=True, check_finite=False)
            values[selected] = full_solution[data]-w.conj().T@v
            correction = np.sum(abs(w)**2, axis=0)
        else:
            values[selected] = full_solution[data]
        if lam == 0:
            gains[selected] = 1.
        else:
            inverse_time = factor.solve(np.eye(channel.length, dtype=complex))
            diagonal = _fourier_diagonal(channel, inverse_time)[data]-correction
            gains[selected] = 1-lam*diagonal
            if np.any(gains[selected] <= np.sqrt(np.finfo(float).eps)):
                # Avoid subtractive loss for very small self-response. Recompute
                # the same gain from its stable dense expression, without floors.
                a = model.block_matrix(block)
                gram = a.conj().T@a
                gains[selected] = np.diag(np.linalg.solve(gram+lam*np.eye(data.size), gram)).real
    return values, gains


def otfs_gain(problem):
    model, channel, lam = problem.model, problem.model.channel, problem.regularization
    rows = np.flatnonzero(model.data_mask[:, 0])
    result = np.zeros(rows.size)
    for block in range(channel.blocks):
        if lam == 0:
            if not certified_full_rank(channel, block):
                return None
            gain = np.ones(rows.size)
        else:
            a = time_matrix(channel, block)[:, rows]
            gram = a.conj().T@a
            factor = splu((gram+lam*eye(rows.size, format="csc")).tocsc())
            gain = 1-lam*np.diag(factor.solve(np.eye(rows.size, dtype=complex))).real
            if np.any(gain <= np.sqrt(np.finfo(float).eps)):
                dense = gram.toarray()
                gain = np.diag(np.linalg.solve(dense+lam*np.eye(rows.size), dense)).real
        result += gain/channel.blocks
    return np.broadcast_to(result[:, None], (rows.size, channel.blocks)).reshape(-1, order="F").copy()
