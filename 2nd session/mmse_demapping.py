"""Standard estimated-channel MMSE gain-aware Gray 16-QAM hard demapping.

Gain is diag(G*A_hat), with G the selected nominal MMSE/zero-noise LS filter.
It uses estimated channels and measured variance only, never a true data fit.
"""

import numpy as np

from qam16 import hard_demodulate


def _self_gain(matrix, regularization):
    if regularization > 0:
        gram = matrix.conj().T@matrix
        normal = gram.copy()
        normal.flat[::normal.shape[0]+1] += regularization
        gain = np.diag(np.linalg.solve(normal, gram)).real.copy()
    else:
        # The same machine-precision rank rule as numpy.linalg.lstsq(rcond=None).
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        cutoff = (np.finfo(float).eps*max(matrix.shape)*singular[0]
                  if singular.size else 0.)
        retained = singular > cutoff
        if np.count_nonzero(retained) == matrix.shape[1]:
            gain = np.ones(matrix.shape[1])
        else:
            gain = np.sum(abs(vh[retained])**2, axis=0)
    if not np.isfinite(gain).all() or np.any(gain < 0):
        raise RuntimeError("The computed MMSE self-gain is invalid.")
    return gain


def effective_symbol_gain(problem):
    """Return the exact nominal self-gain in the original data-cell order.

    For the selected OTFS mask, every Doppler column has the same delay rows.
    Its unitary inverse-Zak time mapping acts independently on each delay row.
    Therefore diag(G_DD*A_DD)[q,k] is the mean over slots of the corresponding
    time-domain data self-gain at row q. This exact unitary identity avoids a
    dense DD inverse, stochastic probes or any truncation. The data solve itself
    remains full-frame CG/LSQR under the existing numerical policy.
    """
    model = problem.model
    if problem.scheme != "OTFS":
        result = np.empty(problem.data_count)
        for block in range(problem.grid_shape[1]):
            result[model.block_data_slice(block)] = _self_gain(model.block_matrix(block),
                                                              problem.regularization)
        return result
    mask = model.data_mask
    if not np.array_equal(mask, np.broadcast_to(mask[:, :1], mask.shape)):
        raise ValueError("The selected OTFS data mask must share delay rows across Doppler columns.")
    channel = model.channel
    rows = np.flatnonzero(mask[:, 0])
    try:
        from fast_mmse import otfs_gain
        accelerated = otfs_gain(problem)
        if accelerated is not None and np.isfinite(accelerated).all() and np.all(accelerated >= 0):
            return accelerated
    except (ImportError, RuntimeError, np.linalg.LinAlgError):
        pass
    length, blocks = problem.grid_shape
    grid_gain = np.zeros(rows.size)
    receive_rows = np.arange(length)
    for block in range(blocks):
        time_matrix = np.zeros((length, length), dtype=complex)
        for column, delay in enumerate(channel.delays):
            time_matrix[receive_rows, (receive_rows-delay) % length] += channel.taps[:, block, column]
        grid_gain += _self_gain(time_matrix[:, rows], problem.regularization)/blocks
    gain = np.broadcast_to(grid_gain[:, None], (rows.size, blocks))
    return gain.reshape(-1, order="F").copy()


def recover_data_bits(problem, solve_result, *, return_gain=False):
    """Demap a converged raw MMSE result; optionally return its computed gain.

    Perform no feedback or BER scoring. The optional gain is computed from the
    same estimated channel, never accepted from generator or data-fit inputs.
    """
    symbols = solve_result.require_converged()
    if symbols.shape != (problem.data_count,):
        raise ValueError("The solved symbol vector and data model disagree.")
    gains = solve_result.effective_gain
    if gains is None:
        gains = effective_symbol_gain(problem)
    bits = hard_demodulate(symbols, gains)
    return (bits, gains) if return_gain else bits
