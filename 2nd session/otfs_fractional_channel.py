"""CP-covered continuous delays of the existing per-slot Fourier waveform.

Each CP+useful slot is its own finite rectangular-window Fourier series,
with signed FFT frequencies (-N/2 included, +N/2 excluded). No interpolation
FIR, extra filter tail, whole-frame periodic extension, or RF filter is added.
The serial evaluator explicitly selects the source slot at t-delay; it does
not assume a circular channel. A separate useful-grid operator is valid only
when every physical delay lies in [0, CP].
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).with_name(".runtime-deps")))
from scipy.linalg import cho_factor, cho_solve


def delay_kernel(length, delay):
    return np.fft.ifft(np.exp(-2j*np.pi*np.fft.fftfreq(length)*delay))


def serial_channel(samples, delays, gains, frequencies, fs, length, cp):
    """Evaluate the piecewise Fourier waveform at delayed serial sample times.

    Before/after this finite transmitted frame the waveform is zero. CP output
    samples may contain a preceding slot; useful outputs must never require
    one when delay <= CP. This independent generator uses no receiver model.
    """
    signal = np.asarray(samples)
    span = length+cp
    if signal.ndim != 1 or signal.size % span:
        raise ValueError("Complete serial CP+useful slots are required.")
    blocks = signal.size//span
    slots = signal.reshape(blocks, span)
    assert np.allclose(slots[:, :cp], slots[:, -cp:], rtol=0, atol=1e-12)
    coefficients = np.fft.fft(slots[:, cp:], axis=1, norm="ortho")
    bins = np.fft.fftfreq(length)
    sample_time = np.arange(signal.size)
    result = np.zeros(signal.size, dtype=complex)
    for delay, gain, frequency in zip(delays, gains, frequencies):
        if not 0 <= delay <= cp:
            raise ValueError("This experiment requires CP-covered physical delays.")
        source = sample_time-float(delay)
        source_blocks = np.floor(source/span).astype(int)
        within = source-source_blocks*span-cp
        valid = (source >= 0) & (source_blocks < blocks)
        # Every source slot has the same fractional sample offset. Evaluate
        # its Fourier series once on that shifted lattice, then gather by
        # explicit source-slot and within-slot indices (including CP samples).
        integer = np.floor(within).astype(int)
        fraction = float(within[valid][0]-integer[valid][0])
        assert np.max(abs(within[valid]-integer[valid]-fraction)) < 1e-10
        interpolated = np.fft.ifft(coefficients*np.exp(2j*np.pi*bins*fraction), axis=1, norm="ortho")
        delayed = np.zeros_like(result)
        delayed[valid] = interpolated[source_blocks[valid], integer[valid] % length]
        result += gain*np.exp(2j*np.pi*frequency*sample_time/fs)*delayed
    return result


class FractionalChannelOperator:
    """Full DD channel from estimated continuous path parameters only."""
    scheme = "OTFS"

    def __init__(self, length, blocks, cp, fs, paths, gains):
        self.length, self.blocks, self.cp, self.fs = length, blocks, cp, fs
        self.grid_shape, self.shape = (length, blocks), (length*blocks,)*2
        self.paths, self.gains = tuple(paths), np.asarray(gains, dtype=complex).copy()
        if len(self.paths) != self.gains.size or not self.paths:
            raise ValueError("A nonempty estimated path list and matching gains are required.")
        if any(not 0 <= d <= cp or not np.isfinite([d, f]).all() for d, f in self.paths):
            raise ValueError("Estimated paths must have finite CP-covered delays.")
        self.time = ((np.arange(blocks)[None, :]*(length+cp)+cp+np.arange(length)[:, None])/fs)
        self.phases = [np.exp(-2j*np.pi*np.fft.fftfreq(length)*d)[:, None] for d, f in self.paths]
        self.weights = [g*np.exp(2j*np.pi*f*self.time) for (d, f), g in zip(self.paths, self.gains)]
        self.kernels = [delay_kernel(length, d) for d, f in self.paths]

    def apply(self, grid):
        useful = np.fft.ifft(grid, axis=1, norm="ortho")
        spectrum = np.fft.fft(useful, axis=0, norm="ortho")
        result = sum(w*np.fft.ifft(spectrum*p, axis=0, norm="ortho") for w, p in zip(self.weights, self.phases))
        return np.fft.fft(result, axis=1, norm="ortho")

    def adjoint(self, grid):
        useful = np.fft.ifft(grid, axis=1, norm="ortho")
        result = sum(np.fft.ifft(np.fft.fft(w.conj()*useful, axis=0, norm="ortho")*p.conj(), axis=0, norm="ortho")
                     for w, p in zip(self.weights, self.phases))
        return np.fft.fft(result, axis=1, norm="ortho")

    def time_matrix(self, block):
        indices = (np.arange(self.length)[:, None]-np.arange(self.length)[None, :]) % self.length
        return sum(w[:, block, None]*kernel[indices] for w, kernel in zip(self.weights, self.kernels))


def exact_effective_gain(problem):
    """Exact diag(G A) using the same OTFS row-mask/unitary identity as before.

    Dense local time matrices retain all fractional-delay spreading. The data
    solve remains the unchanged full-frame CG; these factorizations compute
    only its nominal self-gain, using estimated channels and measured variance.
    """
    model, channel = problem.model, problem.model.channel
    assert np.array_equal(model.data_mask, np.broadcast_to(model.data_mask[:, :1], model.grid_shape))
    rows = np.flatnonzero(model.data_mask[:, 0])
    lam = problem.regularization
    result = np.zeros(rows.size)
    identity = np.eye(rows.size)
    for block in range(channel.blocks):
        a = channel.time_matrix(block)[:, rows]
        gram = a.conj().T@a
        factor = cho_factor(gram+lam*identity, lower=True, check_finite=False)
        gain = 1-lam*np.diag(cho_solve(factor, identity, check_finite=False)).real
        if not np.isfinite(gain).all() or np.any(gain < 0):
            raise RuntimeError("Invalid nominal MMSE self-gain.")
        result += gain/channel.blocks
    return np.broadcast_to(result[:, None], (rows.size, channel.blocks)).reshape(-1, order="F").copy()


def operator_inner(left, right):
    """Evaluator-only exact Frobenius inner product of full useful operators."""
    assert left.grid_shape == right.grid_shape and np.array_equal(left.time, right.time)
    total = 0j
    for i, ((dl, fl), gl) in enumerate(zip(left.paths, left.gains)):
        for j, ((dr, fr), gr) in enumerate(zip(right.paths, right.gains)):
            total += gl.conjugate()*gr*np.vdot(left.kernels[i], right.kernels[j])*np.exp(2j*np.pi*(fr-fl)*left.time).sum()
    return total


def operator_nmse(estimated, truth):
    denominator = operator_inner(truth, truth).real
    numerator = (operator_inner(estimated, estimated)+operator_inner(truth, truth)-2*operator_inner(truth, estimated).real).real
    return float(max(0., numerator)/denominator)
