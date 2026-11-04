"""Unitary block AFDM transforms and the embedded-pilot response.

These helpers do not estimate delays/Dopplers or perform data equalization.
The ordinary CP and analytic pilot response require the agreed even N and
integer 2*N*c1 condition; arbitrary chirp-periodic prefixes are not implemented.
"""

import numpy as np

from cyclic_prefix import add_cyclic_prefix, remove_cyclic_prefix


def _chirps(length, c1, c2):
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("A positive integer DAFT length is required.")
    if not np.isfinite([c1, c2]).all():
        raise ValueError("Finite chirp parameters are required.")
    index = np.arange(length, dtype=float)
    return np.exp(2j * np.pi * c1 * index**2), np.exp(2j * np.pi * c2 * index**2)


def _ordinary_cp_condition(length, c1):
    value = 2 * length * c1
    if length % 2 or not np.isclose(value, round(value), rtol=0, atol=1e-12):
        raise ValueError("The selected ordinary-CP model requires even N and integer 2*N*c1.")


def modulate(daft_grid, c1, c2, *, cp_length=0):
    """IDAFT on each column, then optional CP, serialized block by block."""
    grid = np.asarray(daft_grid, dtype=np.complex128)
    if grid.ndim != 2 or 0 in grid.shape or not np.isfinite(grid).all():
        raise ValueError("A finite, nonempty DAFT-index-by-block grid is required.")
    chirp1, chirp2 = _chirps(grid.shape[0], c1, c2)
    if cp_length:
        _ordinary_cp_condition(grid.shape[0], c1)
    useful = chirp1[:, None] * np.fft.ifft(chirp2[:, None] * grid, axis=0, norm="ortho")
    return add_cyclic_prefix(useful.reshape(-1, order="F"), grid.shape[0], cp_length)


def demodulate(samples, daft_length, block_count, c1, c2, *, cp_length=0):
    """Remove aligned CPs and apply the DAFT, without channel equalization."""
    chirp1, chirp2 = _chirps(daft_length, c1, c2)
    if not isinstance(block_count, int) or isinstance(block_count, bool) or block_count <= 0:
        raise ValueError("A positive integer block count is required.")
    if cp_length:
        _ordinary_cp_condition(daft_length, c1)
    useful = remove_cyclic_prefix(samples, daft_length, cp_length)
    if useful.size != daft_length * block_count or not np.isfinite(useful).all():
        raise ValueError("Finite useful samples must match the requested grid dimensions.")
    grid = useful.reshape((daft_length, block_count), order="F")
    return chirp2.conj()[:, None] * np.fft.fft(chirp1.conj()[:, None] * grid, axis=0, norm="ortho")


def pilot_path_response(daft_length, c1, c2, delay, doppler_hz, sample_rate_hz):
    """Full DAFT template for a candidate integer delay and fractional Doppler.

    Pilot index is zero and unit amplitude. Unit path gain is referenced to
    the first useful receive sample of the block. A frame-reference gain must
    also include exp(+j*2*pi*f*t_block_start), with CP in the clock.
    Our channel sign is exp(+j*2*pi*f*t), so the peak is near
    k = (f*N/Fs - 2*N*c1*delay) modulo N; no integer rounding is applied.
    """
    _chirps(daft_length, c1, c2)
    _ordinary_cp_condition(daft_length, c1)
    if (not isinstance(delay, (int, np.integer)) or isinstance(delay, (bool, np.bool_))
            or not 0 <= delay < daft_length):
        raise ValueError("An integer candidate delay within the block is required.")
    if not np.isfinite([doppler_hz, sample_rate_hz]).all() or sample_rate_hz <= 0:
        raise ValueError("Finite Doppler and a positive sample rate are required.")
    index = np.arange(daft_length)
    offset = doppler_hz * daft_length / sample_rate_hz - 2 * daft_length * c1 * delay - index
    # The normalized finite geometric sum is periodic in offset with period N.
    offset = (offset + daft_length / 2) % daft_length - daft_length / 2
    geometric_sum = (np.exp(1j * np.pi * (daft_length - 1) * offset / daft_length)
                     * np.sinc(offset) / np.sinc(offset / daft_length))
    phase = np.exp(2j * np.pi * (c1 * delay**2 - c2 * index**2))
    return phase * geometric_sum
