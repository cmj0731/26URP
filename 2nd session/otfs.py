"""Unitary useful-signal transforms for rectangular-pulse, OFDM-based OTFS.

DD arrays have delay rows and Doppler columns. TF arrays have frequency rows
and time-slot columns. Serial samples contain each complete time slot in turn.
An optional per-slot CP wraps the useful waveform. Channel effects,
synchronization, equalization, and detection remain outside these transforms.
"""

import numpy as np

from cyclic_prefix import add_cyclic_prefix, remove_cyclic_prefix


def _grid(values):
    grid = np.asarray(values, dtype=np.complex128)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError("A nonempty 2-D grid is required.")
    return grid


def isfft(dd_grid):
    """Map [delay, Doppler] to [frequency, time] using unitary transforms."""
    grid = _grid(dd_grid)
    return np.fft.fft(np.fft.ifft(grid, axis=1, norm="ortho"), axis=0, norm="ortho")


def sfft(tf_grid):
    """Map [frequency, time] to [delay, Doppler] using unitary transforms."""
    grid = _grid(tf_grid)
    return np.fft.fft(np.fft.ifft(grid, axis=0, norm="ortho"), axis=1, norm="ortho")


def modulate(dd_grid, *, cp_length: int = 0):
    """Return serial samples, optionally with a CP on every time slot."""
    tf_grid = isfft(dd_grid)
    time_by_slot = np.fft.ifft(tf_grid, axis=0, norm="ortho")
    useful = time_by_slot.reshape(-1, order="F")
    return add_cyclic_prefix(useful, tf_grid.shape[0], cp_length)


def demodulate(samples, delay_bin_count: int, doppler_bin_count: int, *, cp_length: int = 0):
    """Return DD observations after basis demodulation, without equalization.

    The caller supplies aligned samples and performs required synchronization
    beforehand. A per-slot CP is removed when cp_length is nonzero; the default
    expects only useful samples, preserving the original basis-only interface.
    """
    if any(not isinstance(size, int) or size <= 0 for size in
           (delay_bin_count, doppler_bin_count)):
        raise ValueError("Grid dimensions must be positive integers.")
    signal = remove_cyclic_prefix(samples, delay_bin_count, cp_length)
    if signal.size != delay_bin_count * doppler_bin_count:
        raise ValueError("Useful sample count must match the requested DD grid.")
    time_by_slot = signal.reshape((delay_bin_count, doppler_bin_count), order="F")
    tf_grid = np.fft.fft(time_by_slot, axis=0, norm="ortho")
    return sfft(tf_grid)
