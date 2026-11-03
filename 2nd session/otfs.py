"""Unitary useful-signal transforms for rectangular-pulse, OFDM-based OTFS.

DD arrays have delay rows and Doppler columns. TF arrays have frequency rows
and time-slot columns. Serial samples contain each complete time slot in turn.
Prefixes, channel effects, synchronization, equalization, and detection are
outside these basis transforms.
"""

import numpy as np


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


def modulate(dd_grid):
    """Return useful serial samples, before any prefix or guard insertion."""
    tf_grid = isfft(dd_grid)
    time_by_slot = np.fft.ifft(tf_grid, axis=0, norm="ortho")
    return time_by_slot.reshape(-1, order="F")


def demodulate(samples, delay_bin_count: int, doppler_bin_count: int):
    """Return DD observations after basis demodulation, without equalization.

    The caller supplies aligned useful samples, with any prefix removed and
    any required synchronization performed beforehand.
    """
    if any(not isinstance(size, int) or size <= 0 for size in
           (delay_bin_count, doppler_bin_count)):
        raise ValueError("Grid dimensions must be positive integers.")
    signal = np.asarray(samples, dtype=np.complex128)
    if signal.ndim != 1 or signal.size != delay_bin_count * doppler_bin_count:
        raise ValueError("Useful sample count must match the requested DD grid.")
    time_by_slot = signal.reshape((delay_bin_count, doppler_bin_count), order="F")
    tf_grid = np.fft.fft(time_by_slot, axis=0, norm="ortho")
    return sfft(tf_grid)
