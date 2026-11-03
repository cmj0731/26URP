"""Per-block cyclic prefixes on serial complex-baseband useful samples.

Each block is contiguous in the serial stream. Adding a CP copies samples
without changing their amplitude; the copied energy is a separate overhead.
"""

import numpy as np


def _samples_and_lengths(samples, block_length, cp_length):
    if (not isinstance(block_length, (int, np.integer))
            or isinstance(block_length, (bool, np.bool_)) or block_length <= 0):
        raise ValueError("Useful block length must be a positive integer.")
    if (not isinstance(cp_length, (int, np.integer))
            or isinstance(cp_length, (bool, np.bool_)) or not 0 <= cp_length <= block_length):
        raise ValueError("CP length must be an integer between zero and the useful block length.")
    signal = np.asarray(samples, dtype=np.complex128)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("A nonempty 1-D serial sample stream is required.")
    return signal


def add_cyclic_prefix(samples, block_length: int, cp_length: int):
    """Copy each useful block's last cp_length samples to its beginning."""
    signal = _samples_and_lengths(samples, block_length, cp_length)
    if signal.size % block_length:
        raise ValueError("Useful sample count must contain complete blocks.")
    if cp_length == 0:
        return signal.copy()
    blocks = signal.reshape(-1, block_length)
    return np.concatenate((blocks[:, -cp_length:], blocks), axis=1).reshape(-1)


def remove_cyclic_prefix(samples, block_length: int, cp_length: int):
    """Remove per-block CP samples from an already aligned serial stream."""
    signal = _samples_and_lengths(samples, block_length, cp_length)
    transmitted_block_length = block_length + cp_length
    if signal.size % transmitted_block_length:
        raise ValueError("Prefixed sample count must contain complete transmitted blocks.")
    blocks = signal.reshape(-1, transmitted_block_length)
    return blocks[:, cp_length:].reshape(-1).copy()
