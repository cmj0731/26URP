"""Common unit-energy Cartesian Gray 16-QAM mapping and hard decisions.

Consecutive bits are [I_MSB,I_LSB,Q_MSB,Q_LSB]. On either axis,
00/01/11/10 label -3/-1/+1/+3, with fixed 1/sqrt(10) normalization.
"""

import numpy as np


BIT_LABELS = ((np.arange(16, dtype=np.uint8)[:, None]
               >> np.array([3, 2, 1, 0], dtype=np.uint8)) & 1)
_AXIS = np.array([-3., -1., 3., 1.])
CONSTELLATION = (_AXIS[2*BIT_LABELS[:, 0]+BIT_LABELS[:, 1]]
                 + 1j*_AXIS[2*BIT_LABELS[:, 2]+BIT_LABELS[:, 3]])/np.sqrt(10)
BIT_LABELS.setflags(write=False)
CONSTELLATION.setflags(write=False)


def modulate(bits):
    """Map a 1-D complete-symbol bit vector without empirical normalization."""
    values = np.asarray(bits)
    if (values.ndim != 1 or values.size % 4 or values.dtype.kind not in "biu"
            or not np.all((values == 0) | (values == 1))):
        raise ValueError("A binary vector containing a multiple of four bits is required.")
    labels = values.astype(np.uint8, copy=False).reshape(-1, 4)
    indices = labels@np.array([8, 4, 2, 1], dtype=np.uint8)
    return CONSTELLATION[indices]


def hard_demodulate(symbols, effective_gain=1.0):
    """Nearest-constellation bits with known MMSE self-gain accounted for.

    Minimize |raw_i-gain_i*a|^2. For positive gain this equals dividing raw_i
    by gain_i before ordinary nearest-point detection. The scaled-reference
    form also handles zero gain without division or a numerical noise floor.
    Exact ties use the lowest four-bit label, including unobservable zero gain.
    """
    values = np.asarray(symbols, dtype=complex)
    gain = np.asarray(effective_gain)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("A finite 1-D vector of received symbols is required.")
    if (gain.dtype.kind not in "fiu" or gain.ndim > 1
            or (gain.ndim == 1 and gain.shape != values.shape)
            or not np.isfinite(gain).all() or np.any(gain < 0)):
        raise ValueError("Finite nonnegative real scalar/per-symbol gains are required.")
    gains = np.broadcast_to(gain, values.shape)
    distances = abs(values[:, None]-gains[:, None]*CONSTELLATION[None, :])**2
    indices = np.argmin(distances, axis=1)
    return BIT_LABELS[indices].reshape(-1).copy()
