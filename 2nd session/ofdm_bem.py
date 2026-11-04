"""Pilot-only ordinary-LS prototype for the agreed sample-level OFDM BEM.

This prototype uses all candidate integer delays, not supplied path positions.
Unknown-data ICI is left in the observation residual; it is neither cancelled
with true data nor assumed to be independent white noise. There is no data
equalizer, path-count selection, decision feedback or BER detector here.
"""

from dataclasses import dataclass

import numpy as np

from bem_basis import dpss_subspace
from cyclic_prefix import add_cyclic_prefix, remove_cyclic_prefix
from pilot_layouts import build_grid, build_masks


def modulate(grid, cp_length):
    """Unitary IFFT per frequency column, then copy each block's CP."""
    grid = np.asarray(grid, dtype=np.complex128)
    if grid.ndim != 2 or not np.isfinite(grid).all():
        raise ValueError("A finite frequency-by-time grid is required.")
    useful = np.fft.ifft(grid, axis=0, norm="ortho")
    return add_cyclic_prefix(useful.reshape(-1, order="F"), grid.shape[0], cp_length)


@dataclass
class PilotLsModel:
    block_length: int
    cp_length: int
    block_count: int
    sample_rate_hz: float
    maximum_residual_doppler_hz: float
    delays: tuple
    basis: np.ndarray
    basis_check: dict
    pilot_mask: np.ndarray
    known_pilot_samples: np.ndarray
    observation_matrix: np.ndarray
    pseudoinverse: np.ndarray
    singular_values: np.ndarray
    rank_relative_tolerance: float

    @property
    def useful_indices(self):
        return (np.arange(self.block_count)[:, None] * (self.block_length + self.cp_length)
                + self.cp_length + np.arange(self.block_length)[None, :]).reshape(-1)

    def pilot_observations(self, received_samples):
        """Extract only pilot TF coordinates, in column-major grid order."""
        received = np.asarray(received_samples, dtype=np.complex128)
        if (received.ndim != 1 or received.size != self.basis.shape[0]
                or not np.isfinite(received).all()):
            raise ValueError("One finite, aligned, CP-inclusive frame is required.")
        useful = remove_cyclic_prefix(received, self.block_length, self.cp_length)
        grid = np.fft.fft(useful.reshape((self.block_length, self.block_count), order="F"),
                           axis=0, norm="ortho")
        return grid.reshape(-1, order="F")[self.pilot_mask.reshape(-1, order="F")]

    def fit_observations(self, observations):
        """Return delay-major coefficients; columns may hold separate frames.

        This is c_hat = A_p^dagger y_p, without noise-variance knowledge.
        The interface accepts no true path parameters or transmitted data.
        """
        values = np.asarray(observations, dtype=np.complex128)
        if (values.ndim not in (1, 2) or values.shape[0] != self.observation_matrix.shape[0]
                or not np.isfinite(values).all()):
            raise ValueError("Finite pilot observations with the expected row count are required.")
        return self.pseudoinverse @ values

    def estimate(self, received_samples):
        return self.fit_observations(self.pilot_observations(received_samples))

    def reconstruct_taps(self, coefficients, *, sample_indices=None):
        """Return h[n,l] on requested receive-clock samples, including CP time."""
        values = np.asarray(coefficients, dtype=np.complex128)
        expected = len(self.delays) * self.basis.shape[1]
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError("One finite delay-major coefficient vector is required.")
        basis = self.basis if sample_indices is None else self.basis[sample_indices]
        return basis @ values.reshape((len(self.delays), self.basis.shape[1])).T


def build_pilot_ls_model(settings):
    """Build from design bounds and known pilots, without channel-generator truth.

    Required settings are prefix_policy, sampling_model, receiver_information_policy,
    the OFDM candidate_design and pilot_comparison_policy. channel_model and the
    evaluation section are not read. The DPSS band stays fixed across test levels.
    """
    prefix = settings["prefix_policy"]
    design = settings["receiver_equalization_policy"]["channel_estimation_selection"]["OFDM"]["candidate_design"]
    if design["basis_type"] != "DPSS_Slepian":
        raise ValueError("This prototype implements the agreed DPSS/Slepian basis only.")
    support = settings["receiver_information_policy"]["delay_support"]
    first, last = support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]
    block_length = prefix["useful_samples_per_block"]
    cp_length = prefix["prefix_samples_per_block"]
    length = prefix["transmitted_samples_per_frame"]
    block_count, remainder = divmod(length, block_length + cp_length)
    if remainder or first != 0 or not isinstance(last, int) or not 0 <= last <= cp_length:
        raise ValueError("The prototype requires complete blocks and integer support within CP.")
    fs = settings["sampling_model"]["sample_rate_hz"]
    maximum = design["basis_design"]["maximum_absolute_residual_doppler_hz"]
    basis, basis_check = dpss_subspace(length, maximum / fs, design["basis_function_count"])
    pilot_mask = build_masks("OFDM", scenario=settings)["pilot"]
    if pilot_mask.shape != (block_length, block_count):
        raise ValueError("Pilot grid and transmitted frame dimensions do not agree.")
    pilot_grid = build_grid("OFDM", np.zeros(int((~pilot_mask).sum()), dtype=complex), scenario=settings)
    pilot_samples = modulate(pilot_grid, cp_length)
    useful_indices = (np.arange(block_count)[:, None] * (block_length + cp_length)
                      + cp_length + np.arange(block_length)[None, :]).reshape(-1)
    delays = tuple(range(first, last + 1))
    parts = []
    for delay in delays:
        shifted = np.zeros_like(pilot_samples)
        shifted[delay:] = pilot_samples[:length - delay]
        values = (shifted[:, None] * basis)[useful_indices]
        values = values.reshape((block_count, block_length, basis.shape[1]))
        transformed = np.fft.fft(values, axis=1, norm="ortho")
        parts.append(transformed.reshape((-1, basis.shape[1]))[pilot_mask.T.reshape(-1)])
    matrix = np.concatenate(parts, axis=1)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape)
    if np.count_nonzero(singular > singular[0] * tolerance) != matrix.shape[1]:
        raise ValueError("Pilot-component observation matrix is not numerically full column rank.")
    inverse = (vh.conj().T / singular[None, :]) @ u.conj().T
    return PilotLsModel(block_length, cp_length, block_count, fs, maximum, delays,
                        basis, basis_check, pilot_mask, pilot_samples, matrix, inverse,
                        singular, tolerance)
