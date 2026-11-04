"""Plug-in linear-MMSE equations using estimated channels and measured noise.

Independent zero-mean data have covariance Es*I. No channel-error covariance,
truth-based cancellation, solver, output rescaling or bit decision is added.
The normal operator retains the data model's full native channel couplings.
"""

import numpy as np


def _real_scalar(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar.")
    number = float(raw)
    if not np.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} has an invalid value.")
    return number


class PlugInMMSEProblem:
    """Prepare (A_hat^H A_hat + sigma_hat^2/Es I)d_hat = A_hat^H r_data."""

    def __init__(self, data_model, received_grid, *, estimated_noise_variance,
                 data_symbol_energy):
        self.model = data_model
        self.scheme = data_model.scheme
        self.grid_shape = data_model.grid_shape
        self.data_count = data_model.data_count
        self.shape = (self.data_count, self.data_count)
        self.estimated_noise_variance = _real_scalar(estimated_noise_variance, "Noise variance")
        self.data_symbol_energy = _real_scalar(data_symbol_energy, "Data-symbol energy", positive=True)
        self.regularization = self.estimated_noise_variance/self.data_symbol_energy
        # Cancel once, with the estimated channel, before forming the RHS.
        self.observations = data_model.subtract_pilots(received_grid)
        self.observations.setflags(write=False)
        self.rhs = data_model.adjoint(self.observations)
        self.rhs.setflags(write=False)

    def _data_vector(self, values):
        vector = np.asarray(values, dtype=complex)
        if vector.shape != (self.data_count,) or not np.isfinite(vector).all():
            raise ValueError("One finite vector in the original data-cell order is required.")
        return vector

    def normal(self, data_vector):
        """Apply the normal matrix implicitly; zero variance is not floored."""
        vector = self._data_vector(data_vector)
        return self.model.adjoint(self.model.apply(vector)) + self.regularization*vector

    def objective(self, data_vector):
        """Return ||r_data-A_hat*d||^2 + sigma_hat^2/Es ||d||^2."""
        vector = self._data_vector(data_vector)
        residual = self.observations-self.model.apply(vector)
        return float(np.vdot(residual, residual).real
                     + self.regularization*np.vdot(vector, vector).real)

    def block_system(self, block):
        """Return one TF/DAFT block's matrix and RHS without solving it.

        OTFS deliberately rejects this route and retains its full-frame implicit
        operator. No giant dense normal matrix is constructed for OTFS.
        """
        matrix = self.model.block_matrix(block)
        normal = matrix.conj().T@matrix
        normal.flat[::normal.shape[0]+1] += self.regularization
        rhs = matrix.conj().T@self.observations[:, block]
        return normal, rhs


def build_mmse_problem(data_model, received_grid, estimated_noise_variance, settings):
    """Consume only the agreed public statistical policy and data-symbol energy.

    estimated_noise_variance is the silent-measurement output, not a generator
    parameter. The caller must provide the original native received grid so that
    estimated-pilot subtraction occurs exactly once.
    """
    policy = settings["receiver_equalization_policy"]["statistical_model"]
    if (policy["data_covariance"] != "Es_identity"
            or policy["nominal_noise_covariance"] != "estimated_AWGN_variance_identity"
            or policy["channel_error_policy"] != "no_additional_error_covariance_correction"):
        raise ValueError("The agreed plug-in MMSE statistical model is required.")
    energy = settings["common_symbol_modulation"]["constellation_normalization"]["average_symbol_energy"]
    return PlugInMMSEProblem(data_model, received_grid,
        estimated_noise_variance=estimated_noise_variance, data_symbol_energy=energy)
