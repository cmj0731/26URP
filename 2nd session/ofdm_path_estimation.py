"""Fit CP-covered integer-delay/continuous-Doppler paths on OFDM pilot cells."""

import numpy as np

from pilot_path_search import SequentialEstimator


class PilotModel:
    """Exact response of the known pilot grid, including intercarrier coupling.

    The BEM receiver and this model read the same OFDM pilot observations. ICI
    from unknown data is present in those samples and remains in the residual.
    """

    def __init__(self, settings, pilot_ls_model):
        info = settings["receiver_information_policy"]
        self.delays = pilot_ls_model.delays
        self.maximum_doppler = info["residual_doppler_support"]["maximum_absolute_residual_doppler_hz"]
        self.path_count = info["known_path_count_including_los"]
        self.fs = pilot_ls_model.sample_rate_hz
        self.length = pilot_ls_model.block_length
        self.sample_count = pilot_ls_model.known_pilot_samples.size
        self.observations = pilot_ls_model.pilot_observations
        mask = pilot_ls_model.pilot_mask.T
        active = mask.any(axis=1)
        self.mask = mask[active]
        self.observation_count = np.count_nonzero(self.mask)
        indices = pilot_ls_model.useful_indices.reshape(mask.shape)[active]
        self.starts = indices[:, 0]/self.fs
        self.offsets = np.arange(self.length)/self.fs
        pilot_time = pilot_ls_model.known_pilot_samples[indices]
        self.shifted_pilots = {delay: np.roll(pilot_time, delay, axis=1) for delay in self.delays}

    def atoms(self, paths):
        paths = tuple(paths)
        matrix = np.empty((self.observation_count, len(paths)), dtype=complex, order="F")
        for column, (delay, frequency) in enumerate(paths):
            if delay not in self.delays or not np.isfinite(frequency) or abs(frequency) > self.maximum_doppler:
                raise ValueError("Path hypotheses must satisfy the shared delay/Doppler bounds.")
            within_slot = np.exp(2j*np.pi*frequency*self.offsets)
            slot_phase = np.exp(2j*np.pi*frequency*self.starts)
            response = np.fft.fft(self.shifted_pilots[delay]*within_slot[None, :], axis=1, norm="ortho")
            matrix[:, column] = (response*slot_phase[:, None])[self.mask]
        return matrix

    def fit_gains(self, observations, paths):
        paths = tuple(paths)
        matrix = self.atoms(paths)
        gains, _, rank, singular = np.linalg.lstsq(matrix, observations, rcond=None)
        if rank != len(paths):
            raise ValueError("Path hypotheses are not numerically identifiable.")
        residual = observations-matrix@gains
        return {"paths": paths, "gains": gains, "residual": residual,
                "residual_power": float(np.vdot(residual, residual).real),
                "condition": float(singular[0]/singular[-1])}

    def reconstruct_taps(self, estimate):
        taps = np.zeros((self.sample_count, len(self.delays)), dtype=complex)
        time = np.arange(self.sample_count)/self.fs
        for (delay, frequency), gain in zip(estimate["paths"], estimate["gains"]):
            taps[:, self.delays.index(delay)] += gain*np.exp(2j*np.pi*frequency*time)
        return taps


def build_estimator(settings, pilot_ls_model):
    model = PilotModel(settings, pilot_ls_model)
    search = settings["receiver_equalization_policy"]["channel_estimation_selection"]["OTFS"]["design"]["search_parameters"]
    duration = model.sample_count/model.fs
    spacing = 1/(duration*search["initial_grid_oversampling_factor"])
    return SequentialEstimator(model, spacing, search["refinement_tolerance_hz"])
