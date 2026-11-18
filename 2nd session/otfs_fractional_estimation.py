"""Continuous-delay/Doppler single-pilot approximate ML with joint-LS gains.

Uses the same six receive rows, 64 Doppler bins, P=3, physical delay bounds
0..5 samples and +/-46875 Hz. No trial's true delay, frequency, gain or delta
is supplied. Pilot/data isolation is not assumed under fractional delay.
"""

import numpy as np
from scipy.optimize import minimize

from otfs_estimation import PilotModel
from pilot_path_search import SequentialEstimator

DELAY_STEP = .25
JOINT_OPTIONS = {"maxiter": 150, "maxfun": 3000, "ftol": 1e-14, "gtol": 1e-9}


class ContinuousPilotModel(PilotModel):
    def __init__(self, settings):
        super().__init__(settings)
        # super validates only the original integer-delay resource layout.
        # It is NOT a claim of data isolation for fractional-delay observations.
        self.delays = tuple(np.arange(0., self.cp+DELAY_STEP/2, DELAY_STEP))

    def atoms(self, paths):
        return self.atoms_with_derivatives(paths, derivatives=False)[0]

    def atoms_with_derivatives(self, paths, derivatives=True):
        paths = tuple(paths)
        matrix = np.empty((self.rows.size, self.blocks, len(paths)), dtype=complex)
        gradient = np.empty((self.rows.size, self.blocks, 2*len(paths)), dtype=complex) if derivatives else None
        slot = np.arange(self.blocks)
        pilot_slots = self.pilot/np.sqrt(self.blocks)*np.exp(2j*np.pi*self.pilot_doppler*slot/self.blocks)
        times = self.starts[None, :]+self.rows[:, None]/self.fs
        cycles = np.fft.fftfreq(self.length)
        duration = self.sample_count/self.fs
        for column, (delay, frequency) in enumerate(paths):
            if not 0 <= delay <= self.cp or not np.isfinite([delay, frequency]).all() or abs(frequency) > self.maximum_doppler:
                raise ValueError("Continuous hypotheses must satisfy the common public support.")
            phase = np.exp(-2j*np.pi*cycles*delay)
            selected = (self.rows-self.pilot_delay) % self.length
            kernel = np.fft.ifft(phase)[selected]
            wave = pilot_slots[None, :]*np.exp(2j*np.pi*frequency*times)
            matrix[:, :, column] = np.fft.fft(kernel[:, None]*wave, axis=1, norm="ortho")
            if derivatives:
                derivative_d = np.fft.ifft(-2j*np.pi*cycles*phase)[selected]
                gradient[:, :, 2*column] = np.fft.fft(derivative_d[:, None]*wave, axis=1, norm="ortho")
                gradient[:, :, 2*column+1] = np.fft.fft(kernel[:, None]*wave*(2j*np.pi*times/duration), axis=1, norm="ortho")
        return matrix.reshape((-1, len(paths)), order="F"), (None if gradient is None else gradient.reshape((-1, 2*len(paths)), order="F"))

    def reconstruct_taps(self, estimate):
        raise ValueError("Use the continuous-path operator; six integer taps cannot represent fractional delay.")


class ContinuousEstimator(SequentialEstimator):
    def __init__(self, settings):
        model = ContinuousPilotModel(settings)
        self.duration = model.sample_count/model.fs
        super().__init__(model, 1/(4*self.duration), 1.)

    def fit_observations(self, observations):
        # Initialize from a fixed delay/frequency search, then optimize all six
        # continuous parameters with complex gains eliminated analytically.
        seed = super().fit_observations(observations)
        initial = np.array([[d, f*self.duration] for d, f in seed["paths"]]).reshape(-1)
        values = np.asarray(observations)
        scale = float(np.vdot(values, values).real)
        def unpack(parameters):
            return tuple((float(d), float(f/self.duration)) for d, f in parameters.reshape(-1, 2))
        def objective(parameters):
            matrix, derivative = self.model.atoms_with_derivatives(unpack(parameters))
            gains, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
            if rank != self.model.path_count:
                return 1e6, np.zeros(parameters.size)
            residual = values-matrix@gains
            gradient = -2*np.real(residual.conj()@(derivative*np.repeat(gains, 2)[None, :]))/scale
            return float(np.vdot(residual, residual).real/scale), gradient
        bound_f = self.model.maximum_doppler*self.duration
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(0., float(self.model.cp)), (-bound_f, bound_f)]*self.model.path_count,
                          options=JOINT_OPTIONS)
        best = result.x if result.fun <= objective(initial)[0] else initial
        fit = self.model.fit_gains(values, unpack(best))
        fit["optimizer"] = {"success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(result.nit), "evaluations": int(result.nfev)}
        fit["search_history"] = seed["search_history"]
        return fit
