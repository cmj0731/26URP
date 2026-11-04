"""Shared fixed-grid sequential pilot-template search with joint-LS gains.

The native pilot model supplies observations, atoms and conditional gain fits.
This is a three-step approximation to joint nonlinear ML, not global ML.
"""

import numpy as np


class SequentialEstimator:
    def __init__(self, model, initial_spacing_hz, refinement_tolerance_hz):
        if not np.isfinite([initial_spacing_hz,refinement_tolerance_hz]).all() or not 0 < refinement_tolerance_hz <= initial_spacing_hz:
            raise ValueError("Positive finite spacing and a smaller refinement tolerance are required.")
        if refinement_tolerance_hz < 8*np.spacing(max(1.0,model.maximum_doppler)):
            raise ValueError("Refinement tolerance must exceed floating-point frequency resolution.")
        self.model, self.tolerance = model, refinement_tolerance_hz
        count = int(np.floor(model.maximum_doppler/initial_spacing_hz))
        self.grid = np.unique(np.r_[-model.maximum_doppler,
                                   np.arange(-count,count+1)*initial_spacing_hz,model.maximum_doppler])
        self.pairs = tuple((delay,float(frequency)) for delay in model.delays for frequency in self.grid)
        self.dictionary = model.atoms(self.pairs)
        self.adjoint = self.dictionary.conj().T
        self.norms = np.sum(np.abs(self.dictionary)**2,axis=0)

    @staticmethod
    def _score(atom, residual, previous_q):
        energy = float(np.vdot(atom,atom).real)
        perpendicular = energy
        if previous_q is not None:
            perpendicular -= float(np.linalg.norm(previous_q.conj().T@atom)**2)
        threshold = np.finfo(float).eps*atom.size*energy
        if perpendicular <= threshold:
            return -np.inf
        return float(abs(np.vdot(atom,residual))**2/perpendicular)

    def _refine(self, delay, grid_index, residual, previous_q):
        low = float(self.grid[max(0,grid_index-1)])
        high = float(self.grid[min(self.grid.size-1,grid_index+1)])
        seed = float(self.grid[grid_index])
        score = lambda frequency: self._score(self.model.atoms([(delay,frequency)])[:,0],residual,previous_q)
        evaluated = [(score(seed),seed),(score(low),low),(score(high),high)]
        ratio = (np.sqrt(5)-1)/2
        x1, x2 = high-ratio*(high-low), low+ratio*(high-low)
        f1,f2 = score(x1),score(x2)
        evaluated.extend([(f1,x1),(f2,x2)])
        while high-low > self.tolerance:
            if f1 < f2:
                low,x1,f1 = x1,x2,f2
                x2 = low+ratio*(high-low)
                f2 = score(x2)
                evaluated.append((f2,x2))
            else:
                high,x2,f2 = x2,x1,f1
                x1 = high-ratio*(high-low)
                f1 = score(x1)
                evaluated.append((f1,x1))
        best = max(evaluated,key=lambda item:item[0])
        return float(best[1]),float(high-low)

    def fit_observations(self, observations):
        values = np.asarray(observations,dtype=np.complex128)
        if values.shape != (self.dictionary.shape[0],) or not np.isfinite(values).all():
            raise ValueError("One finite joint-frame pilot observation vector is required.")
        paths,history = [],[]
        residual,previous_q = values,None
        fit = None
        for _ in range(self.model.path_count):
            norms = self.norms.copy()
            if previous_q is not None:
                norms -= np.sum(np.abs(previous_q.conj().T@self.dictionary)**2,axis=0)
            threshold = np.finfo(float).eps*self.dictionary.shape[0]*self.norms
            scores = np.full(norms.shape,-np.inf)
            valid = norms > threshold
            numerator = np.abs(self.adjoint@residual)**2
            scores[valid] = numerator[valid]/norms[valid]
            selected = int(np.argmax(scores))
            if not np.isfinite(scores[selected]):
                raise ValueError("No numerically independent path hypothesis remains.")
            delay,coarse_frequency = self.pairs[selected]
            _,frequency_index = divmod(selected,self.grid.size)
            frequency,width = self._refine(delay,frequency_index,residual,previous_q)
            paths.append((delay,frequency))
            fit = self.model.fit_gains(values,paths)
            residual = fit["residual"]
            previous_q = np.linalg.qr(self.model.atoms(paths),mode="reduced")[0]
            history.append({"coarse_frequency_hz":coarse_frequency,"refinement_final_interval_hz":width,
                            "joint_LS_residual_power":fit["residual_power"]})
        fit["search_history"] = history
        return fit

    def estimate(self, received_samples):
        return self.fit_observations(self.model.observations(received_samples))
