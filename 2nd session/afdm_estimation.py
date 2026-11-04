"""Joint-frame embedded-pilot approximate ML with conditional LS path gains.

Use only public design bounds and received pilot-region observations. Sequential
path addition approximates the joint nonlinear search; it is not a global-ML
solver. Unknown-data leakage stays in the residual. No data equalizer or BER
detector is provided here.
"""

import numpy as np

import afdm
from pilot_layouts import known_pilot_symbols


class PilotModel:
    def __init__(self, settings):
        spec = settings["waveform_design"]["AFDM"]
        prefix = settings["prefix_policy"]
        information = settings["receiver_information_policy"]
        self.length, self.blocks = spec["daft_length"], spec["blocks_per_frame"]
        self.c1, self.c2 = spec["chirp_parameters"]["c1"], spec["chirp_parameters"]["c2"]
        self.cp, self.fs = prefix["prefix_samples_per_block"], settings["sampling_model"]["sample_rate_hz"]
        self.delays = tuple(range(information["delay_support"]["minimum_excess_delay_samples"],
                                  information["delay_support"]["maximum_excess_delay_samples"]+1))
        self.maximum_doppler = information["residual_doppler_support"]["maximum_absolute_residual_doppler_hz"]
        self.path_count = information["known_path_count_including_los"]
        self.pilots = known_pilot_symbols("AFDM", scenario=settings)
        design = settings["receiver_equalization_policy"]["channel_estimation_selection"]["AFDM"]["design"]
        if design["processing_window"]["blocks_per_window"] != self.blocks:
            raise ValueError("This implementation requires the selected joint-frame window.")
        radius = spec["chirp_parameters"]["selection_rule"]["c1_numeric_formula"]["alpha_max"] + spec["chirp_parameters"]["selection_rule"]["c1_numeric_formula"]["xi_nu"]
        slope = int(round(2*self.length*self.c1))
        if slope != 2*radius+1 or max(self.delays) > self.cp:
            raise ValueError("Review the selected chirp/guard/delay design before estimation.")
        self.rows = np.unique(np.arange(-slope*max(self.delays)-radius,radius+1) % self.length)
        self.starts = (np.arange(self.blocks)*(self.length+self.cp)+self.cp)/self.fs
        self.sample_count = (self.length+self.cp)*self.blocks

    def observations(self, received_samples):
        grid = afdm.demodulate(received_samples,self.length,self.blocks,self.c1,self.c2,cp_length=self.cp)
        return grid[self.rows].reshape(-1,order="F")

    def atoms(self, paths):
        paths = tuple(paths)
        if not paths:
            raise ValueError("At least one path hypothesis is required.")
        templates, frequencies = [], []
        for delay,frequency in paths:
            if delay not in self.delays or not np.isfinite(frequency) or abs(frequency) > self.maximum_doppler:
                raise ValueError("Path hypotheses must satisfy the public delay/Doppler bounds.")
            templates.append(afdm.pilot_path_response(self.length,self.c1,self.c2,delay,frequency,self.fs)[self.rows])
            frequencies.append(frequency)
        templates = np.column_stack(templates)
        weights = self.pilots[:,None]*np.exp(2j*np.pi*self.starts[:,None]*np.array(frequencies)[None,:])
        return (templates[:,None,:]*weights[None,:,:]).reshape((-1,len(paths)),order="F")

    def fit_gains(self, observations, paths):
        paths = tuple(paths)
        values = np.asarray(observations,dtype=np.complex128)
        if values.shape != (self.rows.size*self.blocks,) or not np.isfinite(values).all():
            raise ValueError("One finite joint-frame pilot observation vector is required.")
        matrix = self.atoms(paths)
        gains,_,rank,singular = np.linalg.lstsq(matrix,values,rcond=None)
        if rank != matrix.shape[1]:
            raise ValueError("Path hypotheses are not numerically identifiable.")
        residual = values-matrix@gains
        return {"paths":tuple(paths),"gains":gains,"residual":residual,
                "residual_power":float(np.vdot(residual,residual).real),
                "rank":int(rank),"condition":float(singular[0]/singular[-1])}

    def reconstruct_taps(self, estimate):
        taps = np.zeros((self.sample_count,max(self.delays)+1),dtype=complex)
        time = np.arange(self.sample_count)/self.fs
        for (delay,frequency),gain in zip(estimate["paths"],estimate["gains"]):
            taps[:,delay] += gain*np.exp(2j*np.pi*frequency*time)
        return taps


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


def build_estimator(settings):
    selected = settings["receiver_equalization_policy"]["channel_estimation_selection"]["AFDM"]
    if selected["algorithm"] != "embedded_pilot_fractional_Doppler_approximate_ML_with_joint_LS_gains":
        raise ValueError("The selected AFDM estimator does not match this implementation.")
    design = selected["design"]
    parameters = design["search_parameters"]
    model = PilotModel(settings)
    duration = model.sample_count/model.fs
    spacing = 1/(duration*parameters["initial_grid_oversampling_factor"])
    if parameters["path_additions"] != model.path_count or not np.isclose(spacing,parameters["initial_grid_spacing_hz"]):
        raise ValueError("Review the selected path count or search grid after a design change.")
    return SequentialEstimator(model,spacing,
                               parameters["refinement_tolerance_hz"])
