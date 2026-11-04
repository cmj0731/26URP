"""DD-a-OFDM pilot-CTF/DD estimation with fixed three-sweep AP refinement.

The desired self-subcarrier model follows equations (19)-(25) of the selected
paper. Off-diagonal ICI from known pilots and unknown data remains in the
residual. This is not an exact all-pilot-response fit or a data MMSE equalizer.
The receiver reads public design settings, fixed pilots and received samples.
"""

import numpy as np

from cyclic_prefix import remove_cyclic_prefix
from pilot_layouts import build_masks, known_pilot_symbols
from pilot_path_search import SequentialEstimator, refine_frequency


class PilotModel:
    def __init__(self, settings):
        layout = settings["pilot_comparison_policy"]["shared_ofdm_dd_a_tf_layout"]
        prefix = settings["prefix_policy"]
        information = settings["receiver_information_policy"]
        self.length,self.blocks = layout["grid_shape_frequency_by_time"]
        self.cp = prefix["prefix_samples_per_block"]
        self.fs = settings["sampling_model"]["sample_rate_hz"]
        self.delays = tuple(range(information["delay_support"]["minimum_excess_delay_samples"],
                                  information["delay_support"]["maximum_excess_delay_samples"]+1))
        self.maximum_doppler = information["residual_doppler_support"]["maximum_absolute_residual_doppler_hz"]
        self.path_count = information["known_path_count_including_los"]
        self.frequency_rows = np.arange(layout["frequency_offset_bins"],self.length,layout["frequency_spacing_in_bins"])
        self.time_columns = np.arange(layout["time_offset_blocks"],self.blocks,layout["time_spacing_in_blocks"])
        self.shape = (self.frequency_rows.size,self.time_columns.size)
        self.observation_count = int(np.prod(self.shape))
        self.pilots = known_pilot_symbols("DD-a-OFDM",scenario=settings).reshape(self.shape,order="F")
        mask = build_masks("DD-a-OFDM",scenario=settings)["pilot"]
        expected = np.zeros((self.length,self.blocks),dtype=bool)
        expected[np.ix_(self.frequency_rows,self.time_columns)] = True
        if (not np.array_equal(mask,expected) or prefix["useful_samples_per_block"] != self.length
                or min(self.delays) < 0 or max(self.delays) > self.cp):
            raise ValueError("Review the selected lattice, frame or CP-covered delay support.")
        design = settings["waveform_design"]["DD-a-OFDM"]["receiver_design"]["pilot_estimation_design"]
        if design["processing_window"]["time_slots_per_frame"] != self.blocks:
            raise ValueError("This implementation requires the selected whole-frame pilot window.")
        self.starts = ((self.length+self.cp)*self.time_columns+self.cp)/self.fs
        self.sample_count = (self.length+self.cp)*self.blocks
        if self.sample_count != prefix["transmitted_samples_per_frame"]:
            raise ValueError("The transmitted frame and pilot lattice disagree.")
        self.delay_vectors = np.fft.ifft(np.exp(-2j*np.pi*self.frequency_rows[:,None]
                                     * np.array(self.delays)[None,:]/self.length),axis=0,norm="ortho")

    @staticmethod
    def to_dd(pilot_ctf):
        return np.fft.fft(np.fft.ifft(pilot_ctf,axis=0,norm="ortho"),axis=1,norm="ortho")

    def observations(self, received_samples):
        values = np.asarray(received_samples,dtype=np.complex128)
        if values.shape != (self.sample_count,) or not np.isfinite(values).all():
            raise ValueError("One finite aligned CP-inclusive frame is required.")
        useful = remove_cyclic_prefix(values,self.length,self.cp)
        tf = np.fft.fft(useful.reshape((self.length,self.blocks),order="F"),axis=0,norm="ortho")
        pilot_ctf = tf[np.ix_(self.frequency_rows,self.time_columns)]/self.pilots
        return self.to_dd(pilot_ctf).reshape(-1,order="F")

    def atoms(self, paths):
        paths = tuple(paths)
        if not paths:
            raise ValueError("At least one path hypothesis is required.")
        if any(delay not in self.delays or not np.isfinite(frequency)
               or abs(frequency) > self.maximum_doppler for delay,frequency in paths):
            raise ValueError("Path hypotheses must satisfy the public delay/Doppler bounds.")
        frequencies = np.array([frequency for _,frequency in paths])
        # Exact same-subcarrier coefficient of the within-block Doppler matrix.
        # Include this and the first CP offset so fitted gains use receive t=0.
        diagonal = np.mean(np.exp(2j*np.pi*np.arange(self.length)[:,None]
                                  * frequencies[None,:]/self.fs),axis=0)
        time_vectors = np.fft.fft(diagonal[None,:]*np.exp(2j*np.pi*self.starts[:,None]
                                   * frequencies[None,:]),axis=0,norm="ortho")
        delay_vectors = self.delay_vectors[:,[self.delays.index(delay) for delay,_ in paths]]
        return (delay_vectors[:,None,:]*time_vectors[None,:,:]).reshape((-1,len(paths)),order="F")

    def fit_gains(self, observations, paths):
        paths = tuple(paths)
        values = np.asarray(observations,dtype=np.complex128)
        if values.shape != (self.observation_count,) or not np.isfinite(values).all():
            raise ValueError("One finite DD pilot observation vector is required.")
        matrix = self.atoms(paths)
        gains,_,rank,singular = np.linalg.lstsq(matrix,values,rcond=None)
        if rank != len(paths):
            raise ValueError("Path hypotheses are not numerically identifiable.")
        residual = values-matrix@gains
        return {"paths":paths,"gains":gains,"residual":residual,
                "residual_power":float(np.vdot(residual,residual).real),
                "rank":int(rank),"condition":float(singular[0]/singular[-1])}

    def reconstruct_taps(self, estimate):
        taps = np.zeros((self.sample_count,max(self.delays)+1),dtype=complex)
        time = np.arange(self.sample_count)/self.fs
        for (delay,frequency),gain in zip(estimate["paths"],estimate["gains"]):
            taps[:,delay] += gain*np.exp(2j*np.pi*frequency*time)
        return taps


class AlternatingProjectionEstimator(SequentialEstimator):
    def __init__(self, model, spacing, tolerance, outer_sweeps):
        super().__init__(model,spacing,tolerance)
        self.outer_sweeps = outer_sweeps

    @staticmethod
    def _path_score(atom, residual, other_q):
        energy = float(np.vdot(atom,atom).real)
        perpendicular = energy-float(np.linalg.norm(other_q.conj().T@atom)**2)
        if perpendicular <= np.finfo(float).eps*atom.size*energy:
            return -np.inf
        # Other path amplitudes are fixed during this coordinate update.
        return float(abs(np.vdot(atom,residual))**2/energy)

    def _sweep(self, values, estimate):
        paths = list(estimate["paths"])
        gains = estimate["gains"].copy()
        matrix = self.model.atoms(paths)
        updates = []
        for target in range(self.model.path_count):
            residual = values-matrix@gains+matrix[:,target]*gains[target]
            other = [i for i in range(self.model.path_count) if i != target]
            other_q = np.linalg.qr(matrix[:,other],mode="reduced")[0]
            perpendicular = self.norms-np.sum(np.abs(other_q.conj().T@self.dictionary)**2,axis=0)
            valid = perpendicular > np.finfo(float).eps*self.dictionary.shape[0]*self.norms
            scores = np.full(self.norms.shape,-np.inf)
            numerator = np.abs(self.adjoint@residual)**2
            scores[valid] = numerator[valid]/self.norms[valid]
            selected = int(np.argmax(scores))
            if not np.isfinite(scores[selected]):
                raise ValueError("No numerically independent path candidate remains.")
            delay,coarse_frequency = self.pairs[selected]
            _,frequency_index = divmod(selected,self.grid.size)
            score = lambda frequency: self._path_score(self.model.atoms([(delay,frequency)])[:,0],residual,other_q)
            frequency,width = refine_frequency(self.grid,frequency_index,self.tolerance,score)
            new_path = (delay,frequency)
            atom = self.model.atoms([new_path])[:,0]
            retained = self._path_score(matrix[:,target],residual,other_q) >= self._path_score(atom,residual,other_q)
            if not retained:
                paths[target] = new_path
                matrix[:,target] = atom
            # Conditional scalar amplitude is implicit in the path-wise
            # compressed likelihood; the joint LS still occurs once per sweep.
            gains[target] = np.vdot(matrix[:,target],residual)/np.vdot(matrix[:,target],matrix[:,target]).real
            updates.append({"target_index":target,"coarse_frequency_hz":coarse_frequency,
                            "refinement_final_interval_hz":width,"previous_hypothesis_retained":bool(retained)})
        fitted = self.model.fit_gains(values,paths)
        return fitted,{"path_updates":updates,"joint_LS_residual_power":fitted["residual_power"]}

    def fit_observations(self, observations):
        values = np.asarray(observations,dtype=np.complex128)
        initial = super().fit_observations(values)
        estimate = initial
        history = []
        for _ in range(self.outer_sweeps):
            estimate,record = self._sweep(values,estimate)
            history.append(record)
        estimate["initial_estimate"] = initial
        estimate["search_history"] = initial["search_history"]
        estimate["outer_history"] = history
        return estimate


def build_estimator(settings):
    receiver = settings["waveform_design"]["DD-a-OFDM"]["receiver_design"]
    design,control = receiver["pilot_estimation_design"],receiver["iteration_control"]
    parameters = design["search_parameters"]
    model = PilotModel(settings)
    spacing = 1/(model.sample_count/model.fs*parameters["initial_grid_oversampling_factor"])
    if (design["initialization"]["path_additions"] != model.path_count
            or not np.isclose(spacing,parameters["initial_grid_spacing_hz"])
            or control["outer_sweeps_after_initialization"] != 3
            or control["initialization_counts_as_outer_sweep"]
            or control["convergence_based_early_stopping"]):
        raise ValueError("Review the selected initialization, grid or three-sweep policy.")
    return AlternatingProjectionEstimator(model,spacing,parameters["refinement_tolerance_hz"],3)
