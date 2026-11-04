"""Joint-frame embedded-pilot approximate ML with conditional LS path gains.

Use only public design bounds and received pilot-region observations. Sequential
path addition approximates the joint nonlinear search; it is not a global-ML
solver. Unknown-data leakage stays in the residual. No data equalizer or BER
detector is provided here.
"""

import numpy as np

import afdm
from pilot_layouts import known_pilot_symbols
from pilot_path_search import SequentialEstimator


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
