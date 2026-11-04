"""Full-frame native DD pilot-template path estimation for per-slot-CP OTFS.

Fit continuous fractional Doppler and joint complex gains using public bounds
and received samples only. This selected sequential approximate-ML/LS design
is our adaptation, not the reference paper's threshold-based tap estimator.
No data equalizer or BER detector is provided here.
"""

import numpy as np

import otfs
from pilot_layouts import build_masks, known_pilot_symbols
from pilot_path_search import SequentialEstimator


class PilotModel:
    def __init__(self, settings):
        spec = settings["waveform_design"]["OTFS"]
        layout = settings["pilot_comparison_policy"]["otfs_dd_layout"]
        information = settings["receiver_information_policy"]
        self.length, self.blocks = spec["delay_bin_count"], spec["doppler_bin_count"]
        self.cp = settings["prefix_policy"]["prefix_samples_per_block"]
        self.fs = settings["sampling_model"]["sample_rate_hz"]
        self.delays = tuple(range(information["delay_support"]["minimum_excess_delay_samples"],
                                  information["delay_support"]["maximum_excess_delay_samples"]+1))
        self.maximum_doppler = information["residual_doppler_support"]["maximum_absolute_residual_doppler_hz"]
        self.path_count = information["known_path_count_including_los"]
        self.pilot_delay, self.pilot_doppler = layout["pilot_delay_index"], layout["pilot_doppler_index"]
        self.pilot = known_pilot_symbols("OTFS", scenario=settings)[0]
        design = settings["receiver_equalization_policy"]["channel_estimation_selection"]["OTFS"]["design"]
        if design["processing_window"]["slots_per_window"] != self.blocks:
            raise ValueError("This implementation requires the selected full-frame window.")
        if (min(self.delays) < 0 or max(self.delays) > self.cp
                or self.pilot_delay+max(self.delays) >= self.length):
            raise ValueError("This pilot formula requires CP-covered delays without pilot delay-row wrap.")
        self.rows = self.pilot_delay+np.array(self.delays)
        masks = build_masks("OTFS", scenario=settings)
        if masks["data"].shape != (self.length,self.blocks) or int(masks["pilot"].sum()) != 1:
            raise ValueError("Review the selected single-pilot grid after a layout change.")
        # Every possible input delay row that could reach our observation rows
        # must be free of data. Doppler mixing stays within a delay row here.
        for delay in self.delays:
            if np.any(masks["data"][(self.rows-delay) % self.length]):
                raise ValueError("The selected pilot observation region is not isolated from data.")
        self.starts = (np.arange(self.blocks)*(self.length+self.cp)+self.cp)/self.fs
        self.sample_count = (self.length+self.cp)*self.blocks

    def observations(self, received_samples):
        grid = otfs.demodulate(received_samples,self.length,self.blocks,cp_length=self.cp)
        return grid[self.rows].reshape(-1,order="F")

    def atoms(self, paths):
        paths = tuple(paths)
        if not paths:
            raise ValueError("At least one path hypothesis is required.")
        matrix = np.zeros((self.rows.size,self.blocks,len(paths)),dtype=complex)
        slot = np.arange(self.blocks)
        pilot_slots = self.pilot/np.sqrt(self.blocks)*np.exp(2j*np.pi*self.pilot_doppler*slot/self.blocks)
        for column,(delay,frequency) in enumerate(paths):
            if delay not in self.delays or not np.isfinite(frequency) or abs(frequency) > self.maximum_doppler:
                raise ValueError("Path hypotheses must satisfy the public delay/Doppler bounds.")
            row = self.pilot_delay+delay
            times = self.starts+row/self.fs
            # The full unitary FFT is the fractional-Doppler response, with all
            # 64 bins retained. Gains are referenced to the first CP sample t=0.
            matrix[self.delays.index(delay),:,column] = np.fft.fft(
                pilot_slots*np.exp(2j*np.pi*frequency*times),norm="ortho")
        return matrix.reshape((-1,len(paths)),order="F")

    def fit_gains(self, observations, paths):
        paths = tuple(paths)
        values = np.asarray(observations,dtype=np.complex128)
        if values.shape != (self.rows.size*self.blocks,) or not np.isfinite(values).all():
            raise ValueError("One finite full-frame pilot observation vector is required.")
        matrix = self.atoms(paths)
        gains,_,rank,singular = np.linalg.lstsq(matrix,values,rcond=None)
        if rank != matrix.shape[1]:
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


def build_estimator(settings):
    selected = settings["receiver_equalization_policy"]["channel_estimation_selection"]["OTFS"]
    if selected["algorithm"] != "DD_pilot_fractional_Doppler_sequential_approximate_ML_with_joint_LS_gains":
        raise ValueError("The selected OTFS estimator does not match this implementation.")
    parameters = selected["design"]["search_parameters"]
    model = PilotModel(settings)
    duration = model.sample_count/model.fs
    spacing = 1/(duration*parameters["initial_grid_oversampling_factor"])
    if parameters["path_additions"] != model.path_count or not np.isclose(spacing,parameters["initial_grid_spacing_hz"]):
        raise ValueError("Review the selected path count or search grid after a design change.")
    return SequentialEstimator(model,spacing,parameters["refinement_tolerance_hz"])
