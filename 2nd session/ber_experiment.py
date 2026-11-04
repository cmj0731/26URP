"""Reproducible paired-frame generation and the agreed BER receivers.

Generator truth is confined to frame generation. Receiver builders receive only
public waveform/pilot/receiver design and original received observations.
"""

import math
import numpy as np

import afdm
import otfs
from afdm_estimation import build_estimator as build_afdm
from channel_operators import build_channel_operator
from cyclic_prefix import add_cyclic_prefix, remove_cyclic_prefix
from dd_a_estimation import build_estimator as build_dd_a
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from ofdm_bem import build_pilot_ls_model
from otfs_estimation import build_estimator as build_otfs
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import hard_demodulate, modulate

SCHEMES = ("OFDM", "AFDM", "OTFS", "DD-a-OFDM")
POINTS = (None, 0, 5, 10, 15, 20, 25, 30)
_CONTEXT = None


def transmit(scheme, grid, public, cp):
    if scheme == "AFDM":
        chirps = public["waveform_design"]["AFDM"]["chirp_parameters"]
        return afdm.modulate(grid, chirps["c1"], chirps["c2"], cp_length=cp)
    if scheme == "OTFS":
        return otfs.modulate(grid, cp_length=cp)
    useful = np.fft.ifft(grid, axis=0, norm="ortho").reshape(-1, order="F")
    return add_cyclic_prefix(useful, grid.shape[0], cp)


def receive(scheme, samples, public, cp):
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    if scheme == "AFDM":
        chirps = public["waveform_design"]["AFDM"]["chirp_parameters"]
        return afdm.demodulate(samples, n, blocks, chirps["c1"], chirps["c2"], cp_length=cp)
    if scheme == "OTFS":
        return otfs.demodulate(samples, n, blocks, cp_length=cp)
    useful = remove_cyclic_prefix(samples, n, cp).reshape((n, blocks), order="F")
    return np.fft.fft(useful, axis=0, norm="ortho")


def apply_paths(samples, delays, gains, frequencies, fs):
    result = np.zeros_like(samples)
    time = np.arange(samples.size)/fs
    for delay, gain, frequency in zip(delays, gains, frequencies):
        result[delay:] += gain*np.exp(2j*np.pi*frequency*time[delay:])*samples[:samples.size-delay]
    return result


def theory_16qam(eb_n0_db):
    """Exact Cartesian Gray-4PAM decision-region integral for either axis."""
    sigma = math.sqrt(1/(8*10**(eb_n0_db/10)))
    levels = np.array([-3., -1., 1., 3.])/np.sqrt(10)
    edges = np.r_[-np.inf, (levels[:-1]+levels[1:])/2, np.inf]
    labels = ((0, 0), (0, 1), (1, 1), (1, 0))
    cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
    def probability(lower, upper):
        if lower > 0:
            return cdf(-lower)-cdf(-upper)
        return cdf(upper)-cdf(lower)
    return sum(probability((edges[j]-levels[i])/sigma, (edges[j+1]-levels[i])/sigma)
               *sum(a != b for a, b in zip(labels[i], labels[j]))/8
               for i in range(4) for j in range(4))


def initialize_worker(settings, stage, root_seed):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: settings[key] for key in keys}
    context = {"settings": settings, "public": public, "stage": stage, "seed": root_seed}
    if stage == "multipath":
        context["bem"] = build_pilot_ls_model(public)
        context["estimators"] = {"AFDM": build_afdm(public), "OTFS": build_otfs(public), "DD-a-OFDM": build_dd_a(public)}
        context["measurement"] = build_silent_noise_measurement(public)
    _CONTEXT = context


def compute_frame(job):
    index, measured_bits = job
    c = _CONTEXT
    settings, public = c["settings"], c["public"]
    streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(c["seed"], spawn_key=(index, role))))
               for role in range(4)]
    data_rng, channel_rng, noise_rng, calibration_rng = streams
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    fs = public["sampling_model"]["sample_rate_hz"]
    baseline = c["stage"] == "baseline"
    capacity = n*blocks*4 if baseline else 49152
    bits = data_rng.integers(0, 2, capacity, dtype=np.uint8)
    symbols = modulate(bits)
    grids = ({scheme: symbols.reshape((n, blocks), order="F") for scheme in SCHEMES}
             if baseline else {scheme: build_grid(scheme, symbols, scenario=public) for scheme in SCHEMES})
    cp = 0 if baseline else public["prefix_policy"]["prefix_samples_per_block"]
    signals = {scheme: transmit(scheme, grids[scheme], public, cp) for scheme in SCHEMES}
    count = signals["OFDM"].size
    unit_noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    if baseline:
        doppler = settings["doppler_comparison"]
        conditions = [("peak", doppler["low_doppler_hz"]), ("exit", doppler["high_doppler_hz"])]
    else:
        profile = settings["channel_model"]["multipath_followup"]
        paths = profile["path_delay_power_doppler_profile"]
        delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
        powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
        gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., channel_rng.uniform(0, 2*np.pi, 2)])
        sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
        conditions = list(zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]))
        unit_calibration = (calibration_rng.standard_normal(c["measurement"].interval_sample_count)
                            + 1j*calibration_rng.standard_normal(c["measurement"].interval_sample_count))/np.sqrt(2)
    errors = np.zeros((len(conditions), len(POINTS), len(SCHEMES)), dtype=np.int64)
    iterations = np.zeros_like(errors)
    maximum_residual = np.zeros(len(SCHEMES))
    for condition_index, (label, frequency) in enumerate(conditions):
        if baseline:
            phase = np.exp(2j*np.pi*frequency*np.arange(count)/fs)
            clean = {scheme: signals[scheme]*phase for scheme in SCHEMES}
        else:
            clean = {scheme: apply_paths(signals[scheme], delays, gains, [0., frequency, -frequency], fs)
                     for scheme in SCHEMES}
        for point_index, eb_n0 in enumerate(POINTS):
            variance = 0. if eb_n0 is None else 1/(4*10**(eb_n0/10))
            if not baseline:
                measured_variance = c["measurement"].estimate(np.sqrt(variance)*unit_calibration)
            for scheme_index, scheme in enumerate(SCHEMES):
                received = clean[scheme]+np.sqrt(variance)*unit_noise
                if baseline:
                    # Unit-gain LOS and exact common-frequency correction. With
                    # gain-aware hard decisions, identity-channel MMSE reduces
                    # exactly to ordinary nearest-constellation detection. No
                    # generator variance is needed by the baseline receiver.
                    corrected = received*phase.conj()
                    decoded = hard_demodulate(receive(scheme, corrected, public, 0).reshape(-1, order="F"))
                else:
                    if scheme == "OFDM":
                        taps = c["bem"].reconstruct_taps(c["bem"].estimate(received))
                    else:
                        estimator = c["estimators"][scheme]
                        taps = estimator.model.reconstruct_taps(estimator.estimate(received))
                    model = build_pilot_data_model(build_channel_operator(scheme, public, taps), public)
                    problem = build_mmse_problem(model, receive(scheme, received, public, cp), measured_variance, public)
                    solved = solve_mmse_problem(problem, public, include_effective_gain=True)
                    solved.require_converged()
                    decoded = recover_data_bits(problem, solved)
                    iterations[condition_index, point_index, scheme_index] = solved.iterations
                    maximum_residual[scheme_index] = max(maximum_residual[scheme_index], solved.relative_normal_residual)
                errors[condition_index, point_index, scheme_index] = np.count_nonzero(decoded[:measured_bits] != bits[:measured_bits])
    return {"frame_index": index, "measured_bits": measured_bits, "errors": errors.tolist(),
            "iterations": iterations.tolist(), "max_relative_normal_residual": maximum_residual.tolist()}
