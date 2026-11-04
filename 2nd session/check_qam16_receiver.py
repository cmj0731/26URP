"""Validate standard Gray hard demapping and estimated MMSE self-gain.

Dense small-frame references test the exact OTFS gain identity. Scalar AWGN
checks validate the demapper against Gaussian decision-region integrals. Eight
actual estimator/solver/bit-recovery cases are short integration diagnostics,
not the project's full equal-bit-quota BER comparison.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from afdm_estimation import build_estimator as build_afdm
from channel_operators import NativeChannelOperator, build_channel_operator
from check_channel_operators import SCHEMES, receive, relative_error, transmit
from dd_a_estimation import build_estimator as build_dd_a
from mmse_demapping import effective_symbol_gain, recover_data_bits
from mmse_problem import PlugInMMSEProblem, build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from ofdm_bem import build_pilot_ls_model
from otfs_estimation import build_estimator as build_otfs
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import BIT_LABELS, CONSTELLATION, hard_demodulate, modulate
from review_afdm_estimator import apply_paths


class SmallMaskedModel:
    """Small mathematical native-frame model for dense gain references only."""

    def __init__(self, channel, mask):
        self.channel, self.data_mask = channel, mask
        self.scheme, self.grid_shape = channel.scheme, channel.grid_shape
        self.data_indices = np.flatnonzero(mask.reshape(-1, order="F"))
        self.data_count = self.data_indices.size
        self.offsets = np.r_[0, np.cumsum(mask.sum(axis=0))]

    def apply(self, data):
        flat = np.zeros(np.prod(self.grid_shape), dtype=complex)
        flat[self.data_indices] = data
        return self.channel.apply(flat.reshape(self.grid_shape, order="F"))

    def adjoint(self, received):
        return self.channel.adjoint(received).reshape(-1, order="F")[self.data_indices]

    def subtract_pilots(self, received):
        return received.copy()

    def block_data_slice(self, block):
        return slice(int(self.offsets[block]), int(self.offsets[block+1]))

    def block_matrix(self, block):
        return self.channel.block_matrix(block)[:, self.data_mask[:, block]]


def dense_gain_controls():
    records = []
    length, blocks, cp = 8, 4, 2
    time = np.arange((length+cp)*blocks)
    taps = np.column_stack([(1+.2j)*np.exp(.03j*time),
                            .2*np.exp(.2j*time), .1*np.exp(-.07j*time)])
    for scheme in SCHEMES:
        mask = np.zeros((length, blocks), dtype=bool)
        mask[[0, 1, 4, 5], :] = True
        if scheme in ("OFDM", "DD-a-OFDM"):
            mask[:, 1::2] = True
        kwargs = {"c1": 5/16, "c2": np.sqrt(2)/64} if scheme == "AFDM" else {}
        for profile in ("full_rank", "zero", "rank_deficient"):
            current_taps = taps.copy() if profile == "full_rank" else np.zeros_like(taps)
            if profile == "rank_deficient":
                indices = np.arange(blocks)*(length+cp)+cp+1
                current_taps[indices, :2] = 1
            channel = NativeChannelOperator(scheme, current_taps,
                block_length=length, block_count=blocks, cp_length=cp, delays=(0, 1, 2), **kwargs)
            model = SmallMaskedModel(channel, mask)
            basis = np.eye(model.data_count, dtype=complex)
            matrix = np.column_stack([model.apply(basis[:, i]).reshape(-1, order="F")
                                      for i in range(model.data_count)])
            for variance in (.025, 0.):
                problem = PlugInMMSEProblem(model, np.zeros((length, blocks), dtype=complex),
                    estimated_noise_variance=variance, data_symbol_energy=1.)
                gain = effective_symbol_gain(problem)
                if variance > 0:
                    gram = matrix.conj().T@matrix
                    expected = np.diag(np.linalg.solve(gram+variance*np.eye(model.data_count), gram)).real
                else:
                    expected = np.diag(np.linalg.pinv(matrix)@matrix).real
                error = float(np.max(abs(gain-expected)))
                assert error < 1e-11
                records.append({"scheme": scheme, "variance": variance, "control_profile": profile,
                                "full_dense_gain_max_absolute_error": error})
    return records


def gaussian_theory(eb_n0_db):
    """Exact bit error probability by integrating all 16 decision regions."""
    sigma = math.sqrt(1/(8*10**(eb_n0_db/10)))
    levels = np.array([-3., -1., 1., 3.])/np.sqrt(10)
    edges = np.r_[-np.inf, (levels[:-1]+levels[1:])/2, np.inf]
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    def cdf(x):
        return .5*math.erfc(-x/math.sqrt(2))
    total = 0.
    for sent in range(4):
        for detected in range(4):
            probability = cdf((edges[detected+1]-levels[sent])/sigma)-cdf((edges[detected]-levels[sent])/sigma)
            total += probability*np.count_nonzero(gray[sent] != gray[detected])/8
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    assert np.array_equal(hard_demodulate(modulate(BIT_LABELS.reshape(-1))), BIT_LABELS.reshape(-1))
    assert abs(np.mean(abs(CONSTELLATION)**2)-1) < 1e-14
    assert abs(np.mean(CONSTELLATION)) < 1e-14
    adjacency = 0
    for i in range(16):
        for j in range(i+1, 16):
            if np.isclose(abs(CONSTELLATION[i]-CONSTELLATION[j]), 2/np.sqrt(10)):
                assert np.count_nonzero(BIT_LABELS[i] != BIT_LABELS[j]) == 1
                adjacency += 1
    assert adjacency == 24
    gains = np.linspace(.01, 1, 16)
    assert np.array_equal(hard_demodulate(gains*CONSTELLATION, gains), BIT_LABELS.reshape(-1))
    assert np.all(hard_demodulate(np.zeros(16, dtype=complex), np.zeros(16)) == 0)
    controls = dense_gain_controls()
    print("Gray adjacency/roundtrip and 24 small full-native gain references passed.", flush=True)

    seeds = np.random.SeedSequence(20261104).spawn(6)
    data_rng, channel_rng, noise_rng, calibration_rng, awgn_bits_rng, awgn_noise_rng = [
        np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
    awgn = []
    bit_count = 4*131072
    awgn_bits = awgn_bits_rng.integers(0, 2, bit_count, dtype=np.uint8)
    awgn_data = modulate(awgn_bits)
    unit = (awgn_noise_rng.standard_normal(awgn_data.size)
            + 1j*awgn_noise_rng.standard_normal(awgn_data.size))/np.sqrt(2)
    for eb_n0 in (0, 5, 10):
        variance = 1/(4*10**(eb_n0/10))
        raw = (awgn_data+np.sqrt(variance)*unit)/(1+variance)
        estimated = hard_demodulate(raw, 1/(1+variance))
        errors = (estimated != awgn_bits).reshape(-1, 4).mean(axis=1)
        measured = float(errors.mean())
        theory = gaussian_theory(eb_n0)
        standard_error = float(errors.std(ddof=1)/np.sqrt(errors.size))
        assert abs(measured-theory) < 5*standard_error
        awgn.append({"Eb_N0_db": eb_n0, "control_bits": bit_count,
                     "measured_control_BER": measured, "exact_Gaussian_region_BER": theory,
                     "independent_symbol_standard_error": standard_error})
    print("Scalar Gray-16QAM AWGN controls agree with exact Gaussian-region integrals.", flush=True)

    public = {key: settings[key] for key in ("waveform_design", "prefix_policy", "sampling_model",
        "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
        "common_symbol_modulation")}
    tight = copy.deepcopy(public)
    strict_tolerance = public["receiver_equalization_policy"]["numerical_solver"]["validation_relative_normal_residual_tolerance"]
    tight["receiver_equalization_policy"]["numerical_solver"]["relative_normal_residual_tolerance"] = strict_tolerance
    bits = data_rng.integers(0, 2, 49152, dtype=np.uint8)
    data = modulate(bits)
    assert data.size == 12288
    grids = {scheme: build_grid(scheme, data, scenario=public) for scheme in SCHEMES}
    signals = {scheme: transmit(scheme, grids[scheme], public) for scheme in SCHEMES}
    measurement = build_silent_noise_measurement(public)
    # Generator variance below is not supplied to channel estimation/demapping.
    variance = 1/(4*10**(20/10))
    calibration = (calibration_rng.standard_normal(measurement.interval_sample_count)
                   + 1j*calibration_rng.standard_normal(measurement.interval_sample_count))*np.sqrt(variance/2)
    measured = measurement.estimate(calibration)
    noise = (noise_rng.standard_normal(signals["OFDM"].size)
              + 1j*noise_rng.standard_normal(signals["OFDM"].size))*np.sqrt(variance/2)
    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., channel_rng.uniform(0, 2*np.pi, 2)])
    fs = public["sampling_model"]["sample_rate_hz"]
    bem = build_pilot_ls_model(public)
    estimators = {"AFDM": build_afdm(public), "OTFS": build_otfs(public), "DD-a-OFDM": build_dd_a(public)}
    integrations = []
    for label, magnitude in (("A", 0.), ("B-4", 46875.)):
        for scheme in SCHEMES:
            received = apply_paths(signals[scheme], delays, gains, [0., magnitude, -magnitude], fs)+noise
            if scheme == "OFDM":
                taps = bem.reconstruct_taps(bem.estimate(received))
            else:
                estimate = estimators[scheme].estimate(received)
                taps = estimators[scheme].model.reconstruct_taps(estimate)
            model = build_pilot_data_model(build_channel_operator(scheme, public, taps), public)
            problem = build_mmse_problem(model, receive(scheme, received, public), measured, public)
            solved = solve_mmse_problem(problem, public)
            recovered, gain = recover_data_bits(problem, solved, return_gain=True)
            assert recovered.shape == bits.shape and recovered.dtype == np.uint8
            assert gain.shape == (12288,) and np.all(gain > 0)
            assert np.array_equal(recovered, hard_demodulate(solved.symbols/gain))
            strict = (solve_mmse_problem(problem, tight) if scheme == "OTFS" else solved)
            assert strict.relative_normal_residual <= strict_tolerance
            if scheme != "OTFS":
                assert strict.diagnostics["max_block_relative_normal_residual"] <= strict_tolerance
            strict_bits = hard_demodulate(strict.require_converged(), gain)
            assert np.array_equal(recovered, strict_bits)
            integrations.append({"condition": label, "scheme": scheme,
                "diagnostic_Eb_N0_db_generation_only": 20, "diagnostic_bits": int(bits.size),
                "solver_converged": solved.converged, "minimum_estimated_self_gain": float(gain.min()),
                "maximum_estimated_self_gain": float(gain.max()),
                "gain_reference_vs_division_bit_disagreements": 0,
                "stricter_solver_bit_disagreements": 0,
                "diagnostic_bit_errors": int(np.count_nonzero(recovered != bits))})
            print(f"{label} {scheme}: estimated gain and 49152-bit demapping checked.", flush=True)
    report = {"scope": "Standard common Gray-16QAM hard decision and MMSE gain handling validation.",
              "mapping": "[I_MSB,I_LSB,Q_MSB,Q_LSB]; axis labels 00/01/11/10 at -3/-1/+1/+3; scale 1/sqrt(10). This is a common Cartesian Gray convention, not a claim of matching every library's label orientation.",
              "gray_adjacent_pairs_checked": adjacency,
              "small_dense_gain_controls": controls, "scalar_AWGN_controls": awgn,
              "actual_receiver_integrations": integrations,
              "receiver_inputs": "Converged raw MMSE symbols, that receiver's estimated data-channel model and measured noise regularization only. Actual data/noise/channel are generation/scoring inputs; no data-fitted gain, true-channel gain or feedback is used.",
              "OTFS_gain_identity": "The fixed DD data mask shares delay rows across Doppler columns. Unitary inverse-Zak modulation gives diag(G_DD*A_DD)[q,k] = mean_b diag(G_time,b*A_time,b)[q]. Exact averaging of local time-data gains computes the full-DD diagonal; the raw symbol solve remains full-frame CG/LSQR.",
              "limits": ["Eight actual cases use one short frame per A/B-4 setting. Diagnostic bit error counts do not establish the project's BER curves or satisfy its full common quota.",
                         "Scalar AWGN tests validate only mapping/gain/hard decisions against theory, not the four-waveform LOS/common-Doppler baseline.",
                         "The standard per-symbol hard detector is not a claim of optimal joint ML detection under residual ICI or channel uncertainty.",
                         "The gain belongs to the estimated working channel. Actual residual gain/ICI/pilot errors remain in the recovered bits.",
                         "No waveform, pilot, energy, noise calibration, channel estimator, solver tolerance, bit quota, tail policy or confidence interval is changed."]}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Mapping, gain, AWGN and eight receiver bit-recovery integrations passed.")


if __name__ == "__main__":
    main()
