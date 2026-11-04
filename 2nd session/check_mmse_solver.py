"""Verify direct/CG/zero-noise LS solves and 1e-8 versus 1e-10 accuracy.

Small dense and singular controls validate the algorithms independently. Actual
waveform cases use original pilot observations and separate calibration noise.
Nearest-16QAM labels are diagnostic stability probes, not selected bit decisions
or a BER experiment, and do not alter receiver output scaling.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from afdm_estimation import build_estimator as build_afdm
from channel_operators import build_channel_operator
from check_channel_operators import SCHEMES, receive, relative_error, transmit
from dd_a_estimation import build_estimator as build_dd_a
from mmse_problem import PlugInMMSEProblem, build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from ofdm_bem import build_pilot_ls_model
from otfs_estimation import build_estimator as build_otfs
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from review_afdm_estimator import apply_paths


class DenseControlModel:
    """Tiny rectangular mathematical stand-in; not another waveform design."""

    scheme = "OTFS"

    def __init__(self, matrix):
        self.matrix = matrix
        self.grid_shape = (matrix.shape[0], 1)
        self.data_count = matrix.shape[1]

    def subtract_pilots(self, received):
        return np.asarray(received, dtype=complex).copy()

    def apply(self, data):
        return (self.matrix@data).reshape(self.grid_shape)

    def adjoint(self, grid):
        return self.matrix.conj().T@grid.reshape(-1)


def summarize(result):
    ranks = [b["least_squares_rank"] for b in result.diagnostics.get("blocks", [])
             if b["least_squares_rank"] is not None]
    return {"converged": result.converged, "method": result.method,
            "iterations": result.iterations, "relative_normal_residual": result.relative_normal_residual,
            "reason": result.reason,
            "maximum_local_relative_normal_residual": result.diagnostics.get("max_block_relative_normal_residual"),
            "minimum_local_least_squares_rank": min(ranks) if ranks else None}


def mathematical_controls(settings, rng):
    controls = []
    matrix = (rng.standard_normal((32, 12))+1j*rng.standard_normal((32, 12)))/np.sqrt(32)
    observed = (rng.standard_normal((32, 1))+1j*rng.standard_normal((32, 1)))/np.sqrt(2)
    for name, variance, rank_deficient in (("positive_noise_CG", .025, False),
                                          ("small_positive_noise_CG", 1e-12, False),
                                          ("zero_noise_LSQR", 0., False),
                                          ("rank_deficient_zero_noise_LSQR", 0., True)):
        current = matrix.copy()
        if rank_deficient:
            current[:, 10:] = current[:, :2]
        problem = PlugInMMSEProblem(DenseControlModel(current), observed,
                    estimated_noise_variance=variance, data_symbol_energy=1.)
        result = solve_mmse_problem(problem, settings)
        result.require_converged()
        if variance > 0:
            expected = np.linalg.solve(current.conj().T@current+variance*np.eye(12),
                                       current.conj().T@observed[:, 0])
            assert result.method == "full_frame_CG"
        else:
            expected = np.linalg.lstsq(current, observed[:, 0], rcond=None)[0]
            assert result.method == "full_frame_LSQR"
        error = relative_error(result.symbols, expected)
        assert error < 1e-7
        controls.append({"name": name, **summarize(result), "dense_reference_solution_relative_error": error})
    zero = PlugInMMSEProblem(DenseControlModel(np.zeros((32, 12), dtype=complex)), observed,
                           estimated_noise_variance=0., data_symbol_energy=1.)
    zero_result = solve_mmse_problem(zero, settings)
    assert zero_result.converged and zero_result.iterations == 0 and np.all(zero_result.symbols == 0)
    controls.append({"name": "zero_channel_minimum_norm", **summarize(zero_result)})
    capped = copy.deepcopy(settings)
    capped["receiver_equalization_policy"]["numerical_solver"]["maximum_iterations"] = 1
    for variance in (.025, 0.):
        problem = PlugInMMSEProblem(DenseControlModel(matrix), observed,
                                  estimated_noise_variance=variance, data_symbol_energy=1.)
        failure = solve_mmse_problem(problem, capped)
        assert not failure.converged and failure.reason == "maximum_iterations_reached"
        try:
            failure.require_converged()
        except RuntimeError:
            pass
        else:
            raise AssertionError("A numerical failure must not enter bit scoring.")
        controls.append({"name": "one_iteration_failure_guard", **summarize(failure)})
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    public = {key: settings[key] for key in ("waveform_design", "prefix_policy", "sampling_model",
        "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
        "common_symbol_modulation")}
    tight = copy.deepcopy(public)
    policy = public["receiver_equalization_policy"]["numerical_solver"]
    strict_tolerance = policy["validation_relative_normal_residual_tolerance"]
    tight["receiver_equalization_policy"]["numerical_solver"]["relative_normal_residual_tolerance"] = strict_tolerance
    seeds = np.random.SeedSequence(20261104).spawn(5)
    data_rng, channel_rng, noise_rng, calibration_rng, control_rng = [
        np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
    controls = mathematical_controls(public, control_rng)
    print("Dense full-rank/rank-deficient/zero-noise and failure-guard controls passed.", flush=True)
    fs = public["sampling_model"]["sample_rate_hz"]
    count = public["prefix_policy"]["transmitted_samples_per_frame"]
    alphabet = (np.array([-3, -1, 1, 3])[:, None]
                + 1j*np.array([-3, -1, 1, 3])[None, :]).reshape(-1)/np.sqrt(10)
    data = data_rng.choice(alphabet, 12288)
    grids = {scheme: build_grid(scheme, data, scenario=public) for scheme in SCHEMES}
    signals = {scheme: transmit(scheme, grids[scheme], public) for scheme in SCHEMES}
    pilot_settings = {"pilot_comparison_policy": public["pilot_comparison_policy"]}
    # A separate cyclic [1,1] two-tap control has an analytically zero TF bin.
    # This is a numerical rank control, not the selected physical path profile.
    taps = np.zeros((count, 6), dtype=complex)
    taps[:, :2] = 1
    singular_model = build_pilot_data_model(build_channel_operator("OFDM", public, taps), pilot_settings)
    singular_received = singular_model.channel.apply(grids["OFDM"])
    singular_problem = build_mmse_problem(singular_model, singular_received, 0., public)
    singular_result = solve_mmse_problem(singular_problem, public)
    singular_result.require_converged()
    expected_grid = singular_model.embed_data(data)
    expected_grid[128, :] = 0
    expected = expected_grid.reshape(-1, order="F")[singular_model.data_indices]
    error = relative_error(singular_result.symbols, expected)
    assert error < 1e-10 and min(b["least_squares_rank"] for b in singular_result.diagnostics["blocks"]) == 128
    assert any(b["least_squares_rank"] == 255 for b in singular_result.diagnostics["blocks"])
    controls.append({"name": "local_SVD_singular_TF_bin", **summarize(singular_result),
                     "analytic_minimum_norm_solution_relative_error": error})

    measurement = build_silent_noise_measurement(public)
    unit_calibration = (calibration_rng.standard_normal(measurement.interval_sample_count)
                        + 1j*calibration_rng.standard_normal(measurement.interval_sample_count))/np.sqrt(2)
    unit_noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., channel_rng.uniform(0, 2*np.pi, 2)])
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    bem = build_pilot_ls_model(public)
    estimators = {"AFDM": build_afdm(public), "OTFS": build_otfs(public), "DD-a-OFDM": build_dd_a(public)}
    records = []
    for label, magnitude in zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]):
        clean = {scheme: apply_paths(signals[scheme], delays, gains, [0., magnitude, -magnitude], fs)
                 for scheme in SCHEMES}
        for eb_n0 in (None, 0, 20, 30):
            # True variance is used for generation only, including the separate
            # silent observation. Noiseless cases rerun their own estimators.
            variance = 0. if eb_n0 is None else 1/(4*10**(eb_n0/10))
            measured = measurement.estimate(np.sqrt(variance)*unit_calibration)
            for scheme in SCHEMES:
                received = clean[scheme]+np.sqrt(variance)*unit_noise
                original = received.copy()
                if scheme == "OFDM":
                    estimated_taps = bem.reconstruct_taps(bem.estimate(received))
                else:
                    estimate = estimators[scheme].estimate(received)
                    estimated_taps = estimators[scheme].model.reconstruct_taps(estimate)
                model = build_pilot_data_model(build_channel_operator(scheme, public, estimated_taps), pilot_settings)
                problem = build_mmse_problem(model, receive(scheme, received, public), measured, public)
                result = solve_mmse_problem(problem, public)
                # Direct solves have no tolerance-dependent iterations. Reuse
                # their result if it already meets the stricter global AND local
                # residual targets; iterative methods are rerun at 1e-10.
                reuse = (scheme != "OTFS" and result.converged
                         and result.relative_normal_residual <= strict_tolerance
                         and result.diagnostics["max_block_relative_normal_residual"] <= strict_tolerance)
                strict = result if reuse else solve_mmse_problem(problem, tight)
                assert np.array_equal(received, original)
                record = {"condition": label, "scheme": scheme, "diagnostic_Eb_N0_db_generation_only": eb_n0,
                          "noiseless": eb_n0 is None, "measured_noise_variance": measured,
                          "nominal": summarize(result), "strict": summarize(strict),
                          "direct_result_already_meets_stricter_target": reuse}
                if result.converged and strict.converged:
                    base, reference = result.require_converged(), strict.require_converged()
                    record["stricter_solution_relative_L2_difference"] = relative_error(base, reference)
                    record["maximum_absolute_symbol_difference"] = float(np.max(abs(base-reference)))
                    labels = np.argmin(abs(base[:, None]-alphabet[None, :]), axis=1)
                    strict_labels = np.argmin(abs(reference[:, None]-alphabet[None, :]), axis=1)
                    record["diagnostic_nearest_constellation_label_disagreements"] = int(np.count_nonzero(labels != strict_labels))
                records.append(record)
            print(f"{label}: {'no AWGN' if eb_n0 is None else str(eb_n0)+' dB'} four receiver solves checked.", flush=True)
    failures = [row for row in records if not row["nominal"]["converged"] or not row["strict"]["converged"]]
    disagreements = sum(row.get("diagnostic_nearest_constellation_label_disagreements", 0) for row in records)
    report = {"scope": "Numerical direct/CG/zero-noise least-squares validation, not a BER experiment.",
              "numpy_version": np.__version__, "diagnostic_seed": 20261104,
              "receiver_inputs": "Own pilot-estimated channel, original native observations, separately measured calibration noise power, and public Es/numerical settings. No generator truth is read by the solver.",
              "mathematical_controls": controls, "actual_receiver_cases": records,
              "actual_receiver_case_count": len(records), "failed_case_count": len(failures),
              "total_diagnostic_nearest_constellation_label_disagreements": disagreements,
              "limits": ["One diagnostic frame at each A/B level and noiseless/0/20/30-dB setting; not the common BER bit quota or a Monte Carlo BER comparison.",
                         "Nearest-16QAM labels compare numerical stability only. Production output gain/bit mapping/decisions remain unselected and no BER is counted.",
                         "A small normal-equation residual is not a universal bound on symbol error; dense reference controls and stricter-tolerance reruns provide complementary evidence.",
                         "Zero-start LSQR and local SVD handle zero-noise least squares without a noise floor. Rank thresholds follow NumPy rcond=None for local SVD.",
                         "No physical path, pilot, energy, calibration, estimator, waveform extent or measured-bit budget is retuned."]}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"{len(records)} cases; numerical failures={len(failures)}; stability label differences={disagreements}.")
    assert not failures
    assert disagreements == 0


if __name__ == "__main__":
    main()
