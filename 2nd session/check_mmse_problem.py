"""Check plug-in MMSE equations and measured-noise/estimated-channel inputs.

No production data solver or decisions are selected. Scalar closed-form controls
are mathematical references; the eight receiver integrations use pilot estimates
and a separate silent-interval noise measurement, never true channel/noise inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from afdm_estimation import build_estimator as build_afdm
from channel_operators import build_channel_operator
from check_channel_operators import SCHEMES, path_taps, receive, relative_error, transmit
from dd_a_estimation import build_estimator as build_dd_a
from mmse_problem import build_mmse_problem
from noise_estimation import build_silent_noise_measurement
from ofdm_bem import build_pilot_ls_model
from otfs_estimation import build_estimator as build_otfs
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from review_afdm_estimator import apply_paths


def check_equations(problem, vector, direction):
    normal = problem.normal(vector)
    other = problem.normal(direction)
    energy = np.linalg.norm(problem.model.apply(vector))**2
    expected = energy+problem.regularization*np.linalg.norm(vector)**2
    quadratic_error = float(abs(np.vdot(vector, normal)-expected)/max(expected, 1e-300))
    hermitian_error = float(abs(np.vdot(direction, normal)-np.vdot(other, vector))
                            /max(np.linalg.norm(direction)*np.linalg.norm(normal), 1e-300))
    step = 1e-3
    measured_derivative = (problem.objective(vector+step*direction)
                           - problem.objective(vector-step*direction))/(2*step)
    derivative = 2*np.vdot(direction, normal-problem.rhs).real
    gradient_error = float(abs(measured_derivative-derivative)/max(abs(derivative), 1.0))
    block_error = rhs_error = 0.0
    if problem.scheme == "OTFS":
        try:
            problem.block_system(0)
        except ValueError:
            pass
        else:
            raise AssertionError("The full DD problem must not be split into block solves.")
    else:
        for block in (0, 1, 63):
            matrix, rhs = problem.block_system(block)
            selected = problem.model.block_data_slice(block)
            block_error = max(block_error, relative_error(matrix@vector[selected], normal[selected]))
            rhs_error = max(rhs_error, relative_error(rhs, problem.rhs[selected]))
    assert max(quadratic_error, hermitian_error, block_error, rhs_error) < 1e-11
    assert gradient_error < 1e-7
    return {"normal_quadratic_energy_identity_relative_error": quadratic_error,
            "normal_hermitian_inner_product_relative_error": hermitian_error,
            "objective_directional_derivative_relative_error": gradient_error,
            "block_normal_action_relative_error": block_error,
            "block_rhs_relative_error": rhs_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    public = {key: settings[key] for key in ("waveform_design", "prefix_policy", "sampling_model",
        "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
        "common_symbol_modulation")}
    statistical_settings = {key: public[key] for key in ("receiver_equalization_policy", "common_symbol_modulation")}
    pilot_settings = {"pilot_comparison_policy": public["pilot_comparison_policy"]}
    fs = public["sampling_model"]["sample_rate_hz"]
    count = public["prefix_policy"]["transmitted_samples_per_frame"]
    seeds = np.random.SeedSequence(20261104).spawn(4)
    data_rng, channel_rng, noise_rng, calibration_rng = [
        np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
    alphabet = (np.array([-3, -1, 1, 3])[:, None]
                + 1j*np.array([-3, -1, 1, 3])[None, :]).reshape(-1)/np.sqrt(10)
    prior_energy = float(np.mean(abs(alphabet)**2))
    independent_pairs = np.array(np.meshgrid(alphabet, alphabet)).reshape(2, -1)
    prior_covariance = independent_pairs@independent_pairs.conj().T/independent_pairs.shape[1]
    assert abs(alphabet.mean()) < 1e-14 and abs(prior_energy-1) < 1e-14
    assert np.max(abs(prior_covariance-np.eye(2))) < 1e-14
    assert abs(np.mean(alphabet**2)) < 1e-14
    data = data_rng.choice(alphabet, 12288)
    vector = (data_rng.standard_normal(data.size)+1j*data_rng.standard_normal(data.size))/np.sqrt(2)
    direction = (data_rng.standard_normal(data.size)+1j*data_rng.standard_normal(data.size))/np.sqrt(2)
    direction /= np.linalg.norm(direction)
    grids = {scheme: build_grid(scheme, data, scenario=public) for scheme in SCHEMES}
    signals = {scheme: transmit(scheme, grids[scheme], public) for scheme in SCHEMES}
    probe_grid = (data_rng.standard_normal((256, 64))
                  + 1j*data_rng.standard_normal((256, 64)))/np.sqrt(2)
    scalar_controls = []
    gain = .8+.3j
    taps = np.zeros((count, 6), dtype=complex)
    taps[:, 0] = gain
    for scheme in SCHEMES:
        model = build_pilot_data_model(build_channel_operator(scheme, public, taps), pilot_settings)
        for variance in (0.0, .025):
            original = probe_grid.copy()
            problem = build_mmse_problem(model, original, variance, statistical_settings)
            assert np.array_equal(original, probe_grid)
            corrected = probe_grid-gain*model.pilot_grid
            expected_rhs = gain.conjugate()*corrected.reshape(-1, order="F")[model.data_indices]
            rhs_error = relative_error(problem.rhs, expected_rhs)
            normal_error = relative_error(problem.normal(vector), (abs(gain)**2+variance)*vector)
            # A scalar known-model control has an analytic solution; this does
            # not select a numerical solver for the full receiver.
            analytic_solution = expected_rhs/(abs(gain)**2+variance)
            analytic_residual = relative_error(problem.normal(analytic_solution), problem.rhs)
            assert max(rhs_error, normal_error, analytic_residual) < 1e-11
            scalar_controls.append({"scheme": scheme, "control_variance": variance,
                "scalar_rhs_relative_error": rhs_error, "scalar_normal_relative_error": normal_error,
                "analytic_solution_normal_residual": analytic_residual,
                **check_equations(problem, vector, direction)})
    # No hidden floor: zero channel and zero measured power give a singular
    # zero normal operator, whose eventual solve needs a separate design choice.
    zero_model = build_pilot_data_model(build_channel_operator("OTFS", public, np.zeros_like(taps)), pilot_settings)
    zero_problem = build_mmse_problem(zero_model, probe_grid, 0.0, statistical_settings)
    assert zero_problem.regularization == 0 and np.all(zero_problem.normal(vector) == 0)
    assert np.all(zero_problem.rhs == 0)
    for invalid in (-1, np.nan, np.inf, 1+0j, True, [1]):
        try:
            build_mmse_problem(zero_model, probe_grid, invalid, statistical_settings)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid noise variance was accepted.")

    measurement = build_silent_noise_measurement(public)
    true_variance = 1/(4*10**(20/10))  # Generation/scoring only.
    calibration = (calibration_rng.standard_normal(measurement.interval_sample_count)
                   + 1j*calibration_rng.standard_normal(measurement.interval_sample_count))*np.sqrt(true_variance/2)
    measured_variance = measurement.estimate(calibration)
    noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))*np.sqrt(true_variance/2)
    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., channel_rng.uniform(0, 2*np.pi, 2)])
    bem = build_pilot_ls_model(public)
    estimators = {"AFDM": build_afdm(public), "OTFS": build_otfs(public), "DD-a-OFDM": build_dd_a(public)}
    integration = []
    print("Data-prior/scalar/zero-noise controls passed; checking eight real estimator integrations.", flush=True)
    for label, magnitude in (("A", 0.0), ("B-4", 46875.0)):
        for scheme in SCHEMES:
            received = apply_paths(signals[scheme], delays, gains, [0., magnitude, -magnitude], fs)+noise
            original = received.copy()
            if scheme == "OFDM":
                estimated_taps = bem.reconstruct_taps(bem.estimate(received))
            else:
                estimate = estimators[scheme].estimate(received)
                estimated_taps = estimators[scheme].model.reconstruct_taps(estimate)
            model = build_pilot_data_model(build_channel_operator(scheme, public, estimated_taps), pilot_settings)
            observed = receive(scheme, received, public)
            problem = build_mmse_problem(model, observed, measured_variance, statistical_settings)
            assert problem.regularization == measured_variance
            assert np.array_equal(received, original)
            true_taps = path_taps(count, fs, delays, [0., magnitude, -magnitude], gains)
            true_channel = build_channel_operator(scheme, public, true_taps)
            # Truth below is used only to score what the receiver leaves behind.
            pilot_error = true_channel.apply(model.pilot_grid)-model.pilot_response
            expected = (true_channel.apply(model.embed_data(data))-model.apply(data)
                        + pilot_error+receive(scheme, noise, public))
            error = relative_error(problem.observations-model.apply(data), expected)
            assert error < 1e-11
            integration.append({"condition": label, "scheme": scheme,
                "diagnostic_Eb_N0_db_generation_only": 20,
                "measured_noise_variance_used": measured_variance,
                "regularization_used": problem.regularization,
                "remaining_channel_pilot_and_noise_identity_relative_error": error,
                "remaining_pilot_error_L2_norm": float(np.linalg.norm(pilot_error)),
                **check_equations(problem, vector, direction)})
    report = {"scope": "Plug-in MMSE statistical model and normal equations; no full data solve or BER.",
        "receiver_inputs": "Each receiver's pilot-estimated data model, original native receive grid, separate silent-interval measured variance, and public Es/statistical policy only. True channel/data/variance and target Eb/N0 remain generation/scoring inputs only.",
        "data_prior": {"constellation_point_count": 16, "mean_magnitude": float(abs(alphabet.mean())),
            "average_energy": prior_energy, "two_independent_symbols_covariance_identity_max_error": float(np.max(abs(prior_covariance-np.eye(2))))},
        "equations": {"normal": "A_hat^H*A_hat + sigma_hat^2/Es I", "rhs": "A_hat^H*(r-H_hat*p)"},
        "scalar_controls": scalar_controls, "actual_estimated_channel_integrations": integration,
        "zero_noise_singular_control": "Zero estimated channel and zero measured noise produce a zero normal operator with no hidden regularization floor. No inverse or solver is called.",
        "limits": ["Only normal equations/objectives are prepared, not symbol estimates, decisions or BER.",
            "No actual channel error is erased or supplied. Nominal noise covariance models AWGN only; actual model residual can be correlated and non-Gaussian.",
            "The prior is finite-alphabet independent 16-QAM with Es=1, not a Gaussian data assumption.",
            "Local TF/DAFT blocks and implicit full-frame DD scope are retained; no giant dense DD matrix is allocated.",
            "Full-frame generator, numerical solver and zero/small-variance strategy, any output gain handling, QAM decisions and BER remain pending."],
        "reference_url": "https://arxiv.org/pdf/2508.04253v2",
        "reference_scope": "Equations (43)-(44) motivate estimated-channel linear MMSE. Data-only columns, estimated-pilot subtraction and independently measured variance follow our agreed setup; the source's parameters/resources are not copied."}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Eight scalar controls, zero/noise-input controls and eight estimated-channel/measured-noise integrations passed.")


if __name__ == "__main__":
    main()
