"""Validate the OFDM BEM-LS prototype, including unknown-data ICI and AWGN.

This is a small channel-estimation diagnostic, not a BER experiment. Generator
truth is used only by controls and the error evaluator, never by the estimator.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from ofdm_bem import build_pilot_ls_model, modulate
from pilot_layouts import build_grid, build_masks, known_pilot_symbols


def apply_time_taps(signal, taps, delays):
    """Independent serial, time-varying convolution for diagnostic controls."""
    received = np.zeros_like(signal)
    for column, delay in enumerate(delays):
        received[delay:] += taps[delay:, column] * signal[:signal.size - delay]
    return received


def channel_nmse(coefficients, basis, truth):
    """Squared tap error / squared true tap norm; all delays are evaluated.

    Compute via the basis Gram matrix to avoid allocating a frame per noisy
    estimate. A direct reconstructed-tap norm is checked independently below.
    """
    count = basis.shape[1]
    values = coefficients.T.reshape((-1, truth.shape[1], count))
    gram = basis.conj().T @ basis
    cross = basis.conj().T @ truth
    energy = np.einsum("mdq,qr,mdr->m", values.conj(), gram, values).real
    inner = np.einsum("mdq,qd->m", values.conj(), cross).real
    reference = float(np.vdot(truth, truth).real)
    return np.maximum(energy - 2 * inner + reference, 0) / reference


def summary(values):
    values = np.asarray(values)
    mean = float(values.mean())
    return {"mean": mean, "mean_db": float(10 * np.log10(max(mean, 1e-300))),
            "minimum": float(values.min()), "maximum": float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=16,
                        help="Diagnostic frames only; does not change the BER bit quota.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--basis-count", type=int,
                        help="Diagnostic override restricted to the agreed validation alternatives; does not edit scenario.json.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    root = Path(__file__).resolve().parent
    settings = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    selected_design = settings["receiver_equalization_policy"]["channel_estimation_selection"]["OFDM"]["candidate_design"]
    selected_count = selected_design["basis_function_count"]
    alternatives = selected_design["basis_review"]["initial_basis_decision"]["validation_alternative_counts"]
    if args.basis_count is not None and args.basis_count not in [selected_count, *alternatives]:
        parser.error("--basis-count must be the selected initial count or an agreed validation alternative")
    # A separate public-settings projection makes generator access impossible
    # through the configuration object supplied to the estimator builder.
    public = copy.deepcopy({key: settings[key] for key in (
        "prefix_policy", "sampling_model", "receiver_information_policy",
        "receiver_equalization_policy", "pilot_comparison_policy")})
    if args.basis_count is not None:
        public["receiver_equalization_policy"]["channel_estimation_selection"]["OFDM"]["candidate_design"]["basis_function_count"] = args.basis_count
    model = build_pilot_ls_model(public)
    n, count = model.basis.shape
    print(f"Built fixed DPSS {count} pilot-LS model.", flush=True)
    delays = model.delays
    useful_indices = model.useful_indices
    useful_basis = model.basis[useful_indices]
    last_indices = useful_indices[-model.block_length:]
    last_basis = model.basis[last_indices]
    assert not model.pilot_mask[:, -1].any()
    expected_shape = (4096, len(delays) * count)
    assert model.observation_matrix.shape == expected_shape
    basis_review = json.loads((root / "data/bem_basis_review.json").read_text(encoding="utf-8"))
    previous = next(row for row in basis_review["results"]
                    if row["basis_family"] == "DPSS_Slepian" and row["basis_function_count"] == count)
    condition = float(model.singular_values[0] / model.singular_values[-1])
    np.testing.assert_allclose(condition, previous["known_pilot_matrix_condition_2norm"], rtol=1e-10)
    np.testing.assert_allclose(model.pilot_observations(model.known_pilot_samples),
                               known_pilot_symbols("OFDM", scenario=public), atol=1e-12, rtol=1e-12)

    seed = 20261104
    streams = np.random.SeedSequence(seed).spawn(4)
    data_rng, phase_rng, noise_rng, control_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams]
    # This algebraic control spans every allowed delay; it is not an additional
    # physical multipath experiment or a change to the selected P=3 profile.
    exact_coeff = ((control_rng.standard_normal(len(delays) * count)
                    + 1j * control_rng.standard_normal(len(delays) * count))
                   * np.sqrt(n / (2 * len(delays) * count)))
    exact_taps = model.basis @ exact_coeff.reshape((len(delays), count)).T
    control_rx = apply_time_taps(model.known_pilot_samples, exact_taps, delays)
    observed = model.pilot_observations(control_rx)
    matrix_error = float(np.linalg.norm(observed - model.observation_matrix @ exact_coeff)
                         / np.linalg.norm(observed))
    fitted = model.estimate(control_rx)
    coefficient_error = float(np.linalg.norm(fitted - exact_coeff) / np.linalg.norm(exact_coeff))
    tap_error = float(np.linalg.norm(model.reconstruct_taps(fitted) - exact_taps)
                      / np.linalg.norm(exact_taps))
    assert max(matrix_error, coefficient_error, tap_error) < 1e-10

    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    true_delays = np.array([p["excess_delay_samples_at_60mhz"] for p in paths])
    powers = 10 ** (np.array([p["relative_power_db_to_los"] for p in paths]) / 10)
    magnitudes = np.sqrt(powers / powers.sum())
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    labels = sweep["condition_labels"]
    levels = sweep["maximum_residual_doppler_levels_hz"]
    eb_n0 = settings["evaluation"]["eb_n0_exploratory_sweep_db"]
    masks = build_masks("OFDM", scenario=public)
    data_count = int(masks["data"].sum())
    times = np.arange(n) / model.sample_rate_hz
    orthogonal = np.linalg.qr(useful_basis, mode="reduced")[0]
    bins = [{"projection": [], "pilot_only": [], "with_data": [],
             "pilot_only_last": [], "with_data_last": [], "data_ici_ratio": [],
             "data_ici_power": [], "residual": [],
             "awgn": [[] for _ in eb_n0], "awgn_last": [[] for _ in eb_n0]}
            for _ in levels]
    max_static_data_ici = 0.0
    max_observation_decomposition_error = 0.0
    noise_gains = []
    for trial in range(args.trials):
        data = (data_rng.choice([-3, -1, 1, 3], data_count)
                + 1j * data_rng.choice([-3, -1, 1, 3], data_count)) / np.sqrt(10)
        grid = build_grid("OFDM", data, scenario=public)
        full_samples = modulate(grid, model.cp_length)
        data_samples = full_samples - model.known_pilot_samples
        phases = np.r_[0.0, phase_rng.uniform(0, 2 * np.pi, 2)]
        gains = magnitudes * np.exp(1j * phases)
        noise = (noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n)) / np.sqrt(2)
        noise_coeff = model.fit_observations(model.pilot_observations(noise))
        noise_taps = model.reconstruct_taps(noise_coeff, sample_indices=useful_indices)
        noise_gains.append(float(np.vdot(noise_taps, noise_taps).real / useful_indices.size))
        for index, magnitude in enumerate(levels):
            frequencies = np.array([0.0, magnitude, -magnitude])
            path_taps = gains[None, :] * np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
            full_rx = apply_time_taps(full_samples, path_taps, true_delays)
            pilot_rx = apply_time_taps(model.known_pilot_samples, path_taps, true_delays)
            data_rx = apply_time_taps(data_samples, path_taps, true_delays)
            pilot_y = model.pilot_observations(pilot_rx)
            full_y = model.pilot_observations(full_rx)
            data_y = model.pilot_observations(data_rx)
            decomposition_error = float(np.linalg.norm(full_y - pilot_y - data_y) / np.linalg.norm(full_y))
            max_observation_decomposition_error = max(max_observation_decomposition_error, decomposition_error)
            if magnitude == 0:
                max_static_data_ici = max(max_static_data_ici, float(np.linalg.norm(data_y) / np.linalg.norm(pilot_y)))
            observations = [pilot_y, full_y]
            for db in eb_n0:
                variance = 1 / (4 * 10 ** (db / 10))
                observations.append(model.pilot_observations(full_rx + np.sqrt(variance) * noise))
            y = np.column_stack(observations)
            coefficients = model.fit_observations(y)
            truth = np.zeros((n, len(delays)), dtype=complex)
            truth[:, true_delays] = path_taps
            errors = channel_nmse(coefficients, useful_basis, truth[useful_indices])
            last_errors = channel_nmse(coefficients, last_basis, truth[last_indices])
            if trial == 0 and index == len(levels) - 1:
                direct = model.reconstruct_taps(coefficients[:, 1], sample_indices=useful_indices)
                direct_error = float(np.linalg.norm(direct - truth[useful_indices])**2
                                     / np.linalg.norm(truth[useful_indices])**2)
                np.testing.assert_allclose(errors[1], direct_error, rtol=1e-9, atol=1e-13)
            entry = bins[index]
            target = truth[useful_indices]
            projection_error = max(float(np.vdot(target, target).real
                                         - np.linalg.norm(orthogonal.conj().T @ target)**2), 0)
            entry["projection"].append(projection_error / float(np.vdot(target, target).real))
            entry["pilot_only"].append(float(errors[0]))
            entry["with_data"].append(float(errors[1]))
            entry["pilot_only_last"].append(float(last_errors[0]))
            entry["with_data_last"].append(float(last_errors[1]))
            entry["data_ici_ratio"].append(float(np.linalg.norm(data_y)**2 / np.linalg.norm(pilot_y)**2))
            entry["data_ici_power"].append(float(np.vdot(data_y, data_y).real / data_y.size))
            entry["residual"].append(float(np.linalg.norm(full_y - model.observation_matrix @ coefficients[:, 1])**2
                                           / np.linalg.norm(full_y)**2))
            for j in range(len(eb_n0)):
                entry["awgn"][j].append(float(errors[j + 2]))
                entry["awgn_last"][j].append(float(last_errors[j + 2]))
        if (trial + 1) % 4 == 0 or trial + 1 == args.trials:
            print(f"Completed {trial + 1}/{args.trials} diagnostic frames per condition.", flush=True)
    assert max_static_data_ici < 1e-12
    assert max_observation_decomposition_error < 1e-12
    results = []
    for label, magnitude, entry in zip(labels, levels, bins):
        result = {"condition": label, "maximum_residual_doppler_hz": magnitude,
                  "best_basis_projection_NMSE_truth_only_control": summary(entry["projection"]),
                  "pilot_only_noiseless_channel_NMSE_control": summary(entry["pilot_only"]),
                  "data_and_pilots_noiseless_channel_NMSE": summary(entry["with_data"]),
                  "pilot_only_noiseless_last_block_NMSE_control": summary(entry["pilot_only_last"]),
                  "data_and_pilots_noiseless_last_block_NMSE": summary(entry["with_data_last"]),
                  "unknown_data_ICI_to_pilot_component_power_ratio": summary(entry["data_ici_ratio"]),
                  "mean_unknown_data_ICI_power_per_pilot_observation": float(np.mean(entry["data_ici_power"])),
                  "noiseless_pilot_observation_residual_power_ratio": summary(entry["residual"]),
                  "AWGN_with_data_and_pilots": [
                      {"received_Eb_N0_dB": db, "channel_NMSE": summary(values),
                       "last_block_NMSE": summary(last_values)}
                      for db, values, last_values in zip(eb_n0, entry["awgn"], entry["awgn_last"])]}
        results.append(result)
    report = {
        "purpose": "OFDM ordinary BEM-LS feasibility diagnostic; no equalization or BER.",
        "selected_initial_basis_count": selected_count,
        "diagnostic_basis_count_override": args.basis_count,
        "numpy_version": np.__version__, "diagnostic_seed": seed,
        "rng": "SeedSequence(seed).spawn(4), PCG64 streams for data/phases/noise/algebraic control",
        "diagnostic_frames_per_condition": args.trials,
        "pairing": "Reuse each frame's data, initial path phases and unit-variance noise across A/B and scaled Eb/N0 diagnostics; independent frames redraw them. Fixed pilots use their separate agreed seed.",
        "receiver_input_scope": "Public design settings and received pilot observations only; no channel_model, true transmitted data or true active path parameters are passed to the estimator.",
        "physical_profile_scope": "Agreed three-path residual A/B channels after perfect common LOS Doppler correction, continuous CP-inclusive clock. The orbit snapshots are not distinct residual channels under this ideal correction.",
        "delay_candidates_samples": list(delays), "basis_function_count": count,
        "fixed_design_maximum_residual_doppler_hz": model.maximum_residual_doppler_hz,
        "observation_matrix_shape": list(model.observation_matrix.shape),
        "observation_matrix_rank": int(np.count_nonzero(model.singular_values > model.singular_values[0] * model.rank_relative_tolerance)),
        "observation_matrix_condition_2norm": condition,
        "rank_relative_tolerance": model.rank_relative_tolerance,
        "solver": "Thin SVD pseudoinverse; no regularization, true-data cancellation, path pruning or iterative refinement.",
        "basis_numerical_check": model.basis_check,
        "algebraic_model_controls": {"pilot_matrix_vs_serial_channel_relative_error": matrix_error,
                                      "noiseless_coefficient_recovery_relative_error": coefficient_error,
                                      "noiseless_tap_recovery_relative_error": tap_error},
        "maximum_static_data_ICI_relative_amplitude": max_static_data_ici,
        "maximum_pilot_plus_data_observation_decomposition_error": max_observation_decomposition_error,
        "noise_only_unit_observation_variance_channel_MSE_gain": summary(noise_gains),
        "previous_ideal_white_noise_gain_reference": previous["ideal_white_noise_channel_MSE_gain"],
        "NMSE_definition": "Sum of squared tap errors over delays 0..5 and evaluated receive samples, divided by true squared tap norm. Frame metric uses useful samples from all 64 blocks; last-block metric uses pilot-free block 63. Linear NMSEs are averaged before conversion to dB.",
        "limits": [
            "Small feasibility diagnostic, not a BER run or a selected Monte Carlo stopping policy. The agreed information-bit quota and BER confidence-interval choices are unchanged.",
            "Noiseless pilot-only and true best-basis projection are diagnostic controls. The deployable estimate never receives true coefficients, delays, Dopplers, gains or data.",
            "Unknown-data ICI is an unmodeled residual in ordinary LS, without an independence, Gaussian or white-noise assumption. Colored-interference treatment, regularization and decision feedback are not selected.",
            "All six candidate integer taps are fit without P=3 sparsity enforcement. The known path-count prior is available but not used by this dense LS prototype.",
            "The fixed maximum Doppler envelope is now recorded as common support in receiver_information_policy.residual_doppler_support. It is not a supplied actual path Doppler; other receiver implementations remain pending.",
            "No TF equalizer, QAM detector, receiver for the other three methods or four-way BER result is introduced.",
        ],
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for result in results:
        pilot = result["pilot_only_noiseless_channel_NMSE_control"]["mean_db"]
        full = result["data_and_pilots_noiseless_channel_NMSE"]["mean_db"]
        last = result["data_and_pilots_noiseless_last_block_NMSE"]["mean_db"]
        print(f"{result['condition']}: pilot-only {pilot:.2f} dB, with-data {full:.2f} dB, last-block {last:.2f} dB NMSE")
    print("Algebraic, static-channel and observation-decomposition checks passed; BER remains pending.")


if __name__ == "__main__":
    main()
