"""Compare BEM-LS and a sparse three-path estimator on paired OFDM frames."""

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ber_experiment import POINTS, SCHEMES, apply_paths, receive, transmit
from ber_statistics import summarize_frames
from channel_operators import build_channel_operator
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from ofdm_bem import build_pilot_ls_model
from ofdm_path_estimation import build_estimator
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

REFERENCE = Path("data/ber_comparison_2026_11_04/multipath")
CONDITIONS = ("A", "B-4")
METHODS = ("BEM-LS", "3-path")
EB_N0_DB = 15.
_CONTEXT = None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    bem = build_pilot_ls_model(public)
    sparse = build_estimator(public, bem)
    assert bem.basis.shape[1] == 31 and sparse.model.path_count == 3
    assert bem.pilot_mask.sum() == public["pilot_comparison_policy"]["total_pilot_energy"] == 4096
    _CONTEXT = (public, manifest, bem, sparse, build_silent_noise_measurement(public))


def check_pilot_templates(bem, sparse):
    """Check the pilot prediction against the existing serial channel."""
    probes = [(delay, freq) for delay in sparse.model.delays for freq in (0., 12731., -32491., 46875., -46875.)]
    errors = []
    for delay, frequency in probes:
        serial = apply_paths(bem.known_pilot_samples, [delay], [1.], [frequency], bem.sample_rate_hz)
        observed = bem.pilot_observations(serial)
        predicted = sparse.model.atoms([(delay, frequency)])[:, 0]
        errors.append(float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed)))
    assert max(errors) < 1e-11
    return {"probes": len(probes), "max_relative_error": max(errors)}


def evaluate_receiver(taps, rx_grid, measured_variance, bits, truth, bem, public):
    channel = build_channel_operator("OFDM", public, taps)
    model = build_pilot_data_model(channel, public)
    problem = build_mmse_problem(model, rx_grid, measured_variance, public)
    solved = solve_mmse_problem(problem, public, include_effective_gain=True)
    solved.require_converged()
    decoded = recover_data_bits(problem, solved)
    # Integer CP-covered delays are distinct shifts within a useful block.
    # Whole-useful-frame tap Frobenius NMSE equals full-operator NMSE here.
    difference = taps[bem.useful_indices]-truth
    return {"channel_nmse": float(np.linalg.norm(difference)**2/np.linalg.norm(truth)**2),
            "bit_errors": int(np.count_nonzero(decoded[:bits.size] != bits)),
            "normal_residual": solved.relative_normal_residual}


def compute_frame(reference):
    public, manifest, bem, sparse, measurement = _CONTEXT
    index, measured_bits = reference["frame_index"], reference["measured_bits"]
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    cp, fs = bem.cp_length, bem.sample_rate_hz
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    signal = transmit("OFDM", build_grid("OFDM", modulate(bits), scenario=public), public, cp)
    count = signal.size
    unit_noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    delays = manifest["path_delays_samples"]
    powers = 10**(np.array(manifest["path_relative_powers_db"])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    calibration_count = measurement.interval_sample_count
    unit_calibration = (calibration_rng.standard_normal(calibration_count)+1j*calibration_rng.standard_normal(calibration_count))/np.sqrt(2)
    variance = 1/(4*10**(EB_N0_DB/10))
    measured_variance = measurement.estimate(np.sqrt(variance)*unit_calibration)
    useful_time = (np.arange(count)/fs)[bem.useful_indices]
    points = {}
    for condition in CONDITIONS:
        dopplers = manifest["path_dopplers_hz"][condition]
        clean = apply_paths(signal, delays, gains, dopplers, fs)
        rx = clean+np.sqrt(variance)*unit_noise
        observations = bem.pilot_observations(rx)
        estimate = sparse.fit_observations(observations)
        fits = (bem.reconstruct_taps(bem.fit_observations(observations)), sparse.model.reconstruct_taps(estimate))
        rx_grid = receive("OFDM", rx, public, cp)

        # The generator truth and data/pilot decomposition below are evaluator
        # inputs. Both estimators above consume the same raw pilot observations.
        truth = np.zeros((useful_time.size, len(bem.delays)), dtype=complex)
        for delay, gain, frequency in zip(delays, gains, dopplers):
            truth[:, bem.delays.index(delay)] += gain*np.exp(2j*np.pi*frequency*useful_time)
        pilot_clean = apply_paths(bem.known_pilot_samples, delays, gains, dopplers, fs)
        pilot_obs = bem.pilot_observations(pilot_clean)
        data_obs = bem.pilot_observations(clean-pilot_clean)
        data_to_pilot = float(np.linalg.norm(data_obs)**2/np.linalg.norm(pilot_obs)**2)
        assert data_to_pilot < 1e-20 if condition == "A" else data_to_pilot > 1e-12
        methods = {method: evaluate_receiver(taps, rx_grid, measured_variance, bits[:measured_bits], truth, bem, public)
                   for method, taps in zip(METHODS, fits)}
        expected = reference["errors"][manifest["reference_condition_indices"][condition]][POINTS.index(EB_N0_DB)][SCHEMES.index("OFDM")]
        if methods["BEM-LS"]["bit_errors"] != expected:
            raise RuntimeError(f"BEM-LS reference mismatch in frame {index}, {condition}.")
        points[condition] = {"methods": methods, "estimated_paths": estimate["paths"],
                             "pilot_fit_condition": estimate["condition"], "data_to_pilot_energy_ratio": data_to_pilot}
    record = {"frame_index": index, "measured_bits": measured_bits, "conditions": points,
              "measured_noise_variance": measured_variance, "reference_BEM_LS_reproduced": True}
    if index == 0:
        record["pilot_template_check"] = check_pilot_templates(bem, sparse)
    return record


def aggregate(frames, manifest):
    bits = np.array([frame["measured_bits"] for frame in frames])
    total = int(bits.sum())
    rows = []
    deltas = []
    for condition in CONDITIONS:
        points = [frame["conditions"][condition] for frame in frames]
        estimated_paths = np.array([p["estimated_paths"] for p in points])
        assert estimated_paths.shape == (len(frames), 3, 2) and np.isfinite(estimated_paths).all()
        assert np.all((estimated_paths[:, :, 0] >= 0.) & (estimated_paths[:, :, 0] <= 5.))
        assert np.max(abs(estimated_paths[:, :, 1])) <= 46875.
        leakage = float(np.mean([p["data_to_pilot_energy_ratio"] for p in points]))
        frame_errors = {}
        for method in METHODS:
            outcomes = [point["methods"][method] for point in points]
            frame_errors[method] = np.array([outcome["bit_errors"] for outcome in outcomes])
            nmse = np.array([outcome["channel_nmse"] for outcome in outcomes])
            assert np.isfinite(nmse).all() and np.min(nmse) >= 0.
            max_residual = max(outcome["normal_residual"] for outcome in outcomes)
            assert max_residual <= 1e-8
            summary = summarize_frames(frame_errors[method], bits, full_frame_capacity=manifest["frame_capacity_bits"])
            summary.update(condition=condition, method=method, Eb_N0_db=EB_N0_DB,
                mean_channel_nmse=float(nmse.mean()), mean_channel_nmse_db=float(10*np.log10(nmse.mean())),
                mean_data_to_pilot_observation_energy_ratio=leakage, max_normal_residual=max_residual)
            if method == "3-path":
                indices = np.argsort(estimated_paths[:, :, 0], axis=1)
                sorted_paths = np.take_along_axis(estimated_paths, indices[:, :, None], axis=1)
                correct = np.all(sorted_paths[:, :, 0] == manifest["path_delays_samples"], axis=1)
                summary["frames_with_correct_delay_set"] = int(correct.sum())
                if correct.any():
                    df = sorted_paths[correct, :, 1]-manifest["path_dopplers_hz"][condition]
                    summary["doppler_rmse_hz_when_delay_set_correct"] = float(np.sqrt(np.mean(df**2)))
            rows.append(summary)
        differences = frame_errors["3-path"]-frame_errors["BEM-LS"]
        delta = float(differences.sum()/total)
        paired_ci = None
        if len(bits) > 1:
            se = np.sqrt(len(bits)/(len(bits)-1)*np.sum((differences-delta*bits)**2))/total
            half_width = float(NormalDist().inv_cdf(.975)*se)
            paired_ci = [delta-half_width, delta+half_width]
        deltas.append({"condition": condition, "paired_BER_difference_3_path_minus_BEM_LS": delta,
                       "paired_95pct_frame_cluster_interval": paired_ci})
    return {"complete": len(frames) == manifest["requested_frame_count"], "completed_frame_count": len(frames),
            "measured_bits_per_point": total, "BEM_LS_reference_reproduced_for_all_frames": all(f["reference_BEM_LS_reproduced"] for f in frames),
            "pilot_template_check": frames[0]["pilot_template_check"], "results": rows, "paired_BER_differences": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import fast_mmse  # Make the workspace's optional SciPy installation available.
    import scipy
    reference_manifest = read_json(REFERENCE/"manifest.json")
    reference_results = read_json(REFERENCE/"results.json")
    assert reference_results["complete"] and not reference_results["diagnostic_only"]
    base = read_json(REFERENCE/"scenario_used.json")
    paths = base["channel_model"]["multipath_followup"]["path_delay_power_doppler_profile"]
    sweep = base["channel_model"]["multipath_followup"]["doppler_effect_followup"]["sweep_magnitude_design"]
    magnitudes = dict(zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]))
    sources = (*CORE_FILES, "ofdm_path_estimation.py", Path(__file__).name)
    hashes = {name: sha256(Path(name)) for name in sources}
    assert all(hashes[name] == digest for name, digest in reference_manifest["code_sha256"].items())
    reference_frames = sorted(read_frames(REFERENCE/"frames.jsonl"), key=lambda f: f["frame_index"])
    counts = reference_manifest["measurement_counts_by_frame"]
    assert [f["frame_index"] for f in reference_frames] == list(range(214))
    assert [f["measured_bits"] for f in reference_frames] == counts and sum(counts) == 10485760
    manifest = {"experiment": "OFDM_BEM_LS_vs_known_three_path_estimator", "methods": METHODS,
        "conditions": CONDITIONS, "data_Eb_N0_db": EB_N0_DB, "CP_samples": 5, "pilot_cells": 4096,
        "pilot_energy_Es": 4096, "data_cells": 12288, "root_seed": reference_manifest["root_seed"],
        "frame_capacity_bits": reference_manifest["frame_capacity_bits"], "requested_frame_count": len(counts),
        "measurement_counts_by_frame": counts, "reference_run": REFERENCE.as_posix(),
        "reference_condition_indices": {c: sweep["condition_labels"].index(c) for c in CONDITIONS},
        "reference_frames_sha256": sha256(REFERENCE/"frames.jsonl"), "reference_scenario_sha256": sha256(REFERENCE/"scenario_used.json"),
        "code_sha256": hashes, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "path_delays_samples": [p["excess_delay_samples_at_60mhz"] for p in paths],
        "path_relative_powers_db": [p["relative_power_db_to_los"] for p in paths],
        "path_dopplers_hz": {c: [0., magnitudes[c], -magnitudes[c]] for c in CONDITIONS},
        "receiver_controls": "Same Tx, same received pilot TF cells, same measured-noise full ICI MMSE and Gray detector. Sparse estimator uses shared known path count P=3, all integer delays 0..5 and fixed Doppler bound +/-46875 Hz. Actual paths stay in generator/evaluator.",
        "path_search": "Same sequential approximate-ML search as the original OTFS receiver: quarter-frame-bin Doppler initialization, <=1 Hz scalar refinement and joint LS after every path addition. Each template includes the full known OFDM pilot grid contribution to the observed TF cells. Unknown-data ICI remains in the fitted pilot observations.",
        "nmse_definition": "Linear mean over frames of the Frobenius squared error of estimated channel taps on all useful receive samples divided by true tap energy. Equivalent to full-operator NMSE for the selected integer-delay CP-covered model."}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == json.loads(json.dumps(manifest)), "Resume requires identical sources and reference files."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [f["frame_index"] for f in frames] == list(range(len(frames)))
    assert [f["measured_bits"] for f in frames] == counts[:len(frames)]
    print(f"OFDM estimator comparison: {len(frames)}/{len(counts)} frames, {sum(counts):,} bits per point.", flush=True)
    if len(frames) < len(counts):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, reference_frames[len(frames):]):
                    stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":"))+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        atomic_json(output/"results.json", aggregate(frames, manifest))
                        print(f"OFDM estimator comparison: {len(frames)}/{len(counts)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["BEM_LS_reference_reproduced_for_all_frames"]
    for row in (r for r in result["results"] if r["method"] == "BEM-LS"):
        original = next(r for r in reference_results["results"] if r["condition"] == row["condition"]
                        and r["waveform"] == "OFDM" and r["Eb_N0_db"] == EB_N0_DB)
        for key in ("bit_errors", "confidence_interval"):
            assert row[key] == original[key], (row["condition"], key)
    atomic_json(output/"results.json", result)
    print(f"Complete: {output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
