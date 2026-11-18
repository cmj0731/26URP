"""Paired OTFS reflection-delay-spacing experiment with the existing receiver."""

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from ber_experiment import receive, transmit
from ber_statistics import summarize_frames
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from otfs_fractional_channel import FractionalChannelOperator, exact_effective_gain, operator_nmse, serial_channel
from otfs_fractional_estimation import ContinuousEstimator
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import atomic_json

CONDITIONS = {"A": (0., 0., 0.), "B-4": (0., 46875., -46875.)}
DELAYS = ((0., 2., 5.), (0., 4., 5.), (0., 4.5, 5.), (0., 4.75, 5.))
REFERENCE = Path("data/otfs_fractional_delay_2026_11_18")
_CONTEXT = None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    paths = base["channel_model"]["multipath_followup"]["path_delay_power_doppler_profile"]
    assert [p["excess_delay_samples_at_60mhz"] for p in paths] == list(DELAYS[0])
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    _CONTEXT = (public, manifest, ContinuousEstimator(public), build_silent_noise_measurement(public),
                np.sqrt(powers/powers.sum()))


def evaluate_point(signal, bits, noise, measured_variance, delays, dopplers, gains):
    public, _, estimator, _, _ = _CONTEXT
    n, blocks, cp, fs = estimator.model.length, estimator.model.blocks, estimator.model.cp, estimator.model.fs
    clean = serial_channel(signal, delays, gains, dopplers, fs, n, cp)
    received = clean+noise
    estimate = estimator.estimate(received)
    estimated = FractionalChannelOperator(n, blocks, cp, fs, estimate["paths"], estimate["gains"])
    model = build_pilot_data_model(estimated, public)
    problem = build_mmse_problem(model, receive("OTFS", received, public, cp), measured_variance, public)
    solved = solve_mmse_problem(problem, public)
    solved.require_converged()
    solved.effective_gain = exact_effective_gain(problem)
    decoded = recover_data_bits(problem, solved)
    truth = FractionalChannelOperator(n, blocks, cp, fs, tuple(zip(delays, dopplers)), gains)
    return {"bit_errors": int(np.count_nonzero(decoded[:bits.size] != bits)),
            "operator_nmse": operator_nmse(estimated, truth), "estimated_paths": estimate["paths"],
            "optimizer": estimate["optimizer"], "iterations": solved.iterations,
            "normal_residual": solved.relative_normal_residual}


def baseline_point(frame, condition_index):
    # Previous run axes: condition, delay offset (0 = [0,2,5]), receiver
    # (1 = continuous). Reuse the original raw estimates and bit outcomes.
    i = condition_index
    return {"bit_errors": frame["errors"][i][0][1], "operator_nmse": frame["operator_nmse"][i][0][1],
            "estimated_paths": frame["estimated_paths_delay_samples_frequency_hz"][i][0][1],
            "optimizer": frame["continuous_optimizer"][i][0], "iterations": frame["iterations"][i][0][1],
            "normal_residual": frame["relative_normal_residual"][i][0][1]}


def compute_frame(reference):
    public, manifest, estimator, measurement, magnitudes = _CONTEXT
    index, measured_bits = reference["frame_index"], reference["measured_bits"]
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    signal = transmit("OTFS", build_grid("OTFS", modulate(bits), scenario=public), public, estimator.model.cp)
    gains = magnitudes*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    unit_noise = (noise_rng.standard_normal(signal.size)+1j*noise_rng.standard_normal(signal.size))/np.sqrt(2)
    count = measurement.interval_sample_count
    calibration = (calibration_rng.standard_normal(count)+1j*calibration_rng.standard_normal(count))/np.sqrt(2)
    variance = 1/(4*10**(manifest["data_Eb_N0_db"]/10))
    measured_variance = measurement.estimate(np.sqrt(variance)*calibration)
    assert measured_variance == reference["estimated_noise_variance"]
    noise = np.sqrt(variance)*unit_noise
    points = {}
    for ci, (condition, dopplers) in enumerate(CONDITIONS.items()):
        baseline = baseline_point(reference, ci)
        if index == 0:
            replay = evaluate_point(signal, bits[:measured_bits], noise, measured_variance, DELAYS[0], dopplers, gains)
            assert replay["bit_errors"] == baseline["bit_errors"]
            assert replay["operator_nmse"] == baseline["operator_nmse"]
            np.testing.assert_array_equal(replay["estimated_paths"], baseline["estimated_paths"])
        points[condition] = [baseline]+[evaluate_point(signal, bits[:measured_bits], noise, measured_variance,
            delays, dopplers, gains) for delays in DELAYS[1:]]
    return {"frame_index": index, "measured_bits": measured_bits, "conditions": points,
            "measured_noise_variance": measured_variance, "baseline_replayed": index == 0}


def aggregate(frames, manifest):
    frames = sorted(frames, key=lambda frame: frame["frame_index"])
    counts = [frame["measured_bits"] for frame in frames]
    rows = []
    for condition in CONDITIONS:
        for di, delays in enumerate(DELAYS):
            points = [frame["conditions"][condition][di] for frame in frames]
            estimates = np.array([p["estimated_paths"] for p in points])
            assert estimates.shape == (len(frames), 3, 2) and np.isfinite(estimates).all()
            assert np.all((estimates[:, :, 0] >= 0.) & (estimates[:, :, 0] <= 5.))
            assert np.max(abs(estimates[:, :, 1])) <= 46875.
            # Sorted pairing minimizes squared error over one-to-one delay
            # matches. The score describes the unlabeled delay set.
            ordered_delays = np.sort(estimates[:, :, 0], axis=1)
            reflection_errors = ordered_delays[:, 1:]-np.array(delays[1:])
            gap_errors = np.diff(ordered_delays[:, 1:], axis=1)[:, 0]-(delays[2]-delays[1])
            operator_errors = np.array([p["operator_nmse"] for p in points])
            assert np.isfinite(operator_errors).all() and np.min(operator_errors) >= 0.
            mean_nmse = float(operator_errors.mean())
            worst_residual = max(p["normal_residual"] for p in points)
            assert worst_residual <= 1e-8
            row = summarize_frames([p["bit_errors"] for p in points], counts,
                                   full_frame_capacity=manifest["frame_capacity_bits"])
            row.update(condition=condition, physical_delays_samples=delays,
                reflection_spacing_samples=delays[2]-delays[1],
                reflection_delay_rmse_samples=float(np.sqrt(np.mean(reflection_errors**2))),
                per_reflection_delay_rmse_samples=np.sqrt(np.mean(reflection_errors**2, axis=0)).tolist(),
                worst_reflection_delay_error_p95_samples=float(np.quantile(np.max(abs(reflection_errors), axis=1), .95)),
                reflection_spacing_rmse_samples=float(np.sqrt(np.mean(gap_errors**2))),
                los_delay_rmse_samples=float(np.sqrt(np.mean(ordered_delays[:, 0]**2))),
                mean_operator_nmse=mean_nmse, mean_operator_nmse_db=float(10*np.log10(max(mean_nmse, 1e-300))),
                max_normal_residual=worst_residual, max_mmse_iterations=max(p["iterations"] for p in points),
                optimizer_non_success_frames=sum(not p["optimizer"]["success"] for p in points),
                baseline_reused=di==0)
            rows.append(row)
    return {"complete": len(frames)==manifest["requested_frame_count"], "completed_frame_count": len(frames),
            "measured_bits_per_point": sum(counts), "reference_frame_replayed": any(f["baseline_replayed"] for f in frames),
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reference_manifest = read_json(REFERENCE/"manifest.json")
    reference_results = read_json(REFERENCE/"results.json")
    assert reference_results["complete"] and not reference_results["diagnostic_only"]
    assert reference_manifest["conditions"] == list(CONDITIONS) and reference_manifest["delta_samples"][0] == 0.
    assert reference_manifest["receivers"][1] == "continuous_delay"
    base = read_json(REFERENCE/"scenario_used.json")
    numeric_hashes = reference_manifest["code_sha256"]
    assert all(source_hash(Path(name)) == digest for name, digest in numeric_hashes.items())
    reference_frames = sorted(read_frames(REFERENCE/"frames.jsonl"), key=lambda frame: frame["frame_index"])
    counts = reference_manifest["measurement_counts_by_frame"]
    assert [f["frame_index"] for f in reference_frames] == list(range(len(counts)))
    assert [f["measured_bits"] for f in reference_frames] == counts
    assert len(counts) == 214 and sum(counts) == 10485760
    manifest = {"experiment": "OTFS_reflection_delay_spacing", "conditions_hz": CONDITIONS,
                "physical_delays_samples": DELAYS, "CP_samples": reference_manifest["CP_samples"],
                "pilot_energy_Es": reference_manifest["pilot_energy_Es"], "data_Eb_N0_db": reference_manifest["data_Eb_N0_db"],
                "root_seed": reference_manifest["root_seed"], "frame_capacity_bits": reference_manifest["frame_capacity_bits"],
                "requested_frame_count": len(counts), "measurement_counts_by_frame": counts,
                "original_source_sha256": numeric_hashes, "runner_sha256": source_hash(Path(__file__)),
                "reference_run": REFERENCE.as_posix(), "reference_frames_sha256": source_hash(REFERENCE/"frames.jsonl"),
                "reference_scenario_sha256": source_hash(REFERENCE/"scenario_used.json"),
                "reference_rule": "Reuse [0,2,5] continuous-delay per-frame records and independently replay frame 0 in both conditions. Pair all new variants to original frame-role RNG streams.",
                "delay_score": "Sort the three delay estimates and match by order to the three true delays. Pool squared errors of the two later delays over frames, then take square root. This scores set recovery without Doppler labels.",
                "channel_score": "Mean full useful-channel operator Frobenius NMSE in linear units, then convert to dB.",
                "controls": "Same known P=3 continuous-delay receiver, W64 pilot/data layout, path powers and initial phases, noise and calibration draws. Move only the -10 dB reflection toward the -15 dB reflection at 5 samples. Public bounds and receiver algorithm are identical to the reference."}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == json.loads(json.dumps(manifest)), "Resume requires the same input and code hashes."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [f["frame_index"] for f in frames] == list(range(len(frames)))
    assert [f["measured_bits"] for f in frames] == counts[:len(frames)]
    print(f"OTFS reflection spacing: {len(frames)}/{len(counts)} frames, {sum(counts):,} bits per point.", flush=True)
    if len(frames) < len(counts):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, reference_frames[len(frames):]):
                    stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":"))+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        atomic_json(output/"results.json", aggregate(frames, manifest))
                        print(f"OTFS reflection spacing: {len(frames)}/{len(counts)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["reference_frame_replayed"]
    for row in (r for r in result["results"] if r["baseline_reused"]):
        original = next(r for r in reference_results["results"] if r["condition"] == row["condition"]
                        and r["delta_samples"] == 0. and r["receiver"] == "continuous_delay")
        for key in ("bit_errors", "confidence_interval", "mean_operator_nmse_db"):
            assert row[key] == original[key], (row["condition"], key)
    atomic_json(output/"results.json", result)
    print(f"Complete: {output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
