"""OTFS-only fractional-delay sensitivity, CP=5, paired original/aware receivers."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from ber_experiment import apply_paths, receive, transmit
from ber_statistics import summarize_frames
from channel_operators import build_channel_operator
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from otfs_estimation import build_estimator
from otfs_fractional_channel import FractionalChannelOperator, serial_channel, exact_effective_gain, operator_nmse
from otfs_fractional_estimation import ContinuousEstimator, DELAY_STEP, JOINT_OPTIONS
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

CONDITIONS = ("A", "B-4")
DELTAS = (0., .1, .25, .5)
RECEIVERS = ("original_integer_delay", "continuous_delay")
EB_N0_DB = 15.
_CONTEXT = None


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy", "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    cp, n, blocks = 5, 256, 64
    assert public["prefix_policy"]["prefix_samples_per_block"] == cp
    assert public["pilot_comparison_policy"]["total_pilot_energy"] == 4096
    original, continuous = build_estimator(public), ContinuousEstimator(public)
    assert np.array_equal(original.model.rows, continuous.model.rows)
    profile = base["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    assert [p["excess_delay_samples_at_60mhz"] for p in paths] == [0, 2, 5]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    magnitudes = dict(zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]))
    _CONTEXT = {"public": public, "manifest": manifest, "original": original, "continuous": continuous,
        "path_magnitudes": np.sqrt(powers/powers.sum()), "dopplers": [magnitudes[label] for label in CONDITIONS],
        "measurement": build_silent_noise_measurement(public),
        "pilot_signal": transmit("OTFS", build_grid("OTFS", np.zeros(12288, complex), scenario=public), public, cp)}


def compute_frame(job):
    index, measured_bits = job
    c, n, blocks, cp = _CONTEXT, 256, 64, 5
    public, manifest = c["public"], c["manifest"]
    fs, count = public["sampling_model"]["sample_rate_hz"], (n+cp)*blocks
    rng = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = rng[0].integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    signal = transmit("OTFS", build_grid("OTFS", modulate(bits), scenario=public), public, cp)
    gains = c["path_magnitudes"]*np.exp(1j*np.r_[0., rng[1].uniform(0, 2*np.pi, 2)])
    noise = (rng[2].standard_normal(count)+1j*rng[2].standard_normal(count))/np.sqrt(2)
    calibration_count = c["measurement"].interval_sample_count
    calibration = (rng[3].standard_normal(calibration_count)+1j*rng[3].standard_normal(calibration_count))/np.sqrt(2)
    variance = 1/(4*10**(EB_N0_DB/10))
    measured_variance = c["measurement"].estimate(np.sqrt(variance)*calibration)
    shape = (len(CONDITIONS), len(DELTAS), len(RECEIVERS))
    errors, iterations = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    nmse, residuals = np.empty(shape), np.empty(shape)
    leakage, capture = np.empty(shape[:2]), np.empty(shape[:2])
    optimizer = [[None for _ in DELTAS] for _ in CONDITIONS]
    fitted = [[[None for _ in RECEIVERS] for _ in DELTAS] for _ in CONDITIONS]
    maximum_integer_difference = 0.
    for ci, frequency in enumerate(c["dopplers"]):
        frequencies = [0., frequency, -frequency]
        for di, delta in enumerate(DELTAS):
            delays = [0., 2.+delta, 5.]
            truth = FractionalChannelOperator(n, blocks, cp, fs, tuple(zip(delays, frequencies)), gains)
            clean = serial_channel(signal, delays, gains, frequencies, fs, n, cp)
            if delta == 0:
                reference_clean = apply_paths(signal, [0, 2, 5], gains, frequencies, fs)
                difference = float(np.linalg.norm(reference_clean-clean)/np.linalg.norm(reference_clean))
                assert difference < 1e-11
                maximum_integer_difference = max(maximum_integer_difference, difference)
            pilot_clean = serial_channel(c["pilot_signal"], delays, gains, frequencies, fs, n, cp)
            # Evaluation-only decomposition. Neither separated data/pilot nor
            # true path parameters are supplied to either experiment receiver.
            data_observations = c["original"].model.observations(clean-pilot_clean)
            pilot_observations = c["original"].model.observations(pilot_clean)
            pilot_energy = float(np.linalg.norm(pilot_observations)**2)
            leakage[ci, di] = float(np.linalg.norm(data_observations)**2/pilot_energy)
            capture[ci, di] = pilot_energy/float(np.linalg.norm(receive("OTFS", pilot_clean, public, cp))**2)
            if delta == 0:
                assert leakage[ci, di] < 1e-20
            received = clean+np.sqrt(variance)*noise
            observations = c["original"].model.observations(received)
            for ri, estimator in enumerate((c["original"], c["continuous"])):
                estimate = estimator.fit_observations(observations)
                estimated_paths = tuple(estimate["paths"])
                estimated = FractionalChannelOperator(n, blocks, cp, fs, estimated_paths, estimate["gains"])
                if ri == 0:
                    channel = build_channel_operator("OTFS", public, estimator.model.reconstruct_taps(estimate))
                else:
                    channel = estimated
                    optimizer[ci][di] = estimate["optimizer"]
                model = build_pilot_data_model(channel, public)
                problem = build_mmse_problem(model, receive("OTFS", received, public, cp), measured_variance, public)
                solved = solve_mmse_problem(problem, public)
                solved.require_converged()
                if ri == 1:
                    solved.effective_gain = exact_effective_gain(problem)
                decoded = recover_data_bits(problem, solved)
                errors[ci, di, ri] = np.count_nonzero(decoded[:measured_bits] != bits[:measured_bits])
                iterations[ci, di, ri] = solved.iterations
                residuals[ci, di, ri] = solved.relative_normal_residual
                nmse[ci, di, ri] = operator_nmse(estimated, truth)
                fitted[ci][di][ri] = [[float(d), float(f)] for d, f in estimated_paths]
    expected = manifest["reference_errors_by_frame"][index]
    if errors[:, 0, 0].tolist() != expected:
        raise RuntimeError(f"Original integer-channel reproduction failed in frame {index}: {errors[:,0,0].tolist()} != {expected}")
    return {"frame_index": index, "measured_bits": measured_bits, "errors": errors.tolist(),
        "operator_nmse": nmse.tolist(), "iterations": iterations.tolist(), "relative_normal_residual": residuals.tolist(),
        "data_to_pilot_observation_energy_ratio": leakage.tolist(), "pilot_energy_capture_fraction": capture.tolist(),
        "continuous_optimizer": optimizer, "estimated_paths_delay_samples_frequency_hz": fitted,
        "estimated_noise_variance": measured_variance, "true_noise_variance_evaluator_only": variance,
        "integer_serial_relative_error": maximum_integer_difference, "reference_errors_match": True}


def aggregate(records, manifest):
    ordered = sorted(records, key=lambda r: r["frame_index"])
    counts = np.array([r["measured_bits"] for r in ordered])
    errors = np.array([r["errors"] for r in ordered])
    nmse = np.array([r["operator_nmse"] for r in ordered])
    leakage = np.array([r["data_to_pilot_observation_energy_ratio"] for r in ordered])
    capture = np.array([r["pilot_energy_capture_fraction"] for r in ordered])
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for di, delta in enumerate(DELTAS):
            for ri, receiver in enumerate(RECEIVERS):
                row = summarize_frames(errors[:, ci, di, ri], counts, confidence=.95, full_frame_capacity=manifest["frame_capacity_bits"])
                mean_nmse = float(nmse[:, ci, di, ri].mean())
                row.update(condition=condition, waveform="OTFS", receiver=receiver, delta_samples=delta,
                    physical_delays_samples=[0., 2.+delta, 5.], cp_samples=5, Eb_N0_db=EB_N0_DB, pilot_energy_Es=4096,
                    mean_operator_nmse=mean_nmse, mean_operator_nmse_db=float(10*np.log10(max(mean_nmse, 1e-300))),
                    mean_data_to_pilot_observation_energy_ratio=float(leakage[:, ci, di].mean()),
                    mean_pilot_energy_capture_fraction=float(capture[:, ci, di].mean()),
                    paired_bit_error_difference_vs_delta0=int((errors[:, ci, di, ri]-errors[:, ci, 0, ri]).sum()))
                rows.append(row)
    return {"complete": len(records)==manifest["requested_frame_count"], "diagnostic_only": manifest["diagnostic_only"],
        "completed_frame_count": len(records), "measured_bits_per_point": int(counts.sum()), "results": rows,
        "original_reference_reproduced_all_completed_frames": all(r["reference_errors_match"] for r in ordered),
        "max_iterations": int(max(np.max(r["iterations"]) for r in ordered)),
        "max_relative_normal_residual": float(max(np.max(r["relative_normal_residual"]) for r in ordered)),
        "continuous_optimizer_non_success_count": sum(not item["success"] for r in ordered for condition in r["continuous_optimizer"] for item in condition),
        "interpretation": "CP=5 unchanged. Exact CP-covered piecewise Fourier transmit-waveform delay, no added interpolation FIR or RF filtering. Original W64 mask/pilot/data/noise remain fixed. Original integer-only and continuous-delay estimators see the same six delay rows and 64 Doppler bins, P=3 and public support, never active path truth. Full useful-channel operator Frobenius NMSE, not six-tap-only NMSE. No truth-based cancellation, no whole-frame circularization. Not a realistic RF/pulse-shaping channel or an intrinsic waveform ranking."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=Path("data/ber_comparison_2026_11_04/multipath"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.diagnostic_frames is not None and args.diagnostic_frames < 1):
        parser.error("Positive worker/frame counts are required.")
    import scipy
    root = Path(__file__).parent
    sources = (*CORE_FILES, "otfs_fractional_channel.py", "otfs_fractional_estimation.py", Path(__file__).name)
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path, snapshot_path, checkpoint = (args.output/name for name in ("manifest.json", "scenario_used.json", "frames.jsonl"))
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if hashes != manifest["code_sha256"]:
            raise RuntimeError("Resume requires identical numerical source.")
    else:
        if manifest_path.exists() or checkpoint.exists():
            raise RuntimeError("Output already exists; use --resume or a new directory.")
        reference = json.loads((args.reference_run/"manifest.json").read_text(encoding="utf-8"))
        baseline = json.loads((args.reference_run/"results.json").read_text(encoding="utf-8"))
        assert baseline["complete"] and not baseline["diagnostic_only"]
        assert all(hashes[name]==digest for name, digest in reference["code_sha256"].items())
        base = json.loads((args.reference_run/"scenario_used.json").read_text(encoding="utf-8"))
        originals = sorted([json.loads(x) for x in (args.reference_run/"frames.jsonl").read_text(encoding="utf-8").splitlines()], key=lambda r: r["frame_index"])
        plan = reference["measurement_counts_by_frame"]
        if args.diagnostic_frames is not None:
            plan = plan[:args.diagnostic_frames]
        manifest = {"experiment": "OTFS_fractional_delay_CP5_sensitivity", "conditions": list(CONDITIONS),
            "delta_samples": list(DELTAS), "receivers": list(RECEIVERS), "CP_samples": 5,
            "data_Eb_N0_db": EB_N0_DB, "pilot_energy_Es": 4096,
            "root_seed": reference["root_seed"], "frame_capacity_bits": reference["frame_capacity_bits"],
            "measurement_counts_by_frame": plan, "requested_frame_count": len(plan), "requested_bits_per_point": sum(plan),
            "diagnostic_only": args.diagnostic_frames is not None,
            "reference_errors_by_frame": [[r["errors"][ci][4][2] for ci in (0, 4)] for r in originals],
            "reference_run": str(args.reference_run), "code_sha256": hashes, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "worker_count": args.workers,
            "channel_rule": "Delay only reflection 1 from 2 to 2+delta samples; retain LOS 0 and reflection 2 at 5, fixed path powers and continuous Dopplers. Source waveform: each CP+useful slot is a rectangular-window Fourier series with signed FFT bins [-128..127] in FFT order. Explicit source-slot evaluation at t-delay, zero outside finite frame. No interpolation FIR, added filter, global periodic frame or CP extension.",
            "receiver_rule": "Same original six receive rows 128..133, all 64 Doppler bins, P=3, physical delay bound 0..5, fixed Doppler bound +/-46875 Hz. Original receiver uses six integer delay hypotheses. Continuous receiver uses 0.25-sample initialization lattice, same quarter-frame-bin frequency initialization and <=1-Hz scalar refinement, followed by bounded joint variable-projection L-BFGS-B over all 3 delays/frequencies with analytic gradient and joint complex-LS gains. Actual delta/delays/frequencies/gains are never receiver inputs.",
            "continuous_delay_initial_step_samples": DELAY_STEP, "joint_optimizer_options": JOINT_OPTIONS,
            "controls": "Original CP=5, W64 resource mask, one 4096Es pilot, Es=1 data, all original PCG64 frame-role data/channel/noise/calibration draws, exact bit quota/tail, nominal MMSE and gain-aware Gray decisions. New dense fractional operator and exact nominal gain retain every delay coupling. Generator truth/decomposition only used in evaluator; integer-reference errors must reproduce frame by frame.",
            "nmse_definition": "Frobenius norm squared of estimated minus true full useful channel operator divided by true norm squared, summed over all 64 blocks. For integer delays this equals the previous full useful-tap NMSE; fractional delay is never rounded or truncated to 0..5 taps."}
        atomic_json(snapshot_path, base)
        atomic_json(manifest_path, manifest)
    records = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines()] if checkpoint.exists() else []
    indices = [r["frame_index"] for r in records]
    plan = manifest["measurement_counts_by_frame"]
    assert len(set(indices))==len(indices)
    assert all(0 <= r["frame_index"] < len(plan) and r["measured_bits"]==plan[r["frame_index"]] for r in records)
    remaining = [(i, bits) for i, bits in enumerate(plan) if i not in set(indices)]
    start = time.perf_counter()
    print(f"OTFS fractional CP5: {len(records)}/{len(plan)} frames, {sum(plan)} bits/point.", flush=True)
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            futures = {pool.submit(compute_frame, job): job[0] for job in remaining}
            with checkpoint.open("a", encoding="utf-8") as stream:
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as error:
                        atomic_json(args.output/"failure.json", {"frame_index": futures[future], "error": repr(error)})
                        for item in futures: item.cancel()
                        raise
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    records.append(record)
                    if len(records)%4==0 or len(records)==len(plan):
                        atomic_json(args.output/"results.json", aggregate(records, manifest))
                        print(f"OTFS fractional CP5: {len(records)}/{len(plan)} frames, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    result = aggregate(records, manifest)
    assert result["complete"] and result["measured_bits_per_point"]==manifest["requested_bits_per_point"]
    atomic_json(args.output/"results.json", result)
    print(f"Complete: {args.output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
