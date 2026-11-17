"""Paired OTFS-only pilot-energy sensitivity with the original BER receivers."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
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
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid, build_masks
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

ENERGIES = (4096., 1024., 256., 64., 16., 4., 1.)
CONDITIONS = ("A", "B-4")
EB_N0_DB = 15.
_CONTEXT = None


def energy_settings(public, energy):
    """Make an OTFS-only intervention; all masks/data/noise settings are fixed."""
    settings = copy.deepcopy(public)
    policy = settings["pilot_comparison_policy"]
    policy["total_pilot_energy"] = energy
    layout = policy["otfs_dd_layout"]
    for key in ("total_pilot_energy_in_Es_units", "single_pilot_energy_in_Es_units"):
        layout[key] = energy
    layout["single_pilot_magnitude_at_Es_1"] = float(np.sqrt(energy))
    layout["known_pilot_values"] = {"real": float(np.sqrt(energy)), "imag": 0.}
    policy["known_pilot_sequence"]["OTFS"] = f"Single known real pilot sqrt({energy}) for this OTFS-only diagnostic."
    budget = policy["initial_budget"]["energy_budget"]
    budget["total_pilot_energy_in_Es_units"] = energy
    budget["expected_useful_frame_energy_in_Es_units"] = budget["expected_total_data_energy_in_Es_units"]+energy
    policy["sensitivity_scope"] = "OTFS-only intervention. Other waveforms' original layout metadata is historical and not used. No data/frame/noise renormalization."
    return settings


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    variants = [energy_settings(public, energy) for energy in manifest["pilot_energies_Es"]]
    reference_masks = build_masks("OTFS", scenario=public)
    for variant in variants:
        masks = build_masks("OTFS", scenario=variant)
        assert all(np.array_equal(reference_masks[key], masks[key]) for key in masks)
    profile = base["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    magnitudes = dict(zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]))
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    cp = public["prefix_policy"]["prefix_samples_per_block"]
    useful = (np.arange(blocks)[:, None]*(n+cp)+cp+np.arange(n)[None, :]).reshape(-1)
    _CONTEXT = {"base": base, "manifest": manifest, "public": public, "variants": variants,
        "estimators": [build_estimator(v) for v in variants], "masks": reference_masks,
        "measurement": build_silent_noise_measurement(public), "useful": useful,
        "delays": [p["excess_delay_samples_at_60mhz"] for p in paths],
        "path_magnitudes": np.sqrt(powers/powers.sum()),
        "dopplers": [magnitudes[label] for label in CONDITIONS]}


def compute_frame(job):
    index, measured_bits = job
    c = _CONTEXT
    manifest, public = c["manifest"], c["public"]
    rng = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = rng[0].integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    symbols = modulate(bits)
    gains = c["path_magnitudes"]*np.exp(1j*np.r_[0., rng[1].uniform(0, 2*np.pi, 2)])
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    cp = public["prefix_policy"]["prefix_samples_per_block"]
    count, fs = (n+cp)*blocks, public["sampling_model"]["sample_rate_hz"]
    noise = (rng[2].standard_normal(count)+1j*rng[2].standard_normal(count))/np.sqrt(2)
    calibration_count = c["measurement"].interval_sample_count
    calibration = (rng[3].standard_normal(calibration_count)+1j*rng[3].standard_normal(calibration_count))/np.sqrt(2)
    # This variance stays fixed as the pilot energy changes: Eb is data-only.
    variance = 1/(4*10**(EB_N0_DB/10))
    measured_variance = c["measurement"].estimate(np.sqrt(variance)*calibration)
    signals, useful_energies = [], []
    for energy, variant in zip(manifest["pilot_energies_Es"], c["variants"]):
        grid = build_grid("OTFS", symbols, scenario=variant)
        assert int(c["masks"]["pilot"].sum()) == 1
        assert np.array_equal(grid.reshape(-1, order="F")[c["masks"]["data"].reshape(-1, order="F")], symbols)
        pilot_energy = float(np.sum(abs(grid[c["masks"]["pilot"]])**2))
        assert np.isclose(pilot_energy, energy, rtol=1e-13, atol=1e-13)
        assert not np.any(grid[c["masks"]["guard"] | c["masks"]["reserved"]])
        signal = transmit("OTFS", grid, variant, cp)
        useful_energy = float(np.vdot(signal[c["useful"]], signal[c["useful"]]).real)
        assert np.isclose(useful_energy, float(np.vdot(symbols, symbols).real)+energy, rtol=1e-12)
        signals.append(signal)
        useful_energies.append(useful_energy)
    shape = (len(CONDITIONS), len(signals))
    errors, iterations = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    nmse, residuals = np.empty(shape), np.empty(shape)
    delay_matches = np.zeros(shape, dtype=bool)
    frequency_errors = np.full(shape, np.nan)
    time_axis = np.arange(count)/fs
    for ci, frequency in enumerate(c["dopplers"]):
        frequencies = [0., frequency, -frequency]
        truth = np.zeros((count, max(c["delays"])+1), dtype=complex)
        for delay, gain, fd in zip(c["delays"], gains, frequencies):
            truth[:, delay] += gain*np.exp(2j*np.pi*fd*time_axis)
        true_useful = truth[c["useful"]]
        denominator = float(np.linalg.norm(true_useful)**2)
        for ei, (signal, variant, estimator) in enumerate(zip(signals, c["variants"], c["estimators"])):
            received = apply_paths(signal, c["delays"], gains, frequencies, fs)+np.sqrt(variance)*noise
            estimate = estimator.estimate(received)
            taps = estimator.model.reconstruct_taps(estimate)
            model = build_pilot_data_model(build_channel_operator("OTFS", variant, taps), variant)
            problem = build_mmse_problem(model, receive("OTFS", received, variant, cp), measured_variance, variant)
            solved = solve_mmse_problem(problem, variant, include_effective_gain=True)
            solved.require_converged()
            decoded = recover_data_bits(problem, solved)
            assert decoded.size == bits.size
            errors[ci, ei] = np.count_nonzero(decoded[:measured_bits] != bits[:measured_bits])
            iterations[ci, ei] = solved.iterations
            residuals[ci, ei] = solved.relative_normal_residual
            # Truth is used only by this evaluator, never passed to the receiver.
            nmse[ci, ei] = float(np.linalg.norm(taps[c["useful"]]-true_useful)**2/denominator)
            delay_matches[ci, ei] = sorted(p[0] for p in estimate["paths"]) == sorted(c["delays"])
            if delay_matches[ci, ei]:
                fitted = dict(estimate["paths"])
                frequency_errors[ci, ei] = max(abs(fitted[d]-fd) for d, fd in zip(c["delays"], frequencies))
    expected = manifest["reference_errors_by_frame"][index]
    if errors[:, 0].tolist() != expected:
        raise RuntimeError(f"Original 4096-Es BER reproduction failed in frame {index}: {errors[:, 0].tolist()} != {expected}")
    return {"frame_index": index, "measured_bits": measured_bits, "errors": errors.tolist(),
        "channel_nmse": nmse.tolist(), "delay_matches": delay_matches.tolist(),
        "maximum_frequency_error_hz_when_delays_match": [[float(v) if np.isfinite(v) else None for v in row] for row in frequency_errors],
        "iterations": iterations.tolist(), "relative_normal_residual": residuals.tolist(),
        "estimated_noise_variance": measured_variance, "true_noise_variance_evaluator_only": variance,
        "useful_transmit_energy": useful_energies, "reference_errors_match": True}


def aggregate(records, manifest):
    ordered = sorted(records, key=lambda row: row["frame_index"])
    counts = np.array([row["measured_bits"] for row in ordered])
    errors = np.array([row["errors"] for row in ordered])
    nmse = np.array([row["channel_nmse"] for row in ordered])
    matches = np.array([row["delay_matches"] for row in ordered])
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for ei, energy in enumerate(manifest["pilot_energies_Es"]):
            row = summarize_frames(errors[:, ci, ei], counts, confidence=.95,
                                   full_frame_capacity=manifest["frame_capacity_bits"])
            mean_nmse = float(nmse[:, ci, ei].mean())
            row.update(condition=condition, waveform="OTFS", Eb_N0_db=EB_N0_DB,
                pilot_energy_Es=energy, pilot_amplitude=float(np.sqrt(energy)),
                mean_channel_nmse=mean_nmse, mean_channel_nmse_db=float(10*np.log10(max(mean_nmse, 1e-300))),
                frames_with_correct_delay_multiset=int(matches[:, ci, ei].sum()),
                paired_bit_error_difference_vs_4096_Es=int((errors[:, ci, ei]-errors[:, ci, 0]).sum()))
            rows.append(row)
    return {"complete": len(records) == manifest["requested_frame_count"],
        "diagnostic_only": manifest["diagnostic_only"], "waveform": "OTFS",
        "completed_frame_count": len(records), "requested_frame_count": manifest["requested_frame_count"],
        "measured_bits_per_point": int(counts.sum()), "confidence_level": .95,
        "original_reference_reproduced_all_completed_frames": all(r["reference_errors_match"] for r in ordered),
        "max_iterations": int(max(np.max(r["iterations"]) for r in ordered)),
        "max_relative_normal_residual": float(max(np.max(r["relative_normal_residual"]) for r in ordered)),
        "results": rows,
        "interpretation": "Within-OTFS controlled pilot-energy sensitivity, not an equal-total-transmit-energy waveform ranking. Pilot count, masks, data Es, noise and receiver algorithms are fixed; complete random frames and the original exact bit-count prefix are retained. NMSE uses the whole useful-frame true channel only in the evaluator. Mean NMSE is averaged in linear units before dB conversion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=Path("data/ber_comparison_2026_11_04/multipath"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.diagnostic_frames is not None and args.diagnostic_frames < 1):
        parser.error("Positive worker/frame counts are required.")
    import fast_mmse  # Loads the optional local dependency directory before SciPy.
    import scipy
    root = Path(__file__).parent
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in (*CORE_FILES, Path(__file__).name)}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path, snapshot_path = args.output/"manifest.json", args.output/"scenario_used.json"
    checkpoint = args.output/"frames.jsonl"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if manifest["code_sha256"] != hashes:
            raise RuntimeError("Resume requires the same numerical source.")
    else:
        if manifest_path.exists() or checkpoint.exists():
            raise RuntimeError("Output already contains a run; use --resume or a new output.")
        reference = json.loads((args.reference_run/"manifest.json").read_text(encoding="utf-8"))
        base = json.loads((args.reference_run/"scenario_used.json").read_text(encoding="utf-8"))
        baseline = json.loads((args.reference_run/"results.json").read_text(encoding="utf-8"))
        assert baseline["complete"] and not baseline["diagnostic_only"]
        assert all(hashes[name] == digest for name, digest in reference["code_sha256"].items())
        assert base["pilot_comparison_policy"]["total_pilot_energy"] == ENERGIES[0]
        records = sorted([json.loads(line) for line in (args.reference_run/"frames.jsonl").read_text(encoding="utf-8").splitlines()], key=lambda r: r["frame_index"])
        assert [r["frame_index"] for r in records] == list(range(reference["requested_frame_count"]))
        plan = reference["measurement_counts_by_frame"]
        if args.diagnostic_frames is not None:
            plan = plan[:args.diagnostic_frames]
        manifest = {"experiment": "OTFS_single_pilot_energy_sensitivity", "pilot_energies_Es": list(ENERGIES),
            "conditions": list(CONDITIONS), "Eb_N0_db": EB_N0_DB,
            "root_seed": reference["root_seed"], "frame_capacity_bits": reference["frame_capacity_bits"],
            "measurement_counts_by_frame": plan, "requested_frame_count": len(plan),
            "requested_bits_per_point": sum(plan), "diagnostic_only": args.diagnostic_frames is not None,
            "reference_errors_by_frame": [[r["errors"][ci][4][2] for ci in (0, 4)] for r in records],
            "reference_run": str(args.reference_run),
            "reference_checkpoint_sha256": hashlib.sha256((args.reference_run/"frames.jsonl").read_bytes()).hexdigest(),
            "code_sha256": hashes, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "worker_count": args.workers,
            "control": "Reuse original PCG64 frame-index/role streams, all original data/noise/calibration draws and the exact quota/tail. Only the one OTFS pilot amplitude sqrt(Ep) changes. Receiver templates and known-pilot subtraction use that same transmitted known value. Es=1 data and data-only Eb/N0=15 dB fix the noise; do not renormalize the whole frame. The 4096-Es errors must match the original run per frame in A and B-4.",
            "reference_result_row_scope": "NMSE is evaluated against generator truth after estimation. Truth and true noise variance never enter the estimator or equalizer."}
        atomic_json(snapshot_path, base)
        atomic_json(args.output/"variant_overrides.json", {"scope": "OTFS only", "varying": "pilot energy and matching known receiver pilot/template amplitude", "variants": [{"pilot_energy_Es": e, "known_pilot_real": float(np.sqrt(e)), "pilot_count": 1, "guard_reserved_count": 4095, "data_count": 12288, "data_Es": 1., "data_Eb_N0_db": EB_N0_DB, "noise_variance": 1/(4*10**(EB_N0_DB/10)), "expected_useful_energy": 12288+e} for e in ENERGIES]})
        atomic_json(manifest_path, manifest)
    records = ([json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
               if checkpoint.exists() else [])
    indices = [r["frame_index"] for r in records]
    assert len(set(indices)) == len(indices)
    plan = manifest["measurement_counts_by_frame"]
    assert all(0 <= r["frame_index"] < len(plan) and r["measured_bits"] == plan[r["frame_index"]] for r in records)
    remaining = [(i, bits) for i, bits in enumerate(plan) if i not in set(indices)]
    start = time.perf_counter()
    print(f"OTFS energy: {len(records)}/{len(plan)} complete; {sum(plan)} measured bits/point.", flush=True)
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=(base, manifest)) as pool:
            futures = {pool.submit(compute_frame, job): job[0] for job in remaining}
            with checkpoint.open("a", encoding="utf-8") as stream:
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception as error:
                        atomic_json(args.output/"failure.json", {"frame_index": futures[future], "error": repr(error)})
                        for other in futures:
                            other.cancel()
                        raise
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    records.append(row)
                    result = aggregate(records, manifest)
                    if len(records) % 4 == 0 or result["complete"]:
                        atomic_json(args.output/"results.json", result)
                    print(f"OTFS energy: {len(records)}/{len(plan)} frames; {result['measured_bits_per_point']} bits/point; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    result = aggregate(records, manifest)
    assert result["complete"] and result["measured_bits_per_point"] == manifest["requested_bits_per_point"]
    atomic_json(args.output/"results.json", result)
    print(f"Complete: {args.output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
