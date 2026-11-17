"""Paired OTFS local-zero-region sensitivity with a fixed original receiver."""

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
from otfs_guard_design import WIDTHS, build_designs, original_bit_order
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

CONDITIONS = ("A", "B-4")
EB_N0_DB = 15.
_CONTEXT = None


def public_settings(base):
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    return {key: base[key] for key in keys}


def initialize_worker(base, manifest):
    global _CONTEXT
    public = public_settings(base)
    assert public["pilot_comparison_policy"]["total_pilot_energy"] == 4096
    designs = build_designs(public)
    # Deliberately freeze the original pilot-only working model, templates,
    # observations and search. Construct it from the original isolated mask;
    # never claim that narrowed actual masks satisfy its isolation condition.
    # Their unknown-data interference is retained in the received observations.
    estimator = build_estimator(public)
    profile = base["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    magnitudes = dict(zip(sweep["condition_labels"], sweep["maximum_residual_doppler_levels_hz"]))
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    cp = public["prefix_policy"]["prefix_samples_per_block"]
    useful = (np.arange(blocks)[:, None]*(n+cp)+cp+np.arange(n)[None, :]).reshape(-1)
    prefix = (np.arange(blocks)[:, None]*(n+cp)+np.arange(cp)[None, :]).reshape(-1)
    pilot_signal = transmit("OTFS", build_grid("OTFS", np.zeros(12288, dtype=complex), scenario=public), public, cp)
    _CONTEXT = {"public": public, "manifest": manifest, "designs": designs,
        "estimator": estimator, "pilot_signal": pilot_signal,
        "measurement": build_silent_noise_measurement(public), "useful": useful, "prefix": prefix,
        "delays": [p["excess_delay_samples_at_60mhz"] for p in paths],
        "path_magnitudes": np.sqrt(powers/powers.sum()),
        "dopplers": [magnitudes[label] for label in CONDITIONS]}


def compute_frame(job):
    index, measured_bits = job
    c = _CONTEXT
    m, public = c["manifest"], c["public"]
    rng = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        m["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = rng[0].integers(0, 2, m["frame_capacity_bits"], dtype=np.uint8)
    symbols = modulate(bits)
    gains = c["path_magnitudes"]*np.exp(1j*np.r_[0., rng[1].uniform(0, 2*np.pi, 2)])
    n, blocks = public["radio"]["frequency_bin_count"], public["frame_design"]["time_symbol_count"]
    cp = public["prefix_policy"]["prefix_samples_per_block"]
    count, fs = (n+cp)*blocks, public["sampling_model"]["sample_rate_hz"]
    noise = (rng[2].standard_normal(count)+1j*rng[2].standard_normal(count))/np.sqrt(2)
    calibration_count = c["measurement"].interval_sample_count
    calibration = (rng[3].standard_normal(calibration_count)+1j*rng[3].standard_normal(calibration_count))/np.sqrt(2)
    variance = 1/(4*10**(EB_N0_DB/10))
    measured_variance = c["measurement"].estimate(np.sqrt(variance)*calibration)
    signals, energies, prefix_energies, peaks = [], [], [], []
    for design in c["designs"]:
        settings, masks, order = design["settings"], design["masks"], design["order"]
        grid = build_grid("OTFS", symbols[order], scenario=settings)
        assert np.sum(abs(grid[masks["pilot"]])**2) == 4096
        assert not np.any(grid[masks["guard"] | masks["reserved"]])
        assert np.array_equal(grid.reshape(-1, order="F")[masks["data"].reshape(-1, order="F")], symbols[order])
        signal = transmit("OTFS", grid, settings, cp)
        useful_energy = float(np.vdot(signal[c["useful"]], signal[c["useful"]]).real)
        assert np.isclose(useful_energy, float(np.vdot(symbols, symbols).real)+4096, rtol=1e-12)
        if signals:
            assert np.max(abs(signal[c["prefix"]]-signals[0][c["prefix"]])) < 1e-12
        signals.append(signal)
        energies.append(useful_energy)
        prefix_energies.append(float(np.vdot(signal[c["prefix"]], signal[c["prefix"]]).real))
        peaks.append(float(np.max(abs(signal)**2)))
    assert np.ptp(energies) < 1e-8 and np.ptp(prefix_energies) < 1e-8 and np.ptp(peaks) < 1e-9
    shape = (len(CONDITIONS), len(WIDTHS))
    errors, iterations = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    nmse, residuals, leakage, observation_changes, tap_changes = [np.empty(shape) for _ in range(5)]
    matches = np.zeros(shape, dtype=bool)
    max_template_error = 0.
    time_axis = np.arange(count)/fs
    estimator = c["estimator"]
    for ci, frequency in enumerate(c["dopplers"]):
        frequencies = [0., frequency, -frequency]
        truth = np.zeros((count, max(c["delays"])+1), dtype=complex)
        for delay, gain, fd in zip(c["delays"], gains, frequencies):
            truth[:, delay] += gain*np.exp(2j*np.pi*fd*time_axis)
        true_useful = truth[c["useful"]]
        denominator = float(np.linalg.norm(true_useful)**2)
        pilot_clean = apply_paths(c["pilot_signal"], c["delays"], gains, frequencies, fs)
        pilot_observations = estimator.model.observations(pilot_clean)
        pilot_energy = float(np.vdot(pilot_observations, pilot_observations).real)
        prediction = estimator.model.atoms(zip(c["delays"], frequencies))@gains
        template_error = float(np.linalg.norm(pilot_observations-prediction)/np.linalg.norm(pilot_observations))
        assert template_error < 1e-11
        max_template_error = max(max_template_error, template_error)
        original_taps = original_observations = None
        for wi, (signal, design) in enumerate(zip(signals, c["designs"])):
            settings = design["settings"]
            clean = apply_paths(signal, c["delays"], gains, frequencies, fs)
            # Data/pilot separation is evaluator-only, never receiver cancellation.
            data_clean = apply_paths(signal-c["pilot_signal"], c["delays"], gains, frequencies, fs)
            assert np.linalg.norm(clean-pilot_clean-data_clean)/max(np.linalg.norm(clean), 1e-300) < 1e-12
            data_observations = estimator.model.observations(data_clean)
            leakage[ci, wi] = float(np.vdot(data_observations, data_observations).real/pilot_energy)
            isolated = design["descriptor"]["pilot_observation_data_isolated_by_public_support"]
            assert leakage[ci, wi] < 1e-20 if isolated else leakage[ci, wi] > 1e-12
            received = clean+np.sqrt(variance)*noise
            observations = estimator.model.observations(received)
            estimate = estimator.fit_observations(observations)
            taps = estimator.model.reconstruct_taps(estimate)
            if wi == 0:
                original_taps, original_observations = taps.copy(), observations.copy()
            observation_changes[ci, wi] = float(np.linalg.norm(observations-original_observations)/np.linalg.norm(original_observations))
            tap_changes[ci, wi] = float(np.linalg.norm(taps-original_taps)/np.linalg.norm(original_taps))
            if isolated:
                assert observation_changes[ci, wi] < 1e-12 and tap_changes[ci, wi] < 1e-7
            model = build_pilot_data_model(build_channel_operator("OTFS", settings, taps), settings)
            problem = build_mmse_problem(model, receive("OTFS", received, settings, cp), measured_variance, settings)
            solved = solve_mmse_problem(problem, settings, include_effective_gain=True)
            solved.require_converged()
            decoded = original_bit_order(recover_data_bits(problem, solved), design["order"])
            errors[ci, wi] = np.count_nonzero(decoded[:measured_bits] != bits[:measured_bits])
            iterations[ci, wi] = solved.iterations
            residuals[ci, wi] = solved.relative_normal_residual
            nmse[ci, wi] = float(np.linalg.norm(taps[c["useful"]]-true_useful)**2/denominator)
            matches[ci, wi] = sorted(p[0] for p in estimate["paths"]) == sorted(c["delays"])
    expected = m["reference_errors_by_frame"][index]
    if errors[:, 0].tolist() != expected:
        raise RuntimeError(f"Original W64 reference failed in frame {index}: {errors[:, 0].tolist()} != {expected}")
    return {"frame_index": index, "measured_bits": measured_bits, "errors": errors.tolist(),
        "channel_nmse": nmse.tolist(), "data_to_pilot_observation_energy_ratio": leakage.tolist(),
        "pilot_observation_relative_change_vs_W64": observation_changes.tolist(),
        "estimated_tap_relative_change_vs_W64": tap_changes.tolist(), "delay_matches": matches.tolist(),
        "iterations": iterations.tolist(), "relative_normal_residual": residuals.tolist(),
        "estimated_noise_variance": measured_variance, "true_noise_variance_evaluator_only": variance,
        "useful_transmit_energy": energies, "prefix_energy": prefix_energies,
        "sample_grid_peak_power": peaks, "max_known_pilot_template_relative_error": max_template_error,
        "reference_errors_match": True}


def aggregate(records, manifest):
    ordered = sorted(records, key=lambda r: r["frame_index"])
    counts = np.array([r["measured_bits"] for r in ordered])
    errors = np.array([r["errors"] for r in ordered])
    nmse = np.array([r["channel_nmse"] for r in ordered])
    leakage = np.array([r["data_to_pilot_observation_energy_ratio"] for r in ordered])
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for wi, width in enumerate(WIDTHS):
            row = summarize_frames(errors[:, ci, wi], counts, confidence=.95, full_frame_capacity=manifest["frame_capacity_bits"])
            mean_nmse, mean_leakage = float(nmse[:, ci, wi].mean()), float(leakage[:, ci, wi].mean())
            row.update(condition=condition, waveform="OTFS", Eb_N0_db=EB_N0_DB, pilot_energy_Es=4096,
                local_zero_delay_row_count=width, mean_channel_nmse=mean_nmse,
                mean_channel_nmse_db=float(10*np.log10(max(mean_nmse, 1e-300))),
                mean_data_to_pilot_observation_energy_ratio=mean_leakage,
                mean_data_to_pilot_observation_energy_ratio_db=float(10*np.log10(max(mean_leakage, 1e-300))),
                frames_with_correct_delay_multiset=int(sum(r["delay_matches"][ci][wi] for r in ordered)),
                paired_bit_error_difference_vs_W64=int((errors[:, ci, wi]-errors[:, ci, 0]).sum()))
            rows.append(row)
    return {"complete": len(records) == manifest["requested_frame_count"], "diagnostic_only": manifest["diagnostic_only"],
        "waveform": "OTFS", "completed_frame_count": len(records), "requested_frame_count": manifest["requested_frame_count"],
        "measured_bits_per_point": int(counts.sum()), "confidence_level": .95,
        "original_reference_reproduced_all_completed_frames": all(r["reference_errors_match"] for r in ordered),
        "max_iterations": int(max(np.max(r["iterations"]) for r in ordered)),
        "max_relative_normal_residual": float(max(np.max(r["relative_normal_residual"]) for r in ordered)),
        "results": rows,
        "interpretation": "Within-OTFS mask redistribution with the fixed original pilot-only estimator. Total zero-cell/data counts, pilot/data/useful/CP energy, sampled peak, noise and source bits are retained. Local guard removal necessarily relocates data and can affect detection independently of estimation; leakage and NMSE must be read with BER. Narrow masks deliberately admit unknown-data interference, which is not cancelled using truth. This is not an optimized reduced-guard receiver or a physical edge-guard-band experiment."}


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
    import fast_mmse
    import scipy
    root = Path(__file__).parent
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in (*CORE_FILES, "otfs_guard_design.py", Path(__file__).name)}
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
            raise RuntimeError("Output already contains a run; use --resume or another output.")
        reference = json.loads((args.reference_run/"manifest.json").read_text(encoding="utf-8"))
        reference_result = json.loads((args.reference_run/"results.json").read_text(encoding="utf-8"))
        base = json.loads((args.reference_run/"scenario_used.json").read_text(encoding="utf-8"))
        assert reference_result["complete"] and not reference_result["diagnostic_only"]
        assert all(hashes[name] == digest for name, digest in reference["code_sha256"].items())
        originals = sorted([json.loads(line) for line in (args.reference_run/"frames.jsonl").read_text(encoding="utf-8").splitlines()], key=lambda r: r["frame_index"])
        assert [r["frame_index"] for r in originals] == list(range(reference["requested_frame_count"]))
        plan = reference["measurement_counts_by_frame"]
        if args.diagnostic_frames is not None:
            plan = plan[:args.diagnostic_frames]
        designs = build_designs(public_settings(base))
        manifest = {"experiment": "OTFS_local_guard_region_sensitivity", "local_zero_delay_row_counts": list(WIDTHS),
            "conditions": list(CONDITIONS), "data_Eb_N0_db": EB_N0_DB, "pilot_energy_Es": 4096,
            "root_seed": reference["root_seed"], "frame_capacity_bits": reference["frame_capacity_bits"],
            "measurement_counts_by_frame": plan, "requested_frame_count": len(plan),
            "requested_bits_per_point": sum(plan), "diagnostic_only": args.diagnostic_frames is not None,
            "reference_errors_by_frame": [[r["errors"][ci][4][2] for ci in (0, 4)] for r in originals],
            "reference_run": str(args.reference_run),
            "reference_checkpoint_sha256": hashlib.sha256((args.reference_run/"frames.jsonl").read_bytes()).hexdigest(),
            "code_sha256": hashes, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "worker_count": args.workers,
            "layout_rule": "W64 is the original rows 96..159. For odd W, local zeros are rows 128-(W-1)/2 .. 128+(W-1)/2 across all 64 Doppler columns; distant reserved rows are 0..(63-W). All designs retain one pilot, 4095 zero cells and 12288 data cells. Common cells keep their symbols; only data displaced by distant reservations move to released local cells. Decode to the original bit order.",
            "receiver_rule": "Freeze the original pilot-only approximate-ML/joint-LS estimator and its original public bounds/observations/search; do not rebuild it as if narrowed masks were isolated. Retain actual unknown-data interference. Build the data-only MMSE operator with each actual known transmit mask, as required by the unchanged receiver equations.",
            "control": "Original PCG64 frame-index/role streams and exact quota/tail. Same pilot energy, data symbols, AWGN/calibration, channel, useful/CP energy and sampled peak. W64 must reproduce original per-frame BER. True channel/data decomposition is evaluator-only and never receiver cancellation."}
        atomic_json(snapshot_path, base)
        atomic_json(args.output/"variant_overrides.json", {"scope": "OTFS-only local-zero-region redistribution", "variants": [d["descriptor"] for d in designs]})
        atomic_json(manifest_path, manifest)
    records = ([json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
               if checkpoint.exists() else [])
    indices = [r["frame_index"] for r in records]
    assert len(set(indices)) == len(indices)
    plan = manifest["measurement_counts_by_frame"]
    assert all(0 <= r["frame_index"] < len(plan) and r["measured_bits"] == plan[r["frame_index"]] for r in records)
    remaining = [(i, bits) for i, bits in enumerate(plan) if i not in set(indices)]
    start = time.perf_counter()
    print(f"OTFS guard: {len(records)}/{len(plan)} complete; {sum(plan)} measured bits/point.", flush=True)
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
                    if len(records) % 4 == 0 or result["complete"]:
                        print(f"OTFS guard: {len(records)}/{len(plan)} frames; {result['measured_bits_per_point']} bits/point; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    result = aggregate(records, manifest)
    assert result["complete"] and result["measured_bits_per_point"] == manifest["requested_bits_per_point"]
    atomic_json(args.output/"results.json", result)
    print(f"Complete: {args.output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
