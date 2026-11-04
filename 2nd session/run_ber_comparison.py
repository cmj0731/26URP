"""Run/resume the fixed-quota paired BER study with bounded worker processes."""

import os
# Small dense blocks run efficiently without multiplying BLAS threads by the
# worker count. These settings affect this process/children, not system setup.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ber_experiment import SCHEMES, POINTS, compute_frame, initialize_worker, theory_16qam
from ber_statistics import frame_plan, summarize_frames

CORE_FILES = ("ber_experiment.py", "ber_statistics.py", "afdm.py", "otfs.py",
              "cyclic_prefix.py", "pilot_layouts.py", "pilot_data_model.py", "channel_operators.py",
              "bem_basis.py", "ofdm_bem.py", "afdm_estimation.py", "otfs_estimation.py",
              "dd_a_estimation.py", "pilot_path_search.py", "noise_estimation.py", "mmse_problem.py",
              "mmse_solver.py", "mmse_demapping.py", "fast_mmse.py", "qam16.py")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for attempt in range(30):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def aggregate(records, settings, manifest):
    ordered = sorted(records, key=lambda row: row["frame_index"])
    counts = np.array([row["measured_bits"] for row in ordered], dtype=np.int64)
    errors = np.array([row["errors"] for row in ordered], dtype=np.int64)
    baseline = manifest["stage"] == "baseline"
    conditions = (["peak", "exit"] if baseline else
        settings["channel_model"]["multipath_followup"]["doppler_effect_followup"]["sweep_magnitude_design"]["condition_labels"])
    confidence = manifest["confidence_level"]
    rows = []
    for ci, condition in enumerate(conditions):
        for pi, point in enumerate(POINTS):
            for si, scheme in enumerate(SCHEMES):
                summary = summarize_frames(errors[:, ci, pi, si], counts,
                    confidence=confidence, full_frame_capacity=manifest["frame_capacity_bits"])
                summary.update(condition=condition, waveform=scheme, Eb_N0_db=point, noiseless=point is None)
                if baseline and point is not None:
                    summary["ideal_Gray_16QAM_theory_BER"] = theory_16qam(point)
                    se = summary["frame_standard_error"]
                    summary["theory_discrepancy_over_frame_standard_error"] = (
                        (summary["ber_estimate"]-summary["ideal_Gray_16QAM_theory_BER"])/se if se else None)
                rows.append(summary)
    return {"stage": manifest["stage"], "complete": len(records) == manifest["requested_frame_count"],
            "diagnostic_only": manifest["diagnostic_only"],
            "requested_bits_per_point": manifest["requested_bits_per_point"],
            "completed_frame_count": len(records), "requested_frame_count": manifest["requested_frame_count"],
            "measured_bits_per_point": int(counts.sum()), "confidence_level": confidence,
            "max_relative_normal_residual_by_waveform": dict(zip(SCHEMES,
                np.max([row["max_relative_normal_residual"] for row in ordered], axis=0).tolist())),
            "max_iterations": int(max(np.max(row["iterations"]) for row in ordered)),
            "results": rows,
            "interpretation": "Same quota and paired frame inputs across waveforms/conditions/Eb/N0. No error-count early stop. Zero observed BER is not asserted to be true zero. Frame-cluster intervals allow within-frame dependence; zero-error bounds are deliberately conservative. Noiseless points are separate references, not finite Eb/N0 values."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "multipath"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--diagnostic-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.seed < 0 or (args.diagnostic_frames is not None and args.diagnostic_frames < 1):
        parser.error("Worker/frame counts must be positive and the seed nonnegative.")
    root = Path(__file__).parent
    scipy_version = None
    if args.stage == "multipath":
        import fast_mmse
        import scipy
        scipy_version = scipy.__version__
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path, snapshot_path = args.output/"manifest.json", args.output/"scenario_used.json"
    checkpoint = args.output/"frames.jsonl"
    code_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CORE_FILES}
    if args.stage == "baseline":
        names = ("ber_experiment.py", "ber_statistics.py", "afdm.py", "otfs.py", "cyclic_prefix.py", "qam16.py")
        code_hashes = {name: code_hashes[name] for name in names}
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if (manifest["stage"] != args.stage
                or any(manifest["code_sha256"].get(name) != value for name,value in code_hashes.items())):
            raise RuntimeError("Resume requires the same stage and numerical source; use a new output directory after code changes.")
    else:
        if manifest_path.exists() or checkpoint.exists():
            raise RuntimeError("Output already contains a run. Use --resume or another output directory.")
        settings = json.loads((root/"scenario.json").read_text(encoding="utf-8"))
        capacity = (65536 if args.stage == "baseline" else
                    settings["evaluation"]["monte_carlo_measurement"]["pilot_stage_framing_budget"]["data_bits_per_full_frame"])
        quota = settings["evaluation"]["monte_carlo_measurement"]["information_bits_per_measurement"]
        plan = frame_plan(quota, capacity)
        diagnostic = args.diagnostic_frames is not None
        if diagnostic:
            plan = plan[:args.diagnostic_frames]
        manifest = {"stage": args.stage, "root_seed": args.seed, "frame_capacity_bits": capacity,
                    "requested_bits_per_point": int(sum(plan)), "requested_frame_count": len(plan),
                    "diagnostic_only": diagnostic,
                    "confidence_level": settings["evaluation"]["monte_carlo_measurement"]["reporting"]["confidence_interval"]["level"],
                    "measurement_counts_by_frame": plan, "code_sha256": code_hashes,
                    "numpy_version": np.__version__, "worker_count": args.workers,
                    "scipy_version": scipy_version,
                    "final_frame_rule": "Generate a complete normal random-data frame and recover all data cells. Count only the remaining information-bit prefix. The uncounted suffix stays unknown random data, not zero/known padding; frame/pilot/calibration/energy are unchanged.",
                    "seed_rule": "PCG64(SeedSequence(root_seed,spawn_key=(frame_index,role))); roles=data,channel,data-interval-noise,calibration-noise. Reuse within a frame across all waveforms/profiles/Eb/N0, redraw independently between frames. Worker scheduling does not affect generated samples.",
                    "resume_rule": "Resume uses the saved exact configuration snapshot and the same numerical-source hashes. Completed frame records are immutable; metadata changes to the live scenario do not modify an existing run."}
        atomic_json(snapshot_path, settings)
        atomic_json(manifest_path, manifest)
    records = []
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    indices = [row["frame_index"] for row in records]
    if len(set(indices)) != len(indices):
        raise RuntimeError("Duplicate completed-frame checkpoint records.")
    plan = manifest["measurement_counts_by_frame"]
    for row in records:
        if not 0 <= row["frame_index"] < len(plan) or row["measured_bits"] != plan[row["frame_index"]]:
            raise RuntimeError("Checkpoint frame counts disagree with the saved measurement plan.")
    remaining = [(i, bits) for i, bits in enumerate(plan) if i not in set(indices)]
    start = time.perf_counter()
    print(f"{args.stage}: {len(records)}/{len(plan)} frames already complete; {manifest['requested_bits_per_point']} measured bits per point.", flush=True)
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=(settings, args.stage, manifest["root_seed"])) as pool:
            futures = {pool.submit(compute_frame, job): job[0] for job in remaining}
            with checkpoint.open("a", encoding="utf-8") as stream:
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception as error:
                        atomic_json(args.output/"failure.json", {"frame_index": futures[future], "error": repr(error),
                            "rule": "No failed frame is counted or silently dropped. Completed frames remain resumable; numerical/implementation failure must be resolved before final BER."})
                        for other in futures:
                            other.cancel()
                        raise
                    stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                    stream.flush()
                    records.append(row)
                    result = aggregate(records, settings, manifest)
                    if len(records) % 4 == 0 or result["complete"]:
                        atomic_json(args.output/"results.json", result)
                    elapsed = time.perf_counter()-start
                    print(f"{args.stage}: {len(records)}/{len(plan)} frames; {result['measured_bits_per_point']} bits/point; elapsed {elapsed:.1f}s", flush=True)
    result = aggregate(records, settings, manifest)
    if not result["complete"] or result["measured_bits_per_point"] != manifest["requested_bits_per_point"]:
        raise RuntimeError("The run is incomplete or does not meet the exact quota.")
    atomic_json(args.output/"results.json", result)
    print(f"Complete: {args.output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
