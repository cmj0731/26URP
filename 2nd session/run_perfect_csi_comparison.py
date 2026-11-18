"""Compare estimated and true channel inputs on the same received frames."""

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
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

ORIGINAL = Path("data/ber_comparison_2026_11_04/multipath")
SPARSE_OFDM = Path("data/ofdm_estimator_comparison_2026_11_18")
CONDITIONS = ("A", "B-4")
EB_N0_DB = 15
_CONTEXT = None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sorted(rows, key=lambda frame: frame["frame_index"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    support = public["receiver_information_policy"]["delay_support"]
    tap_delays = list(range(support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]+1))
    assert public["prefix_policy"]["prefix_samples_per_block"] == manifest["CP_samples"] == 5
    assert public["pilot_comparison_policy"]["total_pilot_energy"] == manifest["pilot_energy_Es"] == 4096
    _CONTEXT = (public, manifest, build_silent_noise_measurement(public), tap_delays)


def recover_frame(scheme, grid, clean, noise, true_taps, measured_variance, public, cp):
    channel = build_channel_operator(scheme, public, true_taps)
    native_clean = receive(scheme, clean, public, cp)
    model_error = float(np.linalg.norm(channel.apply(grid)-native_clean)/np.linalg.norm(native_clean))
    assert model_error < 1e-11
    received = receive(scheme, clean+noise, public, cp)
    model = build_pilot_data_model(channel, public)
    problem = build_mmse_problem(model, received, measured_variance, public)
    solved = solve_mmse_problem(problem, public, include_effective_gain=True)
    solved.require_converged()
    return recover_data_bits(problem, solved), model_error, solved.relative_normal_residual


def compute_frame(references):
    original, sparse = references
    public, manifest, measurement, tap_delays = _CONTEXT
    index, quota = original["frame_index"], original["measured_bits"]
    assert (sparse["frame_index"], sparse["measured_bits"]) == (index, quota)
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    symbols = modulate(bits)
    cp, fs = manifest["CP_samples"], public["sampling_model"]["sample_rate_hz"]
    grids = {scheme: build_grid(scheme, symbols, scenario=public) for scheme in SCHEMES}
    signals = {scheme: transmit(scheme, grids[scheme], public, cp) for scheme in SCHEMES}
    np.testing.assert_array_equal(signals["OFDM"], signals["DD-a-OFDM"])
    size = signals["OFDM"].size
    delays = manifest["path_delays_samples"]
    powers = 10**(np.array(manifest["path_relative_powers_db"])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    unit_noise = (noise_rng.standard_normal(size)+1j*noise_rng.standard_normal(size))/np.sqrt(2)
    count = measurement.interval_sample_count
    calibration = (calibration_rng.standard_normal(count)+1j*calibration_rng.standard_normal(count))/np.sqrt(2)
    variance = 1/(4*10**(EB_N0_DB/10))
    measured_variance = measurement.estimate(np.sqrt(variance)*calibration)
    assert measured_variance == sparse["measured_noise_variance"]
    noise = np.sqrt(variance)*unit_noise
    time = np.arange(size)/fs
    points = {}
    for condition in CONDITIONS:
        dopplers = manifest["path_dopplers_hz"][condition]
        true_taps = np.zeros((size, len(tap_delays)), dtype=complex)
        for delay, gain, doppler in zip(delays, gains, dopplers):
            true_taps[:, tap_delays.index(delay)] += gain*np.exp(2j*np.pi*doppler*time)
        decoded, errors, model_errors, residuals = {}, {}, {}, {}
        for scheme in SCHEMES:
            clean = apply_paths(signals[scheme], delays, gains, dopplers, fs)
            decoded[scheme], model_errors[scheme], residuals[scheme] = recover_frame(
                scheme, grids[scheme], clean, noise, true_taps, measured_variance, public, cp)
            errors[scheme] = int(np.count_nonzero(decoded[scheme][:quota] != bits[:quota]))
        np.testing.assert_array_equal(decoded["OFDM"], decoded["DD-a-OFDM"])
        ci = manifest["reference_condition_indices"][condition]
        original_errors = dict(zip(SCHEMES, original["errors"][ci][POINTS.index(EB_N0_DB)]))
        sparse_errors = sparse["conditions"][condition]["methods"]["3-path"]["bit_errors"]
        points[condition] = {"perfect_csi_errors": errors, "original_estimated_csi_errors": original_errors,
                             "sparse_ofdm_errors": sparse_errors, "channel_operator_relative_errors": model_errors,
                             "normal_residuals": residuals, "ofdm_and_dd_a_ofdm_decoded_bits_equal": True}
    return {"frame_index": index, "measured_bits": quota, "measured_noise_variance": measured_variance,
            "ofdm_and_dd_a_ofdm_transmit_equal": True, "conditions": points}


def paired_change(perfect, estimated, bits):
    delta = np.array(perfect)-np.array(estimated)
    total = sum(bits)
    change = float(delta.sum()/total)
    if len(bits) < 2:
        return {"perfect_minus_estimated_BER": change, "paired_95pct_CI": None}
    se = np.sqrt(len(bits)/(len(bits)-1)*np.sum((delta-change*np.array(bits))**2))/total
    width = float(NormalDist().inv_cdf(.975)*se)
    return {"perfect_minus_estimated_BER": change, "paired_95pct_CI": [change-width, change+width]}


def aggregate(frames, manifest):
    bits = [frame["measured_bits"] for frame in frames]
    summarize = lambda errors: summarize_frames(errors, bits, full_frame_capacity=manifest["frame_capacity_bits"])
    result_rows = []
    for condition in CONDITIONS:
        points = [frame["conditions"][condition] for frame in frames]
        assert all(p["ofdm_and_dd_a_ofdm_decoded_bits_equal"] for p in points)
        for scheme in SCHEMES:
            perfect = [p["perfect_csi_errors"][scheme] for p in points]
            estimated = [p["original_estimated_csi_errors"][scheme] for p in points]
            row = {"condition": condition, "waveform": scheme, "perfect_csi": summarize(perfect),
                   "original_estimated_csi": summarize(estimated), "paired_change": paired_change(perfect, estimated, bits),
                   "max_channel_operator_relative_error": max(p["channel_operator_relative_errors"][scheme] for p in points),
                   "max_normal_residual": max(p["normal_residuals"][scheme] for p in points)}
            assert row["max_normal_residual"] <= 1e-8
            if scheme == "OFDM":
                sparse = [p["sparse_ofdm_errors"] for p in points]
                row["three_path_estimated_csi"] = summarize(sparse)
                row["paired_change_vs_three_path"] = paired_change(perfect, sparse, bits)
            result_rows.append(row)
    return {"complete": len(frames) == manifest["requested_frame_count"], "completed_frame_count": len(frames),
            "measured_bits_per_point": sum(bits), "Eb_N0_db": EB_N0_DB,
            "OFDM_DD_a_OFDM_identical_transmit_and_perfect_CSI_decisions": all(
                frame["ofdm_and_dd_a_ofdm_transmit_equal"] for frame in frames), "results": result_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import fast_mmse  # Make the workspace's optional SciPy installation available.
    import scipy
    base = read_json(ORIGINAL/"scenario_used.json")
    original_manifest = read_json(ORIGINAL/"manifest.json")
    sparse_manifest = read_json(SPARSE_OFDM/"manifest.json")
    original_results, sparse_results = read_json(ORIGINAL/"results.json"), read_json(SPARSE_OFDM/"results.json")
    assert original_results["complete"] and not original_results["diagnostic_only"] and sparse_results["complete"]
    source_hashes = {name: sha256(Path(name)) for name in (*CORE_FILES, Path(__file__).name)}
    assert all(source_hashes[name] == digest for name, digest in original_manifest["code_sha256"].items())
    references = {str(path.as_posix()): sha256(path) for path in (
        ORIGINAL/"scenario_used.json", ORIGINAL/"frames.jsonl", SPARSE_OFDM/"frames.jsonl")}
    assert references[(ORIGINAL/"frames.jsonl").as_posix()] == sparse_manifest["reference_frames_sha256"]
    assert references[(ORIGINAL/"scenario_used.json").as_posix()] == sparse_manifest["reference_scenario_sha256"]
    original_frames, sparse_frames = read_frames(ORIGINAL/"frames.jsonl"), read_frames(SPARSE_OFDM/"frames.jsonl")
    counts = original_manifest["measurement_counts_by_frame"]
    assert counts == sparse_manifest["measurement_counts_by_frame"] and len(counts) == 214 and sum(counts) == 10485760
    assert original_manifest["root_seed"] == sparse_manifest["root_seed"] and sparse_manifest["data_Eb_N0_db"] == EB_N0_DB
    for reference in (original_frames, sparse_frames):
        assert [frame["frame_index"] for frame in reference] == list(range(len(counts)))
        assert [frame["measured_bits"] for frame in reference] == counts
    manifest = {"experiment": "four_method_paired_perfect_CSI_BER", "waveforms": SCHEMES,
        "conditions": CONDITIONS, "data_Eb_N0_db": EB_N0_DB, "CP_samples": 5, "pilot_energy_Es": 4096,
        "root_seed": original_manifest["root_seed"], "frame_capacity_bits": original_manifest["frame_capacity_bits"],
        "measurement_counts_by_frame": counts, "requested_frame_count": len(counts), "source_sha256": source_hashes,
        "reference_file_sha256": references, "reference_condition_indices": sparse_manifest["reference_condition_indices"],
        "path_delays_samples": sparse_manifest["path_delays_samples"], "path_relative_powers_db": sparse_manifest["path_relative_powers_db"],
        "path_dopplers_hz": sparse_manifest["path_dopplers_hz"], "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "receiver_controls": "Original pilot/data masks, CP, PCG64 data/channel/noise/calibration draws, phase and noise accounting, measured-noise MMSE, effective gain and Gray hard decisions. Supply exact CP-inclusive channel taps to pilot subtraction and detection only in the perfect-CSI control. Estimated-CSI errors come from the original paired runs."}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == json.loads(json.dumps(manifest)), "Resume requires the same sources and references."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [frame["frame_index"] for frame in frames] == list(range(len(frames)))
    assert [frame["measured_bits"] for frame in frames] == counts[:len(frames)]
    print(f"Perfect CSI: {len(frames)}/{len(counts)} frames, {sum(counts):,} bits per point.", flush=True)
    if len(frames) < len(counts):
        jobs = list(zip(original_frames, sparse_frames))[len(frames):]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, jobs):
                    stream.write(json.dumps(frame, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        atomic_json(output/"results.json", aggregate(frames, manifest))
                        print(f"Perfect CSI: {len(frames)}/{len(counts)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["OFDM_DD_a_OFDM_identical_transmit_and_perfect_CSI_decisions"]
    for row in result["results"]:
        original = next(x for x in original_results["results"] if x["waveform"] == row["waveform"]
                        and x["condition"] == row["condition"] and x["Eb_N0_db"] == EB_N0_DB)
        for key in ("bit_errors", "confidence_interval"):
            assert row["original_estimated_csi"][key] == original[key]
        if row["waveform"] == "OFDM":
            sparse = next(x for x in sparse_results["results"] if x["condition"] == row["condition"] and x["method"] == "3-path")
            for key in ("bit_errors", "confidence_interval"):
                assert row["three_path_estimated_csi"][key] == sparse[key]
    atomic_json(output/"results.json", result)
    print(f"Complete: {output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
