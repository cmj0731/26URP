"""Compare all-data waveforms using true CSI and the original MMSE detector."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import NormalDist

# Import before NumPy to retain the existing worker BLAS settings.
from run_perfect_csi_comparison import read_frames, read_json, sha256

import numpy as np

from ber_experiment import SCHEMES, apply_paths, receive, transmit
from ber_statistics import frame_plan, summarize_frames
from channel_operators import build_channel_operator
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from qam16 import modulate
from run_ber_comparison import CORE_FILES, atomic_json

REFERENCE = Path("data/perfect_csi_comparison_2026_11_18")
BASE = Path("data/ber_comparison_2026_11_04/multipath/scenario_used.json")
CONDITION = "B-4"
EB_N0_DB = 15
_CONTEXT = None


class FullDataModel:
    """MMSE data operator with the identity data mask and zero pilot response."""

    def __init__(self, channel):
        self.channel = channel
        self.scheme = channel.scheme
        self.grid_shape = channel.grid_shape
        self.data_count = channel.shape[1]
        self.data_mask = np.ones(self.grid_shape, dtype=bool)
        self.data_mask.setflags(write=False)

    def subtract_pilots(self, received_grid):
        return np.array(received_grid, dtype=complex, copy=True)

    def apply(self, data):
        return self.channel.apply(data.reshape(self.grid_shape, order="F"))

    def adjoint(self, received_grid):
        return self.channel.adjoint(received_grid).reshape(-1, order="F")

    def block_data_slice(self, block):
        length = self.grid_shape[0]
        return slice(block*length, (block+1)*length)

    def block_matrix(self, block):
        return self.channel.block_matrix(block)


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    support = public["receiver_information_policy"]["delay_support"]
    tap_delays = tuple(range(support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]+1))
    _CONTEXT = (public, manifest, build_silent_noise_measurement(public), tap_delays)


def compute_frame(reference):
    public, manifest, measurement, tap_delays = _CONTEXT
    index = reference["frame_index"]
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    measured_bits = manifest["measurement_counts_by_frame"][index]
    shape = tuple(manifest["grid_shape"])
    grid = modulate(bits).reshape(shape, order="F")
    cp, fs = manifest["CP_samples"], public["sampling_model"]["sample_rate_hz"]
    signals = {scheme: transmit(scheme, grid, public, cp) for scheme in SCHEMES}
    np.testing.assert_array_equal(signals["OFDM"], signals["DD-a-OFDM"])
    count = signals["OFDM"].size
    delays, dopplers = manifest["path_delays_samples"], manifest["path_dopplers_hz"]
    powers = 10**(np.array(manifest["path_relative_powers_db"])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    unit_noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    calibration_count = measurement.interval_sample_count
    unit_calibration = (calibration_rng.standard_normal(calibration_count)+1j*calibration_rng.standard_normal(calibration_count))/np.sqrt(2)
    variance = 1/(4*10**(EB_N0_DB/10))
    noise = np.sqrt(variance)*unit_noise
    measured_variance = measurement.estimate(np.sqrt(variance)*unit_calibration)
    assert measured_variance == reference["measured_noise_variance"]
    time = np.arange(count)/fs
    true_taps = np.zeros((count, len(tap_delays)), dtype=complex)
    for delay, gain, doppler in zip(delays, gains, dopplers):
        true_taps[:, tap_delays.index(delay)] += gain*np.exp(2j*np.pi*doppler*time)

    useful_energy = float(np.linalg.norm(grid)**2)
    decoded, errors, residuals, model_errors, cp_energies = {}, {}, {}, {}, {}
    for scheme, signal in signals.items():
        slots = signal.reshape(shape[1], shape[0]+cp)
        np.testing.assert_allclose(np.linalg.norm(slots[:, cp:])**2, useful_energy, rtol=1e-12)
        cp_energies[scheme] = float(np.linalg.norm(slots[:, :cp])**2)
        clean = apply_paths(signal, delays, gains, dopplers, fs)
        channel = build_channel_operator(scheme, public, true_taps)
        clean_grid = receive(scheme, clean, public, cp)
        model_errors[scheme] = float(np.linalg.norm(channel.apply(grid)-clean_grid)/np.linalg.norm(clean_grid))
        assert model_errors[scheme] < 1e-11
        received = receive(scheme, clean+noise, public, cp)
        problem = build_mmse_problem(FullDataModel(channel), received, measured_variance, public)
        solved = solve_mmse_problem(problem, public, include_effective_gain=True)
        solved.require_converged()
        decoded[scheme] = recover_data_bits(problem, solved)
        errors[scheme] = int(np.count_nonzero(decoded[scheme][:measured_bits] != bits[:measured_bits]))
        residuals[scheme] = solved.relative_normal_residual
    np.testing.assert_array_equal(decoded["OFDM"], decoded["DD-a-OFDM"])
    return {"frame_index": index, "measured_bits": measured_bits, "bit_errors": errors,
            "reference_original_layout_bits": reference["measured_bits"],
            "reference_original_layout_errors": reference["conditions"][CONDITION]["perfect_csi_errors"],
            "measured_noise_variance": measured_variance, "useful_transmit_energy_Es": useful_energy,
            "CP_transmit_energy_Es": cp_energies, "operator_relative_error": model_errors,
            "normal_relative_residual": residuals, "ofdm_dd_a_ofdm_transmit_and_full_decoded_bits_equal": True}


def aggregate(frames, manifest):
    bits = [frame["measured_bits"] for frame in frames]
    reference_bits = [frame["reference_original_layout_bits"] for frame in frames]
    rows = []
    for scheme in SCHEMES:
        errors = [frame["bit_errors"][scheme] for frame in frames]
        reference_errors = [frame["reference_original_layout_errors"][scheme] for frame in frames]
        current = summarize_frames(errors, bits, full_frame_capacity=manifest["frame_capacity_bits"])
        matched_reference = summarize_frames(reference_errors, reference_bits,
            full_frame_capacity=manifest["reference_frame_capacity_bits"])
        # Both layouts contribute complete frames, with their own constant bit
        # quotas, so the difference of mean per-frame BERs is paired by draw.
        difference = np.array(errors)/bits-np.array(reference_errors)/reference_bits
        change = float(np.mean(difference))
        interval = None
        if len(frames) > 1:
            width = float(NormalDist().inv_cdf(.975)*np.std(difference, ddof=1)/np.sqrt(len(frames)))
            interval = [change-width, change+width]
        rows.append({"waveform": scheme, "pilot_free": current,
                     "original_layout_full_run": manifest["reference_full_run_BER"][scheme],
                     "original_layout_same_frames": matched_reference,
                     "paired_pilot_free_minus_original_BER": change, "paired_95pct_frame_CI": interval,
                     "mean_CP_transmit_energy_Es": float(np.mean([f["CP_transmit_energy_Es"][scheme] for f in frames])),
                     "max_relative_normal_residual": max(f["normal_relative_residual"][scheme] for f in frames),
                     "max_operator_relative_error": max(f["operator_relative_error"][scheme] for f in frames)})
    return {"complete": len(frames) == len(manifest["measurement_counts_by_frame"]), "frame_count": len(frames),
            "measured_bits_per_point": sum(bits), "reference_matched_bits_per_point": sum(reference_bits),
            "mean_useful_transmit_energy_Es": float(np.mean([f["useful_transmit_energy_Es"] for f in frames])),
            "OFDM_DD_a_OFDM_identical_every_frame": all(f["ofdm_dd_a_ofdm_transmit_and_full_decoded_bits_equal"] for f in frames),
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import fast_mmse  # Load the local SciPy installation before starting workers.
    import scipy
    base, reference = read_json(BASE), read_json(REFERENCE/"manifest.json")
    old_results = read_json(REFERENCE/"results.json")
    assert old_results["complete"] and reference["CP_samples"] == 5 and reference["data_Eb_N0_db"] == EB_N0_DB
    assert sha256(BASE) == reference["reference_file_sha256"][BASE.as_posix()]
    sources = (*CORE_FILES, "run_perfect_csi_comparison.py", Path(__file__).name)
    hashes = {name: sha256(Path(name)) for name in sources}
    assert all(hashes[name] == digest for name, digest in reference["source_sha256"].items())
    n, blocks = base["radio"]["frequency_bin_count"], base["frame_design"]["time_symbol_count"]
    assert (n, blocks) == (256, 64)
    capacity = n*blocks*4
    quota = old_results["measured_bits_per_point"]
    counts = frame_plan(quota, capacity)
    assert len(counts) == 160 and all(bits == capacity for bits in counts)
    refs = read_frames(REFERENCE/"frames.jsonl")[:len(counts)]
    assert [f["frame_index"] for f in refs] == list(range(len(counts)))
    assert all(f["measured_bits"] == reference["frame_capacity_bits"] for f in refs)
    manifest = {"experiment": "pilot_free_all_data_perfect_CSI", "condition": CONDITION, "Eb_N0_db": EB_N0_DB,
        "grid_shape": [n, blocks], "data_cells": n*blocks, "pilot_cells": 0, "guard_or_reserved_cells": 0,
        "CP_samples": reference["CP_samples"], "frame_capacity_bits": capacity, "measurement_counts_by_frame": counts,
        "root_seed": reference["root_seed"], "path_delays_samples": reference["path_delays_samples"],
        "path_relative_powers_db": reference["path_relative_powers_db"], "path_dopplers_hz": reference["path_dopplers_hz"][CONDITION],
        "reference_frame_capacity_bits": reference["frame_capacity_bits"], "source_sha256": hashes,
        "reference_file_sha256": {path.as_posix(): sha256(path) for path in (
            BASE, REFERENCE/"manifest.json", REFERENCE/"frames.jsonl", REFERENCE/"results.json")},
        "reference_full_run_BER": {r["waveform"]: r["perfect_csi"] for r in old_results["results"] if r["condition"] == CONDITION},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "controls": "Use every native grid cell for uncoded Gray 16-QAM data. The same bit vector, channel gains, AWGN samples and measured noise variance feed all four waveforms per frame. True sample-clock taps feed the original full-channel MMSE and gain-aware hard decisions. Es=1 and data Eb/N0 set the waveform-independent AWGN variance. Prior-layout full-run BER uses the same total measured bits. Paired differences use the first 160 shared channel/noise frames, with all data bits from each corresponding layout."}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == json.loads(json.dumps(manifest)), "Resume requires the same inputs and sources."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [f["frame_index"] for f in frames] == list(range(len(frames)))
    assert [f["measured_bits"] for f in frames] == counts[:len(frames)]
    print(f"Pilot-free Perfect CSI: {len(frames)}/{len(counts)} frames, {quota:,} bits per point.", flush=True)
    if len(frames) < len(counts):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, refs[len(frames):]):
                    stream.write(json.dumps(frame, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        atomic_json(output/"results.json", aggregate(frames, manifest))
                        print(f"Pilot-free Perfect CSI: {len(frames)}/{len(counts)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["OFDM_DD_a_OFDM_identical_every_frame"]
    assert result["measured_bits_per_point"] == quota
    assert max(row["max_relative_normal_residual"] for row in result["results"]) <= 1e-8
    atomic_json(output/"results.json", result)
    print(f"Complete: {output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
