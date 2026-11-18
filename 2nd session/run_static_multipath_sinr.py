"""Pair the pilot-free B-4 diagnostic with a zero-residual-Doppler control."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import NormalDist

# The earlier runner configures BLAS before loading NumPy and SciPy.
from run_post_mmse_sinr import BASE, BINS_PER_DB, EB_N0_DB, UNIQUE, aggregate, check_dense_reference, pack_histogram
from run_post_mmse_sinr import post_mmse_quality, read_frames, read_json, sha256

import numpy as np

from ber_experiment import SCHEMES, apply_paths, receive, transmit
from ber_statistics import summarize_frames
from channel_operators import build_channel_operator
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from qam16 import modulate
from run_ber_comparison import atomic_json
from run_pilot_free_perfect_csi import FullDataModel, REFERENCE as PERFECT_CSI_REFERENCE

B4 = Path("data/post_mmse_sinr_2026_11_18")
PILOT_FREE = Path("data/pilot_free_perfect_csi_2026_11_18")
THRESHOLD_DB = 17
_CONTEXT = None


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    support = public["receiver_information_policy"]["delay_support"]
    delay_support = list(range(support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]+1))
    powers = 10**(np.array(manifest["path_relative_powers_db"])/10)
    _CONTEXT = (public, manifest, build_silent_noise_measurement(public), delay_support, np.sqrt(powers/powers.sum()))


def compute_frame(reference):
    public, manifest, measurement, delay_support, magnitudes = _CONTEXT
    index = reference["frame_index"]
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    n, blocks = manifest["grid_shape"]
    grid = modulate(bits).reshape((n, blocks), order="F")
    cp, fs = manifest["CP_samples"], public["sampling_model"]["sample_rate_hz"]
    signals = {scheme: transmit(scheme, grid, public, cp) for scheme in SCHEMES}
    np.testing.assert_array_equal(signals["OFDM"], signals["DD-a-OFDM"])
    count = signals["OFDM"].size
    delays, dopplers = manifest["path_delays_samples"], manifest["path_dopplers_hz"]
    assert dopplers == [0., 0., 0.]
    gains = magnitudes*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    unit_noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    calibration_count = measurement.interval_sample_count
    calibration = (calibration_rng.standard_normal(calibration_count)+1j*calibration_rng.standard_normal(calibration_count))/np.sqrt(2)
    es = public["common_symbol_modulation"]["constellation_normalization"]["average_symbol_energy"]
    variance = es/(4*10**(EB_N0_DB/10))
    noise = np.sqrt(variance)*unit_noise
    measured_variance = measurement.estimate(np.sqrt(variance)*calibration)
    assert measured_variance == reference["measured_noise_variance"]
    np.testing.assert_allclose(np.linalg.norm(grid)**2, reference["useful_transmit_energy_Es"], rtol=1e-13)
    true_taps = np.zeros((count, len(delay_support)), dtype=complex)
    for delay, gain in zip(delays, gains):
        true_taps[:, delay_support.index(delay)] += gain

    channels = {scheme: build_channel_operator(scheme, public, true_taps) for scheme in SCHEMES}
    quality = post_mmse_quality(channels["OFDM"], channels["AFDM"], measured_variance/es, es, variance)
    # With static multipath, OFDM is diagonal and the scalar MMSE gain
    # cancels from SINR. Verify every computed OFDM symbol against Es|H_k|²/σ².
    frequency_response = np.fft.fft(true_taps[0], n=n)
    closed_form = np.tile(es*abs(frequency_response)**2/variance, blocks)
    static_sinr_error = float(np.max(abs(quality["OFDM"]["sinr"]/closed_form-1.)))
    assert static_sinr_error < 1e-11

    points = {}
    for scheme, signal in signals.items():
        channel = channels[scheme]
        clean = apply_paths(signal, delays, gains, dopplers, fs)
        native_clean = receive(scheme, clean, public, cp)
        operator_error = float(np.linalg.norm(channel.apply(grid)-native_clean)/np.linalg.norm(native_clean))
        assert operator_error < 1e-11
        received = receive(scheme, clean+noise, public, cp)
        problem = build_mmse_problem(FullDataModel(channel), received, measured_variance, public)
        solved = solve_mmse_problem(problem, public, include_effective_gain=True)
        decoded, detector_gain = recover_data_bits(problem, solved, return_gain=True)
        assert solved.relative_normal_residual <= 1e-8
        errors = np.count_nonzero((decoded != bits).reshape(-1, 4), axis=1)
        current = quality[scheme]
        gain_error = float(np.max(abs(detector_gain-current["gain"])))
        assert gain_error < 1e-11
        if scheme == "OFDM":
            ofdm_decoded = decoded
        if scheme == "DD-a-OFDM":
            np.testing.assert_array_equal(decoded, ofdm_decoded)
        if scheme in UNIQUE:
            sinr_db = 10*np.log10(current["sinr"])
            points[scheme] = {"histogram": pack_histogram(sinr_db, errors),
                "linear_sinr_sum": float(current["sinr"].sum()), "sinr_db_sum": float(sinr_db.sum()),
                "mean_symbol_mse": current["mean_mse"], "bit_errors": int(errors.sum()),
                "detector_gain_max_error": gain_error, "max_relative_normal_residual": solved.relative_normal_residual,
                "channel_operator_relative_error": operator_error}
    assert np.ptp([p["mean_symbol_mse"] for p in points.values()]) < 1e-12
    row = {"frame_index": index, "measured_bits": bits.size, "waveforms": points, "measured_noise_variance": measured_variance,
           "OFDM_closed_form_max_relative_SINR_error": static_sinr_error, "ofdm_dd_a_ofdm_decisions_equal": True}
    if index == 0:
        row["dense_reference_max_relative_sinr_error"] = check_dense_reference()
    return row


def paired_summary(a_frames, b4_frames, a_result, b4_result, capacity):
    total_frames = len(a_frames)
    a, b4 = (dict((r["waveform"], r) for r in result["results"]) for result in (a_result, b4_result))
    rows = []
    for scheme in SCHEMES:
        recorded = "OFDM" if scheme == "DD-a-OFDM" else scheme
        a_errors = np.array([f["waveforms"][recorded]["bit_errors"] for f in a_frames])
        b4_errors = np.array([f["waveforms"][recorded]["bit_errors"] for f in b4_frames])
        bits = [capacity]*total_frames
        rates = (b4_errors-a_errors)/capacity
        change = float(rates.mean())
        half_width = float(NormalDist().inv_cdf(.975)*rates.std(ddof=1)/np.sqrt(total_frames))
        def summarize(row, errors):
            below = [b for b in row["bands_1_db"] if b["upper_db"] <= THRESHOLD_DB]
            lower_symbols = sum(b["symbols"] for b in below)
            lower_errors = sum(b["bit_errors"] for b in below)
            return {"BER": summarize_frames(errors, bits, full_frame_capacity=capacity),
                "quantile_bin_intervals_db": row["quantile_bin_intervals_db"], "linear_mean_sinr_db": row["linear_mean_sinr_db"],
                "symbols_below_17_dB": lower_symbols, "errors_below_17_dB": lower_errors,
                "symbol_share_below_17_dB": lower_symbols/row["symbols"],
                "error_share_below_17_dB": lower_errors/row["bit_errors"] if row["bit_errors"] else None}
        rows.append({"waveform": scheme, "A": summarize(a[scheme], a_errors), "B-4": summarize(b4[scheme], b4_errors),
                     "BER_change_B4_minus_A": change, "paired_95pct_frame_cluster_CI": [change-half_width, change+half_width]})
    return {"complete": True, "matched_frames": total_frames, "bits_per_condition_waveform": total_frames*capacity,
            "low_SINR_threshold_db_exclusive": THRESHOLD_DB, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import scipy
    base, previous = read_json(BASE), read_json(B4/"manifest.json")
    perfect_reference = read_json(PERFECT_CSI_REFERENCE/"manifest.json")
    dopplers = perfect_reference["path_dopplers_hz"]
    assert previous["condition"] == "B-4" and previous["path_dopplers_hz"] == dopplers["B-4"]
    assert previous["CP_samples"] == 5 and previous["Eb_N0_db"] == EB_N0_DB == 15
    hashes = {name: sha256(Path(name)) for name in (*previous["source_sha256"], Path(__file__).name)}
    assert all(hashes[name] == digest for name, digest in previous["source_sha256"].items())
    assert all(sha256(Path(name)) == digest for name, digest in previous["reference_sha256"].items())
    b4_result, b4_frames = read_json(B4/"results.json"), read_frames(B4/"frames.jsonl")
    refs = read_frames(PILOT_FREE/"frames.jsonl")
    assert b4_result["complete"] and len(refs) == len(b4_frames) == 160
    for index, (ref, b4) in enumerate(zip(refs, b4_frames)):
        assert ref["frame_index"] == b4["frame_index"] == index and ref["measured_bits"] == b4["measured_bits"] == 65536
        assert all(ref["bit_errors"][scheme] == b4["waveforms"][scheme]["bit_errors"] for scheme in UNIQUE)
    manifest = {key: previous[key] for key in ("Eb_N0_db", "CP_samples", "root_seed", "grid_shape", "frame_capacity_bits",
        "path_delays_samples", "path_relative_powers_db", "frame_count", "histogram_bin_size_db", "SINR_rule")}
    manifest.update(condition="A", path_dopplers_hz=dopplers["A"], comparison_condition="B-4",
        comparison_reference=B4.as_posix(), source_sha256=hashes, reference_sha256={path.as_posix():sha256(path) for path in (
            BASE, B4/"manifest.json", B4/"results.json", B4/"frames.jsonl", PILOT_FREE/"frames.jsonl", PERFECT_CSI_REFERENCE/"manifest.json")},
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        controls="Replay the exact pilot-free B-4 frame seeds, data, path gains, AWGN and measured noise. All three path Dopplers are zero in condition A. Match the 160 complete frames by index for BER and SINR-tail comparisons.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == manifest, "Resume requires identical sources and references."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [f["frame_index"] for f in frames] == list(range(len(frames)))
    print(f"Static multipath SINR: {len(frames)}/{len(refs)} frames.", flush=True)
    if len(frames) < len(refs):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, refs[len(frames):]):
                    stream.write(json.dumps(frame, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        print(f"Static multipath SINR: {len(frames)}/{len(refs)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["bits_per_point"] == 10485760
    assert result["ofdm_dd_a_ofdm_decisions_equal_all_frames"]
    atomic_json(output/"results.json", result)
    atomic_json(output/"comparison.json", paired_summary(frames, b4_frames, result, b4_result, manifest["frame_capacity_bits"]))
    print(f"Complete: {output/'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
