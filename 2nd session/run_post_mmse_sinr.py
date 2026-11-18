"""Replay the pilot-free Perfect CSI BER frames and score symbol-wise SINR."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

# Load the existing simulation's worker BLAS configuration before NumPy.
from run_pilot_free_perfect_csi import BASE, CONDITION, EB_N0_DB, FullDataModel, read_frames, read_json, sha256

import numpy as np

from ber_experiment import SCHEMES, apply_paths, receive, transmit
from channel_operators import NativeChannelOperator, build_channel_operator
from fast_mmse import eye, inverse_fourier_basis, splu, time_matrix
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from noise_estimation import build_silent_noise_measurement
from qam16 import modulate
from run_ber_comparison import atomic_json

REFERENCE = Path("data/pilot_free_perfect_csi_2026_11_18")
UNIQUE = ("OFDM", "AFDM", "OTFS")
BINS_PER_DB = 50
HISTOGRAM_SIZE = 40*BINS_PER_DB
_CONTEXT = None


def post_mmse_quality(time_channel, afdm_channel, regularization, es, noise_variance):
    """Project K and K² diagonals into the three native symbol bases.

    K=(H_time^H H_time + lambda I)^-1 is block diagonal in useful time.
    The native response is I-lambda*K_native and the filter noise covariance
    is noise_variance*(K_native-lambda*K_native^2).
    """
    n, blocks = time_channel.grid_shape
    assert time_channel.scheme == "OFDM" and afdm_channel.scheme == "AFDM"
    assert np.array_equal(time_channel.taps, afdm_channel.taps)
    unit = np.eye(n, dtype=complex)
    regularizer = regularization*eye(n, format="csc")
    fourier_basis = inverse_fourier_basis(n)
    afdm_basis = afdm_channel.chirp1[:, None]*fourier_basis*afdm_channel.chirp2[None, :]
    diagonal = {scheme: np.empty((n, blocks)) for scheme in UNIQUE}
    square_diagonal = {scheme: np.empty((n, blocks)) for scheme in UNIQUE}
    for block in range(blocks):
        h = time_matrix(time_channel, block)
        k = splu((h.conj().T@h+regularizer).tocsc()).solve(unit)
        diagonal["OTFS"][:, block] = k.diagonal().real
        square_diagonal["OTFS"][:, block] = np.sum(abs(k)**2, axis=0)
        fourier_projected = np.fft.ifft(k, axis=1, norm="ortho")
        afdm_projected = np.fft.ifft(k*afdm_channel.chirp1[None, :], axis=1, norm="ortho")*afdm_channel.chirp2[None, :]
        for scheme, basis, projected in (("OFDM", fourier_basis, fourier_projected),
                                          ("AFDM", afdm_basis, afdm_projected)):
            diagonal[scheme][:, block] = np.sum(basis.conj()*projected, axis=0).real
            square_diagonal[scheme][:, block] = np.sum(abs(projected)**2, axis=0)
    # The DD transform distributes each delay row uniformly over all time
    # slots, so diagonal entries become the mean over slots in that row.
    for values in (diagonal, square_diagonal):
        values["OTFS"][:] = values["OTFS"].mean(axis=1, keepdims=True)
    result = {}
    for scheme in UNIQUE:
        d = diagonal[scheme].reshape(-1, order="F")
        q = square_diagonal[scheme].reshape(-1, order="F")
        gain = 1-regularization*d
        interference = es*regularization**2*(q-d**2)
        assert np.min(interference) >= -1e-13
        interference = np.maximum(interference, 0.)
        output_noise = noise_variance*(d-regularization*q)
        assert np.min(output_noise) > 0.
        sinr = es*gain**2/(interference+output_noise)
        mse = float(np.mean(es*(1-gain)**2+interference+output_noise))
        result[scheme] = {"sinr": sinr, "gain": gain, "mean_mse": mse}
    result["DD-a-OFDM"] = result["OFDM"]
    return result


def check_dense_reference():
    """Use an independent explicit native A and W on a small time-varying frame."""
    n, blocks, cp, lam, es, variance = 8, 3, 2, .013, 1., .009
    t = np.arange((n+cp)*blocks)
    taps = np.column_stack([.90*np.ones(t.size), .28*np.exp(.31j*t), .16*np.exp(-.44j*t)])
    operators = {scheme: NativeChannelOperator(scheme, taps, block_length=n, block_count=blocks, cp_length=cp,
        delays=(0, 1, 2), **({"c1": 3/(2*n), "c2": .017} if scheme == "AFDM" else {})) for scheme in SCHEMES}
    calculated = post_mmse_quality(operators["OFDM"], operators["AFDM"], lam, es, variance)
    size = n*blocks
    unit = np.eye(size, dtype=complex)
    errors = {}
    for scheme, channel in operators.items():
        a = np.column_stack([channel.apply(unit[:, col].reshape((n, blocks), order="F")).reshape(-1, order="F")
                             for col in range(size)])
        w = np.linalg.solve(a.conj().T@a+lam*unit, a.conj().T)
        b = w@a
        desired = es*abs(b.diagonal())**2
        b_off = b-np.diag(b.diagonal())
        dense_sinr = desired/(es*np.sum(abs(b_off)**2, axis=1)+variance*np.sum(abs(w)**2, axis=1))
        errors[scheme] = float(np.max(abs(calculated[scheme]["sinr"]/dense_sinr-1)))
        assert errors[scheme] < 1e-11
    return errors


def initialize_worker(base, manifest):
    global _CONTEXT
    keys = ("radio", "frame_design", "waveform_design", "prefix_policy", "sampling_model",
            "receiver_information_policy", "pilot_comparison_policy", "receiver_equalization_policy",
            "common_symbol_modulation")
    public = {key: base[key] for key in keys}
    support = public["receiver_information_policy"]["delay_support"]
    delays = list(range(support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]+1))
    _CONTEXT = (public, manifest, build_silent_noise_measurement(public), delays)


def pack_histogram(sinr_db, errors):
    indices = np.floor(sinr_db*BINS_PER_DB).astype(int)
    low, high = int(indices.min()), int(indices.max())
    assert 0 <= low <= high < HISTOGRAM_SIZE
    symbols = np.bincount(indices-low, minlength=high-low+1)
    bit_errors = np.bincount(indices-low, weights=errors, minlength=high-low+1).astype(int)
    return {"first_bin": low, "symbol_counts": symbols.tolist(), "bit_error_counts": bit_errors.tolist()}


def compute_frame(reference):
    public, manifest, measurement, tap_delays = _CONTEXT
    index = reference["frame_index"]
    data_rng, phase_rng, noise_rng, calibration_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(manifest["root_seed"], spawn_key=(index, role)))) for role in range(4)]
    bits = data_rng.integers(0, 2, manifest["frame_capacity_bits"], dtype=np.uint8)
    grid = modulate(bits).reshape(manifest["grid_shape"], order="F")
    cp, fs = manifest["CP_samples"], public["sampling_model"]["sample_rate_hz"]
    signals = {scheme: transmit(scheme, grid, public, cp) for scheme in SCHEMES}
    np.testing.assert_array_equal(signals["OFDM"], signals["DD-a-OFDM"])
    count = signals["OFDM"].size
    delays, dopplers = manifest["path_delays_samples"], manifest["path_dopplers_hz"]
    powers = 10**(np.array(manifest["path_relative_powers_db"])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0., phase_rng.uniform(0, 2*np.pi, 2)])
    noise = (noise_rng.standard_normal(count)+1j*noise_rng.standard_normal(count))/np.sqrt(2)
    calibration_count = measurement.interval_sample_count
    calibration = (calibration_rng.standard_normal(calibration_count)+1j*calibration_rng.standard_normal(calibration_count))/np.sqrt(2)
    es = public["common_symbol_modulation"]["constellation_normalization"]["average_symbol_energy"]
    variance = es/(4*10**(EB_N0_DB/10))
    noise = np.sqrt(variance)*noise
    measured_variance = measurement.estimate(np.sqrt(variance)*calibration)
    assert measured_variance == reference["measured_noise_variance"]
    time = np.arange(count)/fs
    true_taps = np.zeros((count, len(tap_delays)), dtype=complex)
    for delay, gain, doppler in zip(delays, gains, dopplers):
        true_taps[:, tap_delays.index(delay)] += gain*np.exp(2j*np.pi*doppler*time)

    channels = {scheme: build_channel_operator(scheme, public, true_taps) for scheme in SCHEMES}
    np.testing.assert_array_equal(channels["OFDM"].taps, channels["DD-a-OFDM"].taps)
    quality = post_mmse_quality(channels["OFDM"], channels["AFDM"], measured_variance/es, es, variance)
    points = {}
    for scheme, signal in signals.items():
        channel = channels[scheme]
        received = apply_paths(signal, delays, gains, dopplers, fs)+noise
        native = receive(scheme, received, public, cp)
        problem = build_mmse_problem(FullDataModel(channel), native, measured_variance, public)
        solved = solve_mmse_problem(problem, public, include_effective_gain=True)
        decoded, detector_gain = recover_data_bits(problem, solved, return_gain=True)
        error_by_symbol = np.count_nonzero((decoded != bits).reshape(-1, 4), axis=1)
        expected = reference["bit_errors"][scheme]
        assert error_by_symbol.sum() == expected, (index, scheme, expected, error_by_symbol.sum())
        assert solved.relative_normal_residual <= 1e-8
        if scheme == "OFDM":
            ofdm_decoded = decoded
        if scheme == "DD-a-OFDM":
            np.testing.assert_array_equal(decoded, ofdm_decoded)
        current = quality[scheme]
        gain_error = float(np.max(abs(detector_gain-current["gain"])))
        assert gain_error < 1e-11
        if scheme in UNIQUE:
            sinr_db = 10*np.log10(current["sinr"])
            points[scheme] = {"histogram": pack_histogram(sinr_db, error_by_symbol),
                "linear_sinr_sum": float(current["sinr"].sum()), "sinr_db_sum": float(sinr_db.sum()),
                "mean_symbol_mse": current["mean_mse"], "bit_errors": int(expected),
                "detector_gain_max_error": gain_error, "max_relative_normal_residual": solved.relative_normal_residual}
    assert np.ptp([p["mean_symbol_mse"] for p in points.values()]) < 1e-12
    row = {"frame_index": index, "measured_bits": bits.size, "waveforms": points,
           "ofdm_dd_a_ofdm_decisions_equal": True}
    if index == 0:
        row["dense_reference_max_relative_sinr_error"] = check_dense_reference()
    return row


def aggregate(frames, manifest):
    total_bits = sum(f["measured_bits"] for f in frames)
    rows = []
    for scheme in UNIQUE:
        points = [f["waveforms"][scheme] for f in frames]
        symbols, errors = np.zeros(HISTOGRAM_SIZE, dtype=np.int64), np.zeros(HISTOGRAM_SIZE, dtype=np.int64)
        for p in points:
            hist = p["histogram"]
            start = hist["first_bin"]
            end = start+len(hist["symbol_counts"])
            symbols[start:end] += hist["symbol_counts"]
            errors[start:end] += hist["bit_error_counts"]
        count, error_count = int(symbols.sum()), int(errors.sum())
        assert count*4 == total_bits and error_count == sum(p["bit_errors"] for p in points)
        cumulative = np.cumsum(symbols)
        quantiles = {}
        for name, rank in (("p01", .01), ("p05", .05), ("p50", .5), ("p95", .95)):
            index = int(np.searchsorted(cumulative, int(np.ceil(rank*count))))
            quantiles[name] = [index/BINS_PER_DB, (index+1)/BINS_PER_DB]
        start, end = np.flatnonzero(symbols)[[0, -1]]
        bands = []
        for db in range(start//BINS_PER_DB, end//BINS_PER_DB+1):
            lo, hi = db*BINS_PER_DB, (db+1)*BINS_PER_DB
            n, err = int(symbols[lo:hi].sum()), int(errors[lo:hi].sum())
            bands.append({"lower_db": int(db), "upper_db": int(db+1), "symbols": n, "bit_errors": err,
                          "BER": err/(4*n) if n else None})
        rows.append({"waveform": scheme, "symbols": count, "bit_errors": error_count, "BER": error_count/total_bits,
            "mean_sinr_db": sum(p["sinr_db_sum"] for p in points)/count,
            "linear_mean_sinr_db": float(10*np.log10(sum(p["linear_sinr_sum"] for p in points)/count)),
            "mean_symbol_mse": float(np.mean([p["mean_symbol_mse"] for p in points])),
            "quantile_bin_intervals_db": quantiles,
            "histogram": {"first_bin": int(start), "symbol_counts": symbols[start:end+1].tolist(),
                          "bit_error_counts": errors[start:end+1].tolist()},
            "bands_1_db": bands, "max_detector_gain_error": max(p["detector_gain_max_error"] for p in points)})
    rows.append({**rows[0], "waveform": "DD-a-OFDM"})
    return {"complete": len(frames) == manifest["frame_count"], "frame_count": len(frames), "bits_per_point": total_bits,
            "histogram_bin_size_db": 1/BINS_PER_DB, "dense_reference_max_relative_sinr_error": frames[0]["dense_reference_max_relative_sinr_error"],
            "ofdm_dd_a_ofdm_decisions_equal_all_frames": all(f["ofdm_dd_a_ofdm_decisions_equal"] for f in frames), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import scipy
    base = read_json(BASE)
    reference = read_json(REFERENCE/"manifest.json")
    reference_result = read_json(REFERENCE/"results.json")
    assert reference_result["complete"] and reference["condition"] == CONDITION == "B-4" and reference["Eb_N0_db"] == EB_N0_DB == 15
    assert reference["CP_samples"] == 5 and reference["grid_shape"] == [256, 64] and reference["pilot_cells"] == 0
    assert sha256(BASE) == reference["reference_file_sha256"][BASE.as_posix()]
    hashes = {name: sha256(Path(name)) for name in (*reference["source_sha256"], Path(__file__).name)}
    assert all(hashes[name] == digest for name, digest in reference["source_sha256"].items())
    refs = read_frames(REFERENCE/"frames.jsonl")
    assert [f["frame_index"] for f in refs] == list(range(160)) and all(f["measured_bits"] == 65536 for f in refs)
    manifest = {key: reference[key] for key in ("condition", "Eb_N0_db", "CP_samples", "root_seed", "grid_shape",
        "frame_capacity_bits", "path_delays_samples", "path_relative_powers_db", "path_dopplers_hz")}
    manifest.update(experiment="post_MMSE_SINR_and_bit_errors", frame_count=len(refs), source_sha256=hashes,
        reference_sha256={path.as_posix(): sha256(path) for path in (BASE, REFERENCE/"manifest.json", REFERENCE/"frames.jsonl", REFERENCE/"results.json")},
        numpy_version=np.__version__, scipy_version=scipy.__version__, histogram_bin_size_db=1/BINS_PER_DB,
        SINR_rule="Exact desired, other-symbol interference and filtered AWGN powers for the all-data linear MMSE filter. The filter uses the original measured variance. SINR evaluation uses the generator AWGN variance with the same filter. Distribution uses half-open 0.02-dB intervals and integer-dB error bands.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, checkpoint = output/"manifest.json", output/"frames.jsonl"
    if args.resume:
        assert read_json(manifest_path) == manifest, "Resume requires the same files and numerical inputs."
    else:
        if manifest_path.exists() or checkpoint.exists():
            parser.error("Output already exists. Use --resume or choose a new directory.")
        atomic_json(manifest_path, manifest)
    frames = read_frames(checkpoint) if checkpoint.exists() else []
    assert [f["frame_index"] for f in frames] == list(range(len(frames)))
    print(f"SINR and errors: {len(frames)}/{len(refs)} frames.", flush=True)
    if len(frames) < len(refs):
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(base, manifest)) as pool:
            with checkpoint.open("a", encoding="utf-8") as stream:
                for frame in pool.map(compute_frame, refs[len(frames):]):
                    stream.write(json.dumps(frame, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    frames.append(frame)
                    if len(frames)%8 == 0:
                        atomic_json(output/"results.json", aggregate(frames, manifest))
                        print(f"SINR and errors: {len(frames)}/{len(refs)} frames.", flush=True)
    result = aggregate(frames, manifest)
    assert result["complete"] and result["bits_per_point"] == 10485760 and result["ofdm_dd_a_ofdm_decisions_equal_all_frames"]
    for row in result["results"]:
        original = next(r for r in reference_result["results"] if r["waveform"] == row["waveform"])
        assert row["bit_errors"] == original["pilot_free"]["bit_errors"]
    atomic_json(output/"results.json", result)
    print(f"Complete: {output/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
