"""Screen AFDM chirps on sampled 16-QAM differences in an ideal cyclic block.

This diagnostic is not a BER transceiver, a prefix design, or an exhaustive
full-diversity proof. Per-path operators exclude the scalar path gains.
"""

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = 20261103
RELATIVE_SVD_THRESHOLD = 1e-10


def sampled_differences(length: int):
    rng = np.random.default_rng(SEED)
    scale = math.sqrt(10)
    alphabet = np.array([
        complex(real, imag) / scale
        for real in range(-6, 7, 2)
        for imag in range(-6, 7, 2)
        if (real, imag) != (0, 0)
    ])
    for index in range(length):
        delta = np.zeros(length, dtype=complex)
        delta[index] = 2 / scale
        yield "single_position", delta
    for _ in range(1024):
        delta = np.zeros(length, dtype=complex)
        positions = rng.choice(length, 2, replace=False)
        delta[positions] = rng.choice(alphabet, 2)
        yield "two_position", delta
    for _ in range(1024):
        left = rng.choice([-3, -1, 1, 3], size=(2, length))
        right = rng.choice([-3, -1, 1, 3], size=(2, length))
        delta = ((left[0] - right[0]) + 1j * (left[1] - right[1])) / scale
        yield "dense_16qam_difference", delta
    for period in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if length % period:
            continue
        pattern = rng.choice(alphabet, period)
        yield "periodic", np.tile(pattern, length // period)
        delta = np.zeros(length, dtype=complex)
        delta[::period] = 2 / scale
        yield "comb", delta
    phases = np.array([1, 1j, -1, -1j])
    for period in [1, 2, 4]:
        yield "phase_pattern", phases[(np.arange(length) * (4 // period)) % 4] * 2 / scale


def main() -> None:
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    afdm = scenario["waveform_design"]["AFDM"]
    length = afdm["daft_length"]
    c1 = afdm["chirp_parameters"]["c1"]
    c2 = afdm["chirp_parameters"]["c2"]
    profile = scenario["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    if scenario["sampling_model"]["sample_rate_hz"] != 60_000_000:
        raise ValueError("Stored sample delays refer to 60 MHz; review the diagnostic grid.")
    delays = [path["excess_delay_samples_at_60mhz"] for path in paths]
    if len(delays) != profile["total_path_count_including_los"]:
        raise ValueError("Path count differs from the stored delay profile.")
    if any(not isinstance(delay, int) or not 0 <= delay < length for delay in delays):
        raise ValueError("This diagnostic needs integer delays within the cyclic block.")
    if any(path["additional_doppler_hz_to_los"] != 0 for path in paths):
        raise ValueError("Review the path operators before checking differential Doppler.")
    if c1 is None or c2 is None or not all(math.isfinite(c) for c in [c1, c2]):
        raise ValueError("Finite c1 and c2 must be selected before the rank screen.")

    index = np.arange(length, dtype=float)
    chirp1 = np.exp(2j * np.pi * c1 * index * index)
    chirp2 = np.exp(2j * np.pi * c2 * index * index)

    def transmit(delta):
        return chirp1 * np.fft.ifft(chirp2 * delta, norm="ortho")

    def receive(signal):
        return chirp2.conj() * np.fft.fft(chirp1.conj() * signal, norm="ortho")

    minimum_rank = len(delays)
    minimum_ratio = 1.0
    maximum_roundtrip_error = 0.0
    maximum_energy_error = 0.0
    counts = Counter()
    for name, delta in sampled_differences(length):
        counts[name] += 1
        signal = transmit(delta)
        roundtrip_error = np.linalg.norm(receive(signal) - delta) / np.linalg.norm(delta)
        energy_error = abs(np.vdot(signal, signal).real / np.vdot(delta, delta).real - 1)
        maximum_roundtrip_error = max(maximum_roundtrip_error, float(roundtrip_error))
        maximum_energy_error = max(maximum_energy_error, float(energy_error))
        phi = np.column_stack([receive(np.roll(signal, delay)) for delay in delays])
        singular_values = np.linalg.svd(phi, compute_uv=False)
        rank = int(np.count_nonzero(
            singular_values > RELATIVE_SVD_THRESHOLD * singular_values[0]
        ))
        minimum_rank = min(minimum_rank, rank)
        minimum_ratio = min(minimum_ratio, float(singular_values[-1] / singular_values[0]))

    print(json.dumps({
        "N": length,
        "c1": c1,
        "c2": c2,
        "path_delays_samples": delays,
        "numpy_version": np.__version__,
        "seed": SEED,
        "case_count": sum(counts.values()),
        "case_counts": dict(counts),
        "relative_svd_threshold": RELATIVE_SVD_THRESHOLD,
        "minimum_observed_rank": minimum_rank,
        "target_column_rank": len(delays),
        "minimum_sigma_ratio": minimum_ratio,
        "maximum_roundtrip_relative_error": maximum_roundtrip_error,
        "maximum_energy_relative_error": maximum_energy_error,
        "scope": "Ideal cyclic block; zero residual common Doppler; unit-gain path "
                 "operators; sampled 16-QAM differences; not an exhaustive proof or BER run.",
    }, indent=2))
    if minimum_rank != len(delays):
        raise SystemExit("Sampled rank screen failed.")
    if max(maximum_roundtrip_error, maximum_energy_error) > 1e-12:
        raise SystemExit("Unitary transform screen failed.")


if __name__ == "__main__":
    main()
