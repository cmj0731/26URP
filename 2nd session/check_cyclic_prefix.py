"""Check common CP timing, energy and linear-convolution block boundaries.

Generator path truth is used only to construct this diagnostic channel and its
independent cyclic-block reference. This is not a channel estimator or BER run.
"""

import json
import math
from pathlib import Path

import numpy as np

from cyclic_prefix import add_cyclic_prefix, remove_cyclic_prefix
from otfs import demodulate, modulate


ROOT = Path(__file__).resolve().parent
SEED = 20261103
TOLERANCE = 1e-12


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main() -> None:
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    prefix = scenario["prefix_policy"]
    length = prefix["useful_samples_per_block"]
    cp_length = prefix["prefix_samples_per_block"]
    count = scenario["frame_design"]["time_symbol_count"]
    rate = scenario["sampling_model"]["sample_rate_hz"]
    profile = scenario["channel_model"]["multipath_followup"]["path_delay_power_doppler_profile"]
    if rate != 60_000_000:
        raise ValueError("Review the stored integer-delay grid before changing the rate.")
    delays = [path["excess_delay_samples_at_60mhz"] for path in profile]
    if cp_length < max(delays):
        raise ValueError("Selected CP does not cover the stored integer-delay profile.")
    rng = np.random.default_rng(SEED)
    useful = (rng.standard_normal(length * count)
              + 1j * rng.standard_normal(length * count)) / math.sqrt(2)
    transmitted = add_cyclic_prefix(useful, length, cp_length)
    restored = remove_cyclic_prefix(transmitted, length, cp_length)
    roundtrip_error = relative_error(restored, useful)
    zero_cp_error = relative_error(
        remove_cyclic_prefix(add_cyclic_prefix(useful, length, 0), length, 0), useful
    )
    copied = useful.reshape(count, length)[:, length - cp_length:]
    useful_energy = float(np.vdot(useful, useful).real)
    cp_energy = float(np.vdot(copied, copied).real)
    transmitted_energy = float(np.vdot(transmitted, transmitted).real)
    energy_error = abs(transmitted_energy - useful_energy - cp_energy) / transmitted_energy

    powers = np.array([10 ** (path["relative_power_db_to_los"] / 10) for path in profile])
    phases = np.concatenate(([0.0], rng.uniform(0, 2 * np.pi, len(profile) - 1)))
    gains = np.sqrt(powers / powers.sum()) * np.exp(1j * phases)
    impulse_response = np.zeros(max(delays) + 1, dtype=complex)
    for delay, gain in zip(delays, gains):
        impulse_response[delay] += gain
    received = np.convolve(transmitted, impulse_response)[:transmitted.size]
    useful_received = remove_cyclic_prefix(received, length, cp_length)
    blocks = useful.reshape(count, length)
    expected = sum(gain * np.roll(blocks, delay, axis=1)
                   for delay, gain in zip(delays, gains)).reshape(-1)
    convolution_error = relative_error(useful_received, expected)

    short_cp = max(delays) - 1
    short_transmitted = add_cyclic_prefix(useful, length, short_cp)
    short_received = np.convolve(short_transmitted, impulse_response)[:short_transmitted.size]
    short_cp_error = relative_error(remove_cyclic_prefix(short_received, length, short_cp), expected)

    afdm = scenario["waveform_design"]["AFDM"]
    c1 = afdm["chirp_parameters"]["c1"]
    twice_n_c1 = 2 * length * c1
    if afdm["daft_length"] != length or length % 2 or not np.isclose(twice_n_c1, round(twice_n_c1)):
        raise ValueError("The selected AFDM CPP does not reduce to this ordinary CP.")
    negative_indices = np.arange(-cp_length, 0)
    cpp_phase_turns = c1 * (length**2 + 2 * length * negative_indices)
    cpp_factors = np.exp(-2j * np.pi * np.remainder(cpp_phase_turns, 1))
    cpp_equivalence_error = float(np.max(np.abs(cpp_factors - 1)))

    dd_grid = (rng.choice([-3,-1,1,3], (length,count))
               + 1j * rng.choice([-3,-1,1,3], (length,count))) / math.sqrt(10)
    otfs_transmitted = modulate(dd_grid, cp_length=cp_length)
    otfs_recovered = demodulate(otfs_transmitted, length, count, cp_length=cp_length)
    otfs_error = relative_error(otfs_recovered, dd_grid)
    errors = [roundtrip_error, zero_cp_error, energy_error, convolution_error,
              cpp_equivalence_error, otfs_error]
    print(json.dumps({
        "seed": SEED,
        "useful_samples_per_block": length,
        "cp_samples_per_block": cp_length,
        "block_count": count,
        "transmitted_sample_count": int(transmitted.size),
        "frame_duration_s": transmitted.size / rate,
        "nominal_doppler_bin_spacing_hz": rate / transmitted.size,
        "path_delays_samples": delays,
        "cp_roundtrip_relative_error": roundtrip_error,
        "zero_cp_relative_error": zero_cp_error,
        "cp_energy_accounting_relative_error": energy_error,
        "linear_vs_cyclic_channel_relative_error": convolution_error,
        "short_cp_samples": short_cp,
        "short_cp_negative_control_relative_error": short_cp_error,
        "afdm_twice_n_c1": twice_n_c1,
        "afdm_cpp_equivalence_error": cpp_equivalence_error,
        "otfs_prefixed_roundtrip_relative_error": otfs_error,
        "relative_tolerance": TOLERANCE,
        "scope": "Aligned integer-delay, static three-path diagnostic without noise; "
                 "no estimator, channel equalizer, multipath Doppler policy, or BER experiment.",
    }, indent=2))
    if max(errors) > TOLERANCE or short_cp_error <= 1e-6:
        raise SystemExit("CP diagnostic failed, including the short-CP negative control.")


if __name__ == "__main__":
    main()
