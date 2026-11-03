"""Check OTFS axis/sign conventions and useful-signal energy; no BER channel."""

import cmath
import json
import math
from pathlib import Path

import numpy as np

from otfs import demodulate, isfft, modulate, sfft


ROOT = Path(__file__).resolve().parent
SEED = 20261103
RELATIVE_TOLERANCE = 1e-12


def direct_isfft(dd_grid):
    """Independent double-sum definition, with delay rows/Doppler columns."""
    delays, dopplers = dd_grid.shape
    tf_grid = np.zeros_like(dd_grid, dtype=complex)
    for m in range(delays):
        for n in range(dopplers):
            tf_grid[m, n] = sum(
                dd_grid[l, k] * cmath.exp(2j * math.pi * (n * k / dopplers - m * l / delays))
                for l in range(delays) for k in range(dopplers)
            ) / math.sqrt(delays * dopplers)
    return tf_grid


def direct_sfft(tf_grid):
    delays, dopplers = tf_grid.shape
    dd_grid = np.zeros_like(tf_grid, dtype=complex)
    for l in range(delays):
        for k in range(dopplers):
            dd_grid[l, k] = sum(
                tf_grid[m, n] * cmath.exp(-2j * math.pi * (n * k / dopplers - m * l / delays))
                for m in range(delays) for n in range(dopplers)
            ) / math.sqrt(delays * dopplers)
    return dd_grid


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main() -> None:
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    design = scenario["waveform_design"]["OTFS"]
    delays = design["delay_bin_count"]
    dopplers = design["doppler_bin_count"]
    if delays != scenario["radio"]["frequency_bin_count"]:
        raise ValueError("OTFS delay bins must match the common subcarrier count.")
    if dopplers != scenario["frame_design"]["time_symbol_count"]:
        raise ValueError("OTFS Doppler bins must match the common time-slot count.")
    if delays * dopplers != scenario["frame_design"]["useful_sample_count_without_prefix"]:
        raise ValueError("OTFS useful sample count differs from the common frame.")
    rng = np.random.default_rng(SEED)

    def qam_grid(shape):
        return (rng.choice([-3, -1, 1, 3], shape)
                + 1j * rng.choice([-3, -1, 1, 3], shape)) / math.sqrt(10)

    errors = {}
    for shape in [(3, 2), (4, 5)]:
        dd_grid = qam_grid(shape)
        tf_grid = direct_isfft(dd_grid)
        errors[f"isfft_{shape[0]}x{shape[1]}"] = relative_error(isfft(dd_grid), tf_grid)
        arbitrary_tf = qam_grid(shape)
        errors[f"sfft_{shape[0]}x{shape[1]}"] = relative_error(sfft(arbitrary_tf), direct_sfft(arbitrary_tf))
        expected_serial = np.array([
            sum(tf_grid[m, n] * cmath.exp(2j * math.pi * q * m / shape[0])
                for m in range(shape[0])) / math.sqrt(shape[0])
            for n in range(shape[1]) for q in range(shape[0])
        ])
        errors[f"serial_samples_{shape[0]}x{shape[1]}"] = relative_error(modulate(dd_grid), expected_serial)

    cases = [qam_grid((delays, dopplers)) for _ in range(8)]
    for l, k in [(0, 0), (min(1, delays - 1), min(1, dopplers - 1)),
                 (delays - 1, dopplers - 1), (delays // 2, dopplers // 2)]:
        impulse = np.zeros((delays, dopplers), dtype=complex)
        impulse[l, k] = 1
        cases.append(impulse)
    inverse_doppler_matrix = np.exp(
        2j * np.pi * np.outer(np.arange(dopplers), np.arange(dopplers)) / dopplers
    ) / math.sqrt(dopplers)
    maximum_roundtrip_error = 0.0
    maximum_energy_error = 0.0
    maximum_zak_equivalence_error = 0.0
    for dd_grid in cases:
        signal = modulate(dd_grid)
        recovered = demodulate(signal, delays, dopplers)
        maximum_roundtrip_error = max(maximum_roundtrip_error, relative_error(recovered, dd_grid))
        input_energy = float(np.vdot(dd_grid, dd_grid).real)
        time_energy = float(np.vdot(signal, signal).real)
        tf_energy = float(np.vdot(isfft(dd_grid), isfft(dd_grid)).real)
        maximum_energy_error = max(maximum_energy_error,
                                   abs(time_energy / input_energy - 1),
                                   abs(tf_energy / input_energy - 1))
        expected_by_slot = dd_grid @ inverse_doppler_matrix
        expected_serial = np.concatenate([expected_by_slot[:, n] for n in range(dopplers)])
        maximum_zak_equivalence_error = max(maximum_zak_equivalence_error,
                                            relative_error(signal, expected_serial))

    print(json.dumps({
        "dd_shape_delay_by_doppler": [delays, dopplers],
        "useful_samples_per_frame": delays * dopplers,
        "seed": SEED,
        "numpy_version": np.__version__,
        "direct_sum_relative_errors": errors,
        "full_grid_case_count": len(cases),
        "full_grid_cases": "8 random normalized 16-QAM grids and 4 DD impulses",
        "maximum_roundtrip_relative_error": maximum_roundtrip_error,
        "maximum_energy_relative_error": maximum_energy_error,
        "maximum_zak_equivalence_relative_error": maximum_zak_equivalence_error,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "scope": "Noiseless useful-signal transform checks only; no channel, prefixes, "
                 "pilots, synchronization, equalization, detection, or BER measurement.",
    }, indent=2))
    if max([*errors.values(), maximum_roundtrip_error, maximum_energy_error,
            maximum_zak_equivalence_error]) > RELATIVE_TOLERANCE:
        raise SystemExit("OTFS transform check failed.")


if __name__ == "__main__":
    main()
