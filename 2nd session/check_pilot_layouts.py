"""Check selected pilots, native masks, transform energy and the OTFS guard.

Use the agreed fixed QPSK sequences and uniform energy allocation.
There is no channel estimation, equalization, AWGN or BER calculation.
"""

import json
from pathlib import Path

import numpy as np

from cyclic_prefix import add_cyclic_prefix
import otfs
from pilot_layouts import build_grid, build_masks, known_pilot_symbols


def main():
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261103)
    tolerance = 1e-12
    schemes = ["OFDM", "DD-a-OFDM", "AFDM", "OTFS"]
    expected = {
        "OFDM": (4096, 0, 0, 12288),
        "DD-a-OFDM": (4096, 0, 0, 12288),
        "AFDM": (64, 4032, 0, 12288),
        "OTFS": (1, 703, 3392, 12288),
    }
    masks = {scheme: build_masks(scheme, scenario=settings) for scheme in schemes}
    for name in masks["OFDM"]:
        np.testing.assert_array_equal(masks["OFDM"][name], masks["DD-a-OFDM"][name])
    pilots_by_scheme = {scheme: known_pilot_symbols(scheme, scenario=settings) for scheme in schemes}
    np.testing.assert_array_equal(pilots_by_scheme["OFDM"], pilots_by_scheme["DD-a-OFDM"])
    np.testing.assert_array_equal(pilots_by_scheme["AFDM"] / 8, pilots_by_scheme["OFDM"][:64])
    np.testing.assert_array_equal(pilots_by_scheme["OTFS"], [64 + 0j])
    # Calls must be fixed across trials and independent of the data stream.
    state_before = rng.bit_generator.state
    for scheme in schemes:
        np.testing.assert_array_equal(pilots_by_scheme[scheme], known_pilot_symbols(scheme, scenario=settings))
    assert rng.bit_generator.state == state_before
    data = (rng.choice([-3, -1, 1, 3], 12288)
            + 1j * rng.choice([-3, -1, 1, 3], 12288)) / np.sqrt(10)
    n = 256
    index = np.arange(n, dtype=float)
    chirps = settings["waveform_design"]["AFDM"]["chirp_parameters"]
    chirp1 = np.exp(2j * np.pi * chirps["c1"] * index**2)[:, None]
    chirp2 = np.exp(2j * np.pi * chirps["c2"] * index**2)[:, None]

    def transmit(scheme, grid):
        if scheme == "OTFS":
            return otfs.modulate(grid)
        if scheme == "AFDM":
            useful = chirp1 * np.fft.ifft(chirp2 * grid, axis=0, norm="ortho")
        else:
            useful = np.fft.ifft(grid, axis=0, norm="ortho")
        return useful.reshape(-1, order="F")

    def receive(scheme, useful):
        if scheme == "OTFS":
            return otfs.demodulate(useful, 256, 64)
        samples = useful.reshape((256, 64), order="F")
        if scheme == "AFDM":
            return chirp2.conj() * np.fft.fft(chirp1.conj() * samples, axis=0, norm="ortho")
        return np.fft.fft(samples, axis=0, norm="ortho")

    report = {"seed": 20261103, "relative_tolerance": tolerance, "schemes": {}}
    grids = {}
    for scheme in schemes:
        layout = masks[scheme]
        counts = tuple(int(layout[k].sum()) for k in ("pilot", "guard", "reserved", "data"))
        assert counts == expected[scheme]
        assert np.all(np.stack(list(layout.values())).sum(axis=0) == 1)
        pilots = pilots_by_scheme[scheme]
        np.testing.assert_allclose(np.abs(pilots)**2, 4096 / counts[0], rtol=0, atol=0)
        grid = build_grid(scheme, data, scenario=settings)
        np.testing.assert_array_equal(grid, build_grid(scheme, data, pilots, scenario=settings))
        grids[scheme] = grid
        assert np.all(grid[layout["guard"] | layout["reserved"]] == 0)
        np.testing.assert_array_equal(grid.reshape(-1, order="F")[layout["data"].reshape(-1, order="F")], data)
        pilot_grid = np.where(layout["pilot"], grid, 0)
        useful = transmit(scheme, grid)
        pilot_useful = transmit(scheme, pilot_grid)
        grid_energy = float(np.vdot(grid, grid).real)
        useful_energy = float(np.vdot(useful, useful).real)
        pilot_energy = float(np.vdot(pilot_useful, pilot_useful).real)
        energy_error = abs(useful_energy - grid_energy) / grid_energy
        pilot_energy_error = abs(pilot_energy - 4096) / 4096
        roundtrip_error = float(np.linalg.norm(receive(scheme, useful) - grid) / np.linalg.norm(grid))
        assert max(energy_error, pilot_energy_error, roundtrip_error) < tolerance
        prefixed = add_cyclic_prefix(pilot_useful, 256, 5)
        prefix_only = prefixed.reshape((261, 64), order="F")[:5, :]
        report["schemes"][scheme] = {
            "pilot_guard_reserved_data_counts": counts,
            "pilot_energy_per_active_cell": float(4096 / counts[0]),
            "pilot_useful_signal_energy": pilot_energy,
            "pilot_energy_relative_error": pilot_energy_error,
            "full_grid_energy_relative_error": energy_error,
            "roundtrip_relative_error": roundtrip_error,
            "diagnostic_pilot_CP_energy_separately_measured": float(np.vdot(prefix_only, prefix_only).real),
        }
        try:
            build_grid(scheme, data, pilots * 2, scenario=settings)
        except ValueError:
            pass
        else:
            raise AssertionError("A pilot-energy mismatch must be rejected, not rescaled.")
    np.testing.assert_array_equal(grids["OFDM"], grids["DD-a-OFDM"])

    # Exact current OTFS transforms and CP, with data only. Truth is used only
    # to construct diagnostic paths, never passed to a receiver estimator.
    dd_data = np.where(masks["OTFS"]["data"], grids["OTFS"], 0)
    signal = otfs.modulate(dd_data, cp_length=5)
    time = np.arange(signal.size) / settings["sampling_model"]["sample_rate_hz"]
    worst = 0.0
    levels = settings["channel_model"]["multipath_followup"]["doppler_effect_followup"]["sweep_magnitude_design"]["maximum_residual_doppler_levels_hz"]
    for magnitude in levels:
        for delay, doppler in zip([0, 2, 5], [0, magnitude, -magnitude]):
            delayed = np.zeros_like(signal)
            if delay:
                delayed[delay:] = signal[:-delay]
            else:
                delayed[:] = signal
            received = delayed * np.exp(2j * np.pi * doppler * time)
            observation = otfs.demodulate(received, 256, 64, cp_length=5)
            worst = max(worst, float(np.linalg.norm(observation[128:134, :]) / np.linalg.norm(dd_data)))
    assert worst < tolerance
    report["otfs_integer_delay_data_leakage_relative_norm"] = worst
    report["fixed_phase_index_sha256"] = settings["pilot_comparison_policy"]["known_pilot_sequence"]["phase_index_sha256"]
    report["scope"] = "Agreed fixed pilots: repeatability, phase reuse, uniform energy, native masks, useful-signal transform energy and roundtrip checks, plus an OTFS data-only guard diagnostic. CP figures refer to the isolated pilot component and remain separate from the common useful-signal energy budget. No channel estimation or BER; AFDM fractional-Doppler estimation leakage remains to be validated."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
