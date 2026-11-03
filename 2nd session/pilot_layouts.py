"""Agreed native masks and fixed, known pilot sequences.

Arrays use frequency/DAFT/delay rows and time-block/Doppler columns.
Symbol vectors fill their selected cells in column-major (Fortran) order.
Energy checks refer to the useful grid before cyclic prefixes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _configuration(scenario):
    if scenario is not None:
        return scenario
    return json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))


def build_masks(scheme, *, scenario=None):
    """Return disjoint boolean pilot, guard, reserved and data masks."""
    settings = _configuration(scenario)
    policy = settings["pilot_comparison_policy"]
    if scheme in ("OFDM", "DD-a-OFDM"):
        spec = policy["shared_ofdm_dd_a_tf_layout"]
        shape = tuple(spec["grid_shape_frequency_by_time"])
    elif scheme == "AFDM":
        spec = policy["afdm_daft_layout"]
        shape = tuple(spec["grid_shape"])
    elif scheme == "OTFS":
        spec = policy["otfs_dd_layout"]
        shape = tuple(spec["grid_shape"])
    else:
        raise ValueError(f"Unknown comparison scheme: {scheme!r}")
    pilot = np.zeros(shape, dtype=bool)
    guard = np.zeros(shape, dtype=bool)
    reserved = np.zeros(shape, dtype=bool)
    if scheme in ("OFDM", "DD-a-OFDM"):
        pilot[spec["frequency_offset_bins"]::spec["frequency_spacing_in_bins"],
              spec["time_offset_blocks"]::spec["time_spacing_in_blocks"]] = True
    elif scheme == "AFDM":
        pilot[spec["pilot_index_per_block"], :] = True
        for start, end in spec["zero_guard_index_ranges_inclusive"]:
            guard[start:end + 1, :] = True
    else:
        pilot[spec["pilot_delay_index"], spec["pilot_doppler_index"]] = True
        start, end = spec["reserved_region_delay_range_inclusive"]
        first, last = spec["reserved_region_doppler_range_inclusive"]
        reserved[start:end + 1, first:last + 1] = True
        start, end = spec["full_doppler_guard_delay_range_inclusive"]
        guard[start:end + 1, :] = True
        guard &= ~pilot
        reserved &= ~(pilot | guard)
    if np.any(pilot & guard) or np.any(pilot & reserved) or np.any(guard & reserved):
        raise ValueError("Pilot, guard and reserved cells must not overlap.")
    data = ~(pilot | guard | reserved)
    expected = policy["pilot_locations_and_counts"][scheme]
    if (int(pilot.sum()), int(guard.sum() + reserved.sum()), int(data.sum())) != (
        expected["active_pilot_count"],
        expected["pilot_specific_guard_or_reserved_count"],
        expected["data_cell_count"],
    ):
        raise ValueError("Native mask counts differ from the agreed resource settings.")
    return {"pilot": pilot, "guard": guard, "reserved": reserved, "data": data}


def known_pilot_symbols(scheme, *, scenario=None):
    """Return the selected fixed pilots without using data/channel RNG state.

    OFDM/DD-a-OFDM share the full QPSK phase sequence; AFDM reuses its first
    64 phases with amplitude eight. The single OTFS pilot is real and positive.
    """
    settings = _configuration(scenario)
    policy = settings["pilot_comparison_policy"]
    if scheme not in policy["pilot_locations_and_counts"]:
        raise ValueError(f"Unknown comparison scheme: {scheme!r}")
    count = policy["pilot_locations_and_counts"][scheme]["active_pilot_count"]
    spec = policy["known_pilot_sequence"]
    amplitude = np.sqrt(policy["total_pilot_energy"] / count)
    if scheme == "OTFS":
        return np.array([amplitude + 0j], dtype=np.complex128)
    generator = np.random.Generator(np.random.PCG64(spec["seed"]))
    codes = generator.integers(0, 4, size=spec["master_phase_count"], dtype=np.int64)
    digest = hashlib.sha256(codes.astype(np.uint8).tobytes()).hexdigest()
    if digest != spec["phase_index_sha256"]:
        raise ValueError("Fixed pilot phase sequence differs from the recorded sequence.")
    if count > codes.size:
        raise ValueError("The selected fixed phase sequence is too short.")
    alphabet = np.array([1, 1j, -1, -1j], dtype=np.complex128)
    return amplitude * alphabet[codes[:count]]


def build_grid(scheme, data_symbols, pilot_symbols=None, *, scenario=None):
    """Fill a native grid using the agreed fixed pilots by default.

    Both symbol vectors are one-dimensional. Explicit caller-supplied pilots
    must already meet the total energy budget; they are never rescaled.
    Guards and reservations remain zero. Symbols fill cells in Fortran order.
    """
    settings = _configuration(scenario)
    masks = build_masks(scheme, scenario=settings)
    data = np.asarray(data_symbols, dtype=np.complex128)
    if pilot_symbols is None:
        pilot_symbols = known_pilot_symbols(scheme, scenario=settings)
    pilots = np.asarray(pilot_symbols, dtype=np.complex128)
    for name, symbols, mask in (("data", data, masks["data"]),
                                ("pilot", pilots, masks["pilot"])):
        if symbols.ndim != 1 or symbols.size != int(mask.sum()):
            raise ValueError(f"Expected {int(mask.sum())} one-dimensional {name} symbols.")
        if not np.isfinite(symbols).all():
            raise ValueError(f"All {name} symbols must be finite.")
    if np.any(pilots == 0):
        raise ValueError("Active pilot cells require nonzero known symbols.")
    energy = float(np.vdot(pilots, pilots).real)
    target = settings["pilot_comparison_policy"]["total_pilot_energy"]
    if not np.isclose(energy, target, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Pilot energy must equal {target}; received {energy}.")
    shape = masks["data"].shape
    flat = np.zeros(int(np.prod(shape)), dtype=np.complex128)
    flat[masks["data"].reshape(-1, order="F")] = data
    flat[masks["pilot"].reshape(-1, order="F")] = pilots
    return flat.reshape(shape, order="F")
