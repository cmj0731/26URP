"""Deterministic local-zero-region interventions with a fixed OTFS budget."""

import copy
import hashlib
import numpy as np

from pilot_layouts import build_masks

WIDTHS = (64, 11, 9, 5, 1)


def build_designs(public):
    """Keep one pilot, 4095 zero cells and 12288 data cells in each design.

    Released local rows receive the same data that occupied the newly reserved
    distant rows. Common data cells keep their exact symbols. The mapping is
    public framing information, not receiver knowledge of data values.
    """
    original = build_masks("OTFS", scenario=public)
    original_positions = np.flatnonzero(original["data"].reshape(-1, order="F"))
    lookup = np.full(original["data"].size, -1, dtype=int)
    lookup[original_positions] = np.arange(original_positions.size)
    n, columns = original["data"].shape
    layout = public["pilot_comparison_policy"]["otfs_dd_layout"]
    center = layout["pilot_delay_index"]
    assert (n, columns, center, layout["pilot_doppler_index"]) == (256, 64, 128, 32)
    support = public["receiver_information_policy"]["delay_support"]
    delays = tuple(range(support["minimum_excess_delay_samples"], support["maximum_excess_delay_samples"]+1))
    observation_rows = center+np.array(delays)
    influence_rows = sorted({int((row-delay) % n) for row in observation_rows for delay in delays})
    result = []
    for width in WIDTHS:
        settings = copy.deepcopy(public)
        policy = settings["pilot_comparison_policy"]
        spec = policy["otfs_dd_layout"]
        if width == 64:
            local = [96, 159]
            distant = None
        else:
            half = width//2
            local = [center-half, center+half]
            distant = [0, 64-width-1]
            spec["full_doppler_guard_delay_range_inclusive"] = local
            spec["reserved_region_delay_range_inclusive"] = distant
            spec["reserved_region_doppler_range_inclusive"] = [0, columns-1]
            guard_count, reserved_count = width*columns-1, (64-width)*columns
            spec["pilot_specific_guard_count_per_frame"] = guard_count
            spec["additional_reserved_count_per_frame"] = reserved_count
            counts = policy["pilot_locations_and_counts"]["OTFS"]
            counts["pilot_specific_guard_count"] = guard_count
            counts["additional_reserved_count"] = reserved_count
            spec["placement_note"] = "Guard sensitivity intervention: local full-Doppler zero rows and distant reserved rows. Fixed original pilot-only estimator; data interference is retained."
        masks = build_masks("OTFS", scenario=settings)
        assert (int(masks["pilot"].sum()), int((masks["guard"] | masks["reserved"]).sum()), int(masks["data"].sum())) == (1, 4095, 12288)
        assert np.array_equal(masks["pilot"], original["pilot"])
        assert np.array_equal(masks["data"], np.broadcast_to(masks["data"][:, :1], masks["data"].shape))
        positions = np.flatnonzero(masks["data"].reshape(-1, order="F"))
        removed = np.flatnonzero((original["data"] & ~masks["data"]).reshape(-1, order="F"))
        order = lookup[positions].copy()
        added = order < 0
        assert int(added.sum()) == removed.size
        order[added] = lookup[removed]
        assert np.array_equal(np.sort(order), np.arange(original_positions.size))
        common = lookup[positions] >= 0
        assert np.array_equal(order[common], lookup[positions[common]])
        tail = (positions % n) >= n-public["prefix_policy"]["prefix_samples_per_block"]
        assert np.array_equal(order[tail], lookup[positions[tail]])
        isolated = not any(np.any(masks["data"][(observation_rows-delay) % n]) for delay in delays)
        assert isolated == (width in (64, 11))
        if distant is not None:
            assert not set(range(distant[0], distant[1]+1)).intersection(influence_rows)
        descriptor = {"label": f"W{width}", "local_zero_delay_row_count": width,
            "local_zero_delay_range_inclusive": local,
            "local_zero_cells_excluding_pilot": width*columns-1,
            "distant_reserved_delay_range_inclusive": distant,
            "distant_reserved_cells": (64-width)*columns,
            "pilot_count": 1, "all_zero_cells": 4095, "data_count": 12288,
            "Doppler_columns_per_zero_row": columns,
            "pilot_observation_data_isolated_by_public_support": isolated,
            "pilot_observation_receive_delay_rows": observation_rows.tolist(),
            "possible_input_delay_rows_for_pilot_observation": influence_rows,
            "symbol_order_sha256_int64_little_endian": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
            "symbol_order_rule": "Keep common data cells' original symbol IDs. Move original data from newly reserved distant cells to released local cells, pairing removed and added cells in column-major order. Decode back to the original information-bit order."}
        policy["guard_sensitivity_scope"] = descriptor["label"]+": receiver channel estimation is frozen at the original pilot-only model; actual mask violations are measured, not cancelled."
        result.append({"settings": settings, "masks": masks, "order": order, "descriptor": descriptor})
    return result


def original_bit_order(native_bits, order):
    """Undo the public symbol-placement permutation without using data truth."""
    values = np.asarray(native_bits)
    if values.ndim != 1 or values.size != 4*order.size:
        raise ValueError("A complete native-order 16-QAM bit vector is required.")
    result = np.empty_like(values)
    result.reshape(-1, 4)[order] = values.reshape(-1, 4)
    return result
