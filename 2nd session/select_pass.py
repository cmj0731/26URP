"""Find the highest-elevation STARLINK-5285 pass in scenario.json."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from sgp4.api import SGP4_ERRORS, Satrec, jday
from sgp4.omm import initialize
from sgp4.propagation import gstime


ROOT = Path(__file__).resolve().parent
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
SPEED_OF_LIGHT_M_S = 299792458.0


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return result.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def station_ecef(station: dict) -> tuple[float, float, float]:
    lat = math.radians(station["latitude_deg"])
    lon = math.radians(station["longitude_deg"])
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    h = station["altitude_m"] / 1000.0
    return (
        (n + h) * math.cos(lat) * math.cos(lon),
        (n + h) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - e2) + h) * math.sin(lat),
    )


def geometry(satellite: Satrec, station: dict, location: tuple, time: datetime) -> tuple[float, float]:
    jd, fr = jday(
        time.year, time.month, time.day, time.hour, time.minute,
        time.second + time.microsecond / 1_000_000.0,
    )
    error, position, _velocity = satellite.sgp4(jd, fr)
    if error:
        raise RuntimeError(f"SGP4 error {error}: {SGP4_ERRORS.get(error, 'unknown')}")

    angle = gstime(jd + fr)  # UTC is used as a UT1 approximation, as in the source project.
    x = math.cos(angle) * position[0] + math.sin(angle) * position[1] - location[0]
    y = -math.sin(angle) * position[0] + math.cos(angle) * position[1] - location[1]
    z = position[2] - location[2]
    lat = math.radians(station["latitude_deg"])
    lon = math.radians(station["longitude_deg"])
    east = -math.sin(lon) * x + math.cos(lon) * y
    north = -math.sin(lat) * math.cos(lon) * x - math.sin(lat) * math.sin(lon) * y + math.cos(lat) * z
    up = math.cos(lat) * math.cos(lon) * x + math.cos(lat) * math.sin(lon) * y + math.sin(lat) * z
    return math.degrees(math.atan2(up, math.hypot(east, north))), math.sqrt(x * x + y * y + z * z)


def common_doppler_hz(geometry_at, time: datetime, carrier_frequency_hz: float) -> float:
    """Estimate raw common Doppler using range at t +/- 0.5 seconds."""
    half_span = timedelta(seconds=0.5)
    before_km = geometry_at(time - half_span)[1]
    after_km = geometry_at(time + half_span)[1]
    range_rate_m_s = (after_km - before_km) * 1000.0
    return -carrier_frequency_hz * range_rate_m_s / SPEED_OF_LIGHT_M_S


def common_doppler_envelope(
    geometry_at, start: datetime, end: datetime, carrier_frequency_hz: float,
    *, scan_step_s: float = 0.25,
) -> dict:
    """Scan a visible pass, including both endpoints; this is a numerical estimate."""
    if not math.isfinite(scan_step_s) or scan_step_s <= 0:
        raise ValueError("Doppler scan step must be finite and positive")
    step = timedelta(seconds=scan_step_s)
    if step <= timedelta(0) or end < start:
        raise ValueError("Doppler scan needs a representable step and an ordered time interval")
    times = [start]
    while times[-1] < end:
        times.append(min(times[-1] + step, end))
    shifts = [common_doppler_hz(geometry_at, time, carrier_frequency_hz) for time in times]
    maximum_index = max(range(len(shifts)), key=lambda index: abs(shifts[index]))
    return {
        "reference": "common_los_doppler_before_receiver_compensation",
        "start_utc": iso(start),
        "end_utc": iso(end),
        "scan_step_s": step.total_seconds(),
        "scan_time_count": len(times),
        "range_rate_central_difference_total_span_s": 1.0,
        "maximum_absolute_doppler_hz": round(abs(shifts[maximum_index]), 1),
        "signed_doppler_at_maximum_hz": round(shifts[maximum_index], 1),
        "maximum_time_utc": iso(times[maximum_index]),
        "note": "Numerical pass scan, not a measured signal or a certified continuous-time bound.",
    }


def bisect_crossing(elevation, left: datetime, right: datetime, threshold: float) -> datetime:
    left_above = elevation(left)[0] >= threshold
    for _ in range(45):
        middle = left + (right - left) / 2
        if (elevation(middle)[0] >= threshold) == left_above:
            left = middle
        else:
            right = middle
        if (right - left).total_seconds() < 0.001:
            break
    return left + (right - left) / 2


def extremum(elevation, start: datetime, end: datetime, *, index: int, maximize: bool) -> datetime:
    left, right = start, end
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(70):
        if (right - left).total_seconds() < 0.001:
            break
        first = right - (right - left) * ratio
        second = left + (right - left) * ratio
        first_value = elevation(first)[index]
        second_value = elevation(second)[index]
        if (first_value < second_value) == maximize:
            left = first
        else:
            right = second
    return left + (right - left) / 2


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doppler-envelope", action="store_true",
        help="Also report the raw common-Doppler envelope of the selected visible pass.",
    )
    parser.add_argument(
        "--doppler-step-s", type=float, default=0.25,
        help="Numerical Doppler scan interval in seconds (default: 0.25).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.doppler_step_s) or args.doppler_step_s <= 0:
        parser.error("--doppler-step-s must be finite and positive")
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    records = json.loads((ROOT / scenario["orbit_source"]["local_copy"]).read_text(encoding="utf-8"))
    record = records[0]
    if int(record["NORAD_CAT_ID"]) != scenario["satellite"]["norad_catalog_id"]:
        raise ValueError("Satellite ID does not match the orbit fixture")
    satellite = Satrec()
    initialize(satellite, {key: str(value) for key, value in record.items()})

    station = scenario["ground_station"]
    location = station_ecef(station)
    def elevation(time: datetime) -> tuple[float, float]:
        return geometry(satellite, station, location, time)

    search = scenario["pass_search"]
    start, end = utc(search["start_utc"]), utc(search["end_utc"])
    step = timedelta(seconds=search["coarse_step_s"])
    threshold = search["minimum_elevation_deg"]
    samples = [start]
    while samples[-1] < end:
        samples.append(min(samples[-1] + step, end))

    intervals = []
    visible_start = start if elevation(start)[0] >= threshold else None
    for left, right in zip(samples, samples[1:]):
        before = elevation(left)[0] >= threshold
        after = elevation(right)[0] >= threshold
        if not before and after:
            visible_start = bisect_crossing(elevation, left, right, threshold)
        elif before and not after:
            visible_end = bisect_crossing(elevation, left, right, threshold)
            intervals.append((visible_start, visible_end))
            visible_start = None
    if visible_start is not None:
        intervals.append((visible_start, end))
    if not intervals:
        raise RuntimeError("No passes met the elevation threshold")

    candidates = [
        (elevation(extremum(elevation, a, b, index=0, maximize=True))[0], a, b)
        for a, b in intervals
    ]
    _max_elevation, pass_start, pass_end = max(candidates, key=lambda item: item[0])
    peak_time = extremum(elevation, pass_start, pass_end, index=0, maximize=True)
    closest_time = extremum(elevation, pass_start, pass_end, index=1, maximize=False)
    peak_elevation, slant_range = elevation(peak_time)
    _closest_elevation, minimum_range = elevation(closest_time)
    result = {
        "visible_start_utc": iso(pass_start),
        "peak_utc": iso(peak_time),
        "closest_approach_utc": iso(closest_time),
        "visible_end_utc": iso(pass_end),
        "duration_s": round((pass_end - pass_start).total_seconds(), 3),
        "peak_elevation_deg": round(peak_elevation, 6),
        "slant_range_at_peak_km": round(slant_range, 3),
        "minimum_slant_range_km": round(minimum_range, 3),
        "visible_passes_in_search": len(intervals),
    }
    if args.doppler_envelope:
        envelope = common_doppler_envelope(
            elevation, utc(result["visible_start_utc"]), utc(result["visible_end_utc"]),
            scenario["radio"]["carrier_frequency_hz"], scan_step_s=args.doppler_step_s,
        )
        afdm_spacing_hz = (
            scenario["sampling_model"]["sample_rate_hz"]
            / scenario["waveform_design"]["AFDM"]["daft_length"]
        )
        envelope["afdm_native_doppler_bin_spacing_hz"] = afdm_spacing_hz
        envelope["maximum_absolute_doppler_in_afdm_bins"] = (
            envelope["maximum_absolute_doppler_hz"] / afdm_spacing_hz
        )
        result = {"selected_pass": result, "common_doppler_envelope": envelope}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
