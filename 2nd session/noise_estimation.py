"""Noise-power measurement from an explicitly scheduled silent receive interval.

The agreed noise model has known zero mean. Only actual calibration observations
enter these functions, never the generated noise from the data interval or Eb/N0.
Measurement counts come explicitly from the selected public design, not defaults.
"""

from dataclasses import dataclass

import numpy as np


def _received_vector(samples):
    values = np.asarray(samples, dtype=complex)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("One nonempty finite received-sample vector is required.")
    return values


def estimate_noise_power(noise_only_samples):
    """Return mean(|z|^2), the complex variance for zero-mean noise observations.

    Signal absence must be ensured by the measurement schedule and channel
    support. This function does not detect silence or remove signal leakage.
    There is no mean subtraction, L-1 correction or positive variance floor.
    """
    values = _received_vector(noise_only_samples)
    power = float(np.vdot(values, values).real / values.size)
    if not np.isfinite(power):
        raise ValueError("Received-sample energy must be finite.")
    return power


def estimate_noise_power_from_silence(received_interval, *, settling_sample_count,
                                    measurement_sample_count):
    """Discard the specified settling prefix and measure the next samples.

    Counts must come from the selected public schedule, not hidden generator
    parameters. Any extra trailing observations are left out of the measurement.
    Neither input observations nor a following data interval are modified.
    """
    for value, minimum in ((settling_sample_count, 0), (measurement_sample_count, 1)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < minimum):
            raise ValueError("Explicit nonnegative settling and positive measurement counts are required.")
    values = _received_vector(received_interval)
    end = settling_sample_count + measurement_sample_count
    if values.size < end:
        raise ValueError("The received interval is shorter than its settling and measurement schedule.")
    return estimate_noise_power(values[settling_sample_count:end])


@dataclass(frozen=True)
class SilentNoiseMeasurement:
    """Public once-per-frame silent measurement, separate from the data frame."""

    settling_sample_count: int
    measurement_sample_count: int
    sample_rate_hz: float

    def __post_init__(self):
        for value, minimum in ((self.settling_sample_count, 0),
                               (self.measurement_sample_count, 1)):
            if (not isinstance(value, (int, np.integer))
                    or isinstance(value, (bool, np.bool_)) or value < minimum):
                raise ValueError("Valid explicit silent-measurement counts are required.")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("A finite positive public sample rate is required.")

    @property
    def interval_sample_count(self):
        return self.settling_sample_count + self.measurement_sample_count

    @property
    def interval_duration_s(self):
        return self.interval_sample_count / self.sample_rate_hz

    def estimate(self, received_silent_interval):
        """Estimate from exactly this calibration interval, without data samples."""
        values = _received_vector(received_silent_interval)
        if values.size != self.interval_sample_count:
            raise ValueError("Supply exactly the scheduled silent interval, without data samples.")
        return estimate_noise_power_from_silence(values,
            settling_sample_count=self.settling_sample_count,
            measurement_sample_count=self.measurement_sample_count)


def build_silent_noise_measurement(settings):
    """Read only public noise-measurement design, delay support and sample rate."""
    policy = settings["receiver_equalization_policy"]["noise_information_policy"]
    if (policy["method"] != "zero_mean_sample_energy"
            or policy["update_interval"] != "once_per_frame"
            or policy["observation_resource"]["type"] != "common_transmit_silence_before_each_frame"):
        raise ValueError("The selected once-per-frame silent energy measurement is required.")
    settling = policy["observation_resource"]["settling_sample_count"]
    bound = settings["receiver_information_policy"]["delay_support"]["maximum_excess_delay_samples"]
    if settling != bound:
        raise ValueError("Review the silent settling design when the public delay support changes.")
    return SilentNoiseMeasurement(settling, policy["sample_count"],
        settings["sampling_model"]["sample_rate_hz"])
