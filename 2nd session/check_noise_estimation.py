"""Diagnose the shared silent-interval estimator and compare measurement lengths.

Generator variance and generated data-interval noise appear only in this scoring
script. The estimator receives a separate calibration interval and public counts.
No channel estimator, MMSE solver, physical frame or BER budget is changed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from noise_estimation import (build_silent_noise_measurement, estimate_noise_power,
                              estimate_noise_power_from_silence)


def rejected(call):
    try:
        call()
    except ValueError:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    public = {key: settings[key] for key in ("sampling_model", "receiver_information_policy",
                                           "receiver_equalization_policy")}
    design = build_silent_noise_measurement(public)
    fs = settings["sampling_model"]["sample_rate_hz"]
    frame_count = settings["prefix_policy"]["transmitted_samples_per_frame"]
    # The settling bound is public; active generator path positions are not read.
    settling = settings["receiver_information_policy"]["delay_support"]["maximum_excess_delay_samples"]
    candidates = (256, 512, 1024)
    trials = 8192
    seeds = np.random.SeedSequence(20261104).spawn(3)
    calibration_rng, data_noise_rng, control_rng = [
        np.random.Generator(np.random.PCG64(seed)) for seed in seeds]

    exact = np.array([1+2j, -3+4j, 0j, 2-1j])
    expected = 35/4
    assert estimate_noise_power(exact) == expected
    # A complex variance counts both quadratures, not half the returned power.
    assert estimate_noise_power(2j*exact) == 4*expected
    assert estimate_noise_power(np.zeros(10, dtype=complex)) == 0.0
    invalid = [lambda: estimate_noise_power([]),
               lambda: estimate_noise_power(np.zeros((2, 2))),
               lambda: estimate_noise_power([np.nan]),
               lambda: estimate_noise_power([np.inf]),
               lambda: estimate_noise_power_from_silence(exact, settling_sample_count=-1,
                                                        measurement_sample_count=2),
               lambda: estimate_noise_power_from_silence(exact, settling_sample_count=True,
                                                        measurement_sample_count=2),
               lambda: estimate_noise_power_from_silence(exact, settling_sample_count=0,
                                                        measurement_sample_count=0),
               lambda: estimate_noise_power_from_silence(exact, settling_sample_count=1,
                                                        measurement_sample_count=4)]
    assert all(rejected(call) for call in invalid)

    # Independent serial test: every allowed integer tap may be active and time
    # varying. A previous arbitrary transmission is followed by exact Tx zeros.
    previous_count = 256
    previous = (control_rng.standard_normal(previous_count)
                + 1j*control_rng.standard_normal(previous_count))/np.sqrt(2)
    tx = np.r_[previous, np.zeros(settling+max(candidates), dtype=complex)]
    rx_signal = np.zeros_like(tx)
    times = np.arange(tx.size)/fs
    for delay in range(settling+1):
        frequency = control_rng.uniform(-46875, 46875)
        tap = (control_rng.standard_normal()+1j*control_rng.standard_normal())/np.sqrt(2)
        delayed = np.zeros_like(tx)
        delayed[delay:] = tx[:tx.size-delay]
        rx_signal += tap*np.exp(2j*np.pi*frequency*times)*delayed
    signal_during_silence = rx_signal[previous_count:]
    assert np.linalg.norm(signal_during_silence[:settling]) > 0
    assert np.all(signal_during_silence[settling:] == 0)
    calibration_noise = (calibration_rng.standard_normal(signal_during_silence.size)
                         + 1j*calibration_rng.standard_normal(signal_during_silence.size))/np.sqrt(2)
    calibration_observations = signal_during_silence+calibration_noise
    original = calibration_observations.copy()
    actual = estimate_noise_power_from_silence(calibration_observations,
        settling_sample_count=settling, measurement_sample_count=max(candidates))
    expected_observed = estimate_noise_power(calibration_noise[settling:])
    assert actual == expected_observed
    assert np.array_equal(calibration_observations, original)
    scheduled = original[:design.interval_sample_count]
    scheduled_power = design.estimate(scheduled)
    assert scheduled_power == estimate_noise_power(calibration_noise[settling:design.interval_sample_count])
    assert design.estimate(np.zeros(design.interval_sample_count, dtype=complex)) == 0.0
    assert rejected(lambda: design.estimate(scheduled[:-1]))
    changed_tail = original.copy()
    changed_tail[:settling] += 1e6+2e6j
    assert estimate_noise_power_from_silence(changed_tail, settling_sample_count=settling,
        measurement_sample_count=max(candidates)) == actual
    assert estimate_noise_power_from_silence(np.r_[original, [1e9]], settling_sample_count=settling,
        measurement_sample_count=max(candidates)) == actual
    # Each branch uses a physically different noise interval; no data-interval
    # noise vector is passed to the measurement estimator.
    data_interval_noise = (data_noise_rng.standard_normal(frame_count)
                          + 1j*data_noise_rng.standard_normal(frame_count))/np.sqrt(2)
    assert not np.array_equal(calibration_noise[settling:], data_interval_noise[:max(candidates)])
    assert rejected(lambda: design.estimate(np.r_[scheduled, data_interval_noise]))

    scale_error = 0.0
    for eb_n0_db in range(0, 31, 5):
        variance = 1/(4*10**(eb_n0_db/10))
        measured = estimate_noise_power_from_silence(np.sqrt(variance)*original,
            settling_sample_count=settling, measurement_sample_count=max(candidates))
        scale_error = max(scale_error, abs(measured-variance*actual)/(variance*actual))
    assert scale_error < 1e-12

    powers = {count: [] for count in candidates}
    for _ in range(trials//256):
        unit_noise = (calibration_rng.standard_normal((256, max(candidates)))
                      + 1j*calibration_rng.standard_normal((256, max(candidates))))/np.sqrt(2)
        for count in candidates:
            powers[count].extend(estimate_noise_power(row[:count]) for row in unit_noise)
    records = []
    for count in candidates:
        values = np.asarray(powers[count])
        theoretical_relative_std = 1/np.sqrt(count)
        mean_ratio = float(values.mean())
        empirical_relative_std = float(values.std(ddof=1))
        # Finite seeded validation against the zero-mean complex Gaussian model.
        assert abs(mean_ratio-1) < 5*theoretical_relative_std/np.sqrt(trials)
        assert abs(empirical_relative_std/theoretical_relative_std-1) < 0.05
        records.append({
            "measurement_sample_count": count,
            "selected": count == design.measurement_sample_count,
            "diagnostic_trials": trials,
            "mean_estimate_over_true_variance": mean_ratio,
            "empirical_std_over_true_variance": empirical_relative_std,
            "theoretical_std_over_true_variance": float(theoretical_relative_std),
            "measurement_duration_s": count/fs,
            "settling_sample_count_derived_from_public_delay_support": settling,
            "total_added_samples_if_measured_once_per_frame": count+settling,
            "total_added_duration_s_if_measured_once_per_frame": (count+settling)/fs,
            "added_time_over_existing_frame_percent": 100*(count+settling)/frame_count,
            "added_time_fraction_of_extended_interval_percent": 100*(count+settling)/(frame_count+count+settling)
        })
    report = {
        "scope": "Shared scheduled-silence noise-power measurement diagnostic, not MMSE or BER.",
        "decision_status": "The user selected 256 measured samples after 5 settling samples, once per frame for all four receivers; 512/1024 remain unselected sensitivity alternatives.",
        "implementation": "noise_estimation.py",
        "receiver_inputs": "The builder reads only the public noise measurement design, sample rate and delay bound. Estimation receives exactly the separate scheduled silent-interval observations. No target Eb/N0, generator variance, channel truth, true data or data-interval noise samples.",
        "selected_design": {"measurement_sample_count": design.measurement_sample_count,
                            "settling_sample_count": design.settling_sample_count,
                            "interval_sample_count": design.interval_sample_count,
                            "interval_duration_s": design.interval_duration_s,
                            "update_interval": "once_per_frame"},
        "estimator": "mean(abs(z)**2), with known zero-mean complex receiver noise; no L-1 correction or regularization floor.",
        "theory": "For L independent CN(0,sigma^2) samples, E[sigma_hat^2]=sigma^2 and Var(sigma_hat^2)=sigma^4/L. Relative standard deviation is 1/sqrt(L). This is not a bound on each estimate or on BER.",
        "settling_control": "An independent prior arbitrary waveform and all allowed 0..5 time-varying integer taps leave no signal after the five-sample silent settling prefix. This is a mathematical support control, not a new physical path profile.",
        "maximum_awgn_grid_scaling_relative_error": scale_error,
        "invalid_input_checks": len(invalid)+2,
        "zero_measurement_result": 0.0,
        "candidate_lengths": records,
        "limits": "Signal absence and stationary noise power over calibration/data are assumptions. Extra settling is needed if delay support, filtering or timing uncertainty changes. The selected calibration adds 261 samples before the existing 16704-sample frame; no original pilot budget, channel estimator or BER bit quota is changed. Generator and MMSE integration is pending. Zero/small-variance handling in the eventual equalizer is still pending; measurement uncertainty is not a BER accuracy guarantee.",
        "reference_url": "https://cris.unibo.it/retrieve/handle/11585/802184/78cc746a-e562-4d62-ba22-f30493ca800a/TWC2982880.pdf",
        "reference_scope": "Equation (18) gives the squared-magnitude sample-mean estimator for noise-only measurements in spectrum sensing, not an actual Starlink silence schedule. Variance and resource calculations here follow our zero-mean independent complex AWGN model."
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
