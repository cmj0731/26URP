"""Fixed information-bit accounting and frame-cluster uncertainty reporting."""

import math
from statistics import NormalDist

import numpy as np


def frame_plan(total_bits, capacity):
    if not isinstance(total_bits, int) or not isinstance(capacity, int) or min(total_bits, capacity) <= 0:
        raise ValueError("Positive integer bit quota and frame capacity are required.")
    full, tail = divmod(total_bits, capacity)
    return [capacity]*full+([tail] if tail else [])


def summarize_frames(errors, bits, *, confidence=.95, full_frame_capacity):
    """Ratio BER and approximate CR1 frame-cluster normal confidence interval.

    Clusters are independent frame draws, not individual decoded bits. For zero
    observed errors, use complete frames' exact zero-event binomial upper bound
    on frame-error probability, which also bounds expected BER conservatively.
    """
    e, n = np.asarray(errors, dtype=float), np.asarray(bits, dtype=float)
    if (e.ndim != 1 or e.shape != n.shape or not e.size or np.any(n <= 0)
            or np.any(e < 0) or np.any(e > n) or not 0 < confidence < 1):
        raise ValueError("Valid per-frame error/bit counts and confidence are required.")
    total, count = int(n.sum()), int(e.sum())
    estimate = count/total
    complete_frames = int(np.count_nonzero(n == full_frame_capacity))
    result = {"measured_information_bits": total, "bit_errors": count, "ber_estimate": estimate,
              "frame_count": int(e.size), "error_frame_count": int(np.count_nonzero(e)),
              "confidence_level": confidence}
    if count == 0:
        upper = (-math.expm1(math.log((1-confidence)/2)/complete_frames)
                 if complete_frames else 1.)
        result.update(confidence_interval=[0., upper],
            confidence_method="conservative_complete_frame_zero_event_upper_bound",
            zero_error_label="no_errors_observed", frame_standard_error=None,
            interval_note="Upper bound is on the probability that a complete frame has any measured error; expected BER is no larger. Complete frames only, to exclude the partial tail. This deliberately allows arbitrary within-frame dependence and can be much wider than an iid-bit bound.")
    elif e.size < 2:
        result.update(confidence_interval=[0., 1.], confidence_method="insufficient_independent_frames",
                      frame_standard_error=None)
    else:
        residual = e-estimate*n
        se = float(np.sqrt(e.size/(e.size-1)*np.sum(residual**2))/total)
        z = NormalDist().inv_cdf((1+confidence)/2)
        result.update(confidence_interval=[max(0., estimate-z*se), min(1., estimate+z*se)],
            confidence_method="approximate_CR1_frame_cluster_normal",
            frame_standard_error=se,
            interval_note="Large-independent-frame approximation; arbitrary dependence is allowed within each frame. Sparse error-bearing frames can limit accuracy. Not an exact independent-bit binomial interval.")
    return result
