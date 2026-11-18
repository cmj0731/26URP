"""Independent serial-waveform, CP, adjoint, pilot, NMSE and MMSE checks."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from ber_experiment import apply_paths, receive, transmit
from mmse_demapping import recover_data_bits
from mmse_problem import build_mmse_problem
from mmse_solver import solve_mmse_problem
from otfs_fractional_channel import FractionalChannelOperator, serial_channel, operator_inner, operator_nmse, exact_effective_gain
from otfs_fractional_estimation import ContinuousEstimator
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid
from qam16 import modulate
from run_ber_comparison import atomic_json


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(Path("data/ber_comparison_2026_11_04/multipath/scenario_used.json").read_text(encoding="utf-8"))
    n, blocks, cp, fs = 256, 64, 5, 60e6
    rng = np.random.default_rng(20261118)
    bits = rng.integers(0, 2, 49152, dtype=np.uint8)
    grid = build_grid("OTFS", modulate(bits), scenario=base)
    signal = transmit("OTFS", grid, base, cp)
    gains = np.sqrt(np.array([1., .1, 10**-1.5])/(1+.1+10**-1.5))*np.exp(1j*np.array([0., .7, 1.9]))
    estimator = ContinuousEstimator(base)
    summary = {"cp_samples": cp, "sample_rate_hz": fs, "integer_serial_relative_error": 0.,
               "serial_vs_useful_operator_relative_error": 0., "adjoint_relative_error": 0.,
               "pilot_template_relative_error": 0., "noiseless_recovery_errors": [], "clean_parameter_fit": []}
    for frequencies in ([0., 0., 0.], [0., 46875., -46875.]):
        actual = serial_channel(signal, [0., 2., 5.], gains, frequencies, fs, n, cp)
        previous = apply_paths(signal, [0, 2, 5], gains, frequencies, fs)
        summary["integer_serial_relative_error"] = max(summary["integer_serial_relative_error"], relative(previous, actual))
        for delta in (0., .1, .25, .5):
            paths = tuple(zip([0., 2.+delta, 5.], frequencies))
            channel = FractionalChannelOperator(n, blocks, cp, fs, paths, gains)
            output = serial_channel(signal, [p[0] for p in paths], gains, frequencies, fs, n, cp)
            native = receive("OTFS", output, base, cp)
            summary["serial_vs_useful_operator_relative_error"] = max(summary["serial_vs_useful_operator_relative_error"], relative(native, channel.apply(grid)))
            a = rng.standard_normal((n, blocks))+1j*rng.standard_normal((n, blocks))
            b = rng.standard_normal((n, blocks))+1j*rng.standard_normal((n, blocks))
            lhs, rhs = np.vdot(channel.apply(a), b), np.vdot(a, channel.adjoint(b))
            summary["adjoint_relative_error"] = max(summary["adjoint_relative_error"], float(abs(lhs-rhs)/max(abs(lhs), abs(rhs))))
            pilot = build_grid("OTFS", np.zeros(12288, complex), scenario=base)
            pilot_samples = transmit("OTFS", pilot, base, cp)
            pilot_output = serial_channel(pilot_samples, [p[0] for p in paths], gains, frequencies, fs, n, cp)
            observations = estimator.model.observations(pilot_output)
            template = estimator.model.atoms(paths)@gains
            summary["pilot_template_relative_error"] = max(summary["pilot_template_relative_error"], relative(observations, template))
            fit = estimator.fit_observations(observations)
            estimated = FractionalChannelOperator(n, blocks, cp, fs, fit["paths"], fit["gains"])
            summary["clean_parameter_fit"].append({"delta_samples": delta, "maximum_doppler_hz": max(abs(f) for f in frequencies),
                "paths": fit["paths"], "operator_nmse": operator_nmse(estimated, channel), "optimizer": fit["optimizer"]})
            # True channel is allowed ONLY in this labeled noiseless operator /
            # solver unit control. It is never used as an experiment receiver.
            model = build_pilot_data_model(channel, base)
            problem = build_mmse_problem(model, native, 0., base)
            solved = solve_mmse_problem(problem, base)
            solved.effective_gain = np.ones(problem.data_count)  # Full-column-rank dominance certificate below.
            assert abs(gains[0]) > sum(abs(gains[1:]))
            errors = int(np.count_nonzero(recover_data_bits(problem, solved) != bits))
            summary["noiseless_recovery_errors"].append(errors)
    # Dense tiny reference checks interpolation and normal self-gain, without
    # sharing the generator's FFT evaluation path.
    length, small_blocks = 16, 4
    tiny = FractionalChannelOperator(length, small_blocks, cp, fs, [(0., 0.), (2.35, 31000.)], [.9, .2j])
    basis = np.eye(length*small_blocks).reshape((length, small_blocks, -1), order="F")
    dense = np.column_stack([tiny.apply(basis[:, :, i]).reshape(-1, order="F") for i in range(basis.shape[2])])
    assert abs(operator_inner(tiny, tiny).real-np.linalg.norm(dense)**2) < 1e-10
    other = FractionalChannelOperator(length, small_blocks, cp, fs, [(0., 0.), (2.4, 30000.)], [.9, .19j])
    dense_other = np.column_stack([other.apply(basis[:, :, i]).reshape(-1, order="F") for i in range(basis.shape[2])])
    expected = np.linalg.norm(dense-dense_other)**2/np.linalg.norm(dense)**2
    assert abs(operator_nmse(other, tiny)-expected) < 1e-12
    mask = np.ones((length, small_blocks), dtype=bool)
    mask[4:8] = False
    selected = np.flatnonzero(mask.reshape(-1, order="F"))
    a = dense[:, selected]
    gram = a.conj().T@a
    lam = .0079
    expected_gain = np.diag(np.linalg.solve(gram+lam*np.eye(gram.shape[0]), gram)).real
    small_problem = SimpleNamespace(model=SimpleNamespace(channel=tiny, data_mask=mask, grid_shape=mask.shape), regularization=lam)
    gain_error = float(np.max(abs(exact_effective_gain(small_problem)-expected_gain)))
    assert gain_error < 1e-12
    summary["dense_full_DD_self_gain_max_absolute_error"] = gain_error
    # Analytic atom derivatives against centered differences, including the
    # normalized frequency coordinates used by the joint optimizer.
    trial = [(0.2, 100.), (2.2, 24000.), (4.8, -31000.)]
    matrix, derivative = estimator.model.atoms_with_derivatives(trial)
    duration = estimator.duration
    parameter = np.array([[d, f*duration] for d, f in trial]).reshape(-1)
    derivative_error = 0.
    for j in range(parameter.size):
        low, high = parameter.copy(), parameter.copy()
        low[j] -= 1e-5
        high[j] += 1e-5
        convert = lambda p: [(d, f/duration) for d, f in p.reshape(-1, 2)]
        difference = (estimator.model.atoms(convert(high))-estimator.model.atoms(convert(low)))/(2e-5)
        derivative_error = max(derivative_error, relative(derivative[:, j], difference[:, j//2]))
    assert derivative_error < 1e-7
    summary["analytic_atom_derivative_relative_error"] = derivative_error
    # Explicit finite Fourier sums at selected serial coordinates, including
    # the first useful sample and the inter-slot CP boundary.
    span = n+cp
    coefficients = np.fft.fft(signal.reshape(blocks, span)[:, cp:], axis=1, norm="ortho")
    delayed = serial_channel(signal, [2.35], [1.], [0.], fs, n, cp)
    tone_error = 0.
    for j in [0, 2, 3, cp, span, span+cp, span+17, signal.size-1]:
        source = j-2.35
        if source < 0: expected_sample = 0j
        else:
            block = int(np.floor(source/span))
            position = source-block*span-cp
            expected_sample = np.sum(coefficients[block]*np.exp(2j*np.pi*np.fft.fftfreq(n)*position))/np.sqrt(n)
        tone_error = max(tone_error, float(abs(delayed[j]-expected_sample)))
    summary["direct_Fourier_sum_max_absolute_error"] = tone_error
    # Change the entire preceding slot. With physical delays <=CP the next
    # useful slot must remain unchanged, although its early CP samples may
    # change. This checks CP coverage independently of the circular operator.
    changed = signal.copy()
    changed[:span] = 0.
    before = serial_channel(signal, [0., 2.5, 5.], gains, [0., 46875., -46875.], fs, n, cp)
    after = serial_channel(changed, [0., 2.5, 5.], gains, [0., 46875., -46875.], fs, n, cp)
    cp_coverage_error = float(np.max(abs(before[span+cp:2*span]-after[span+cp:2*span])))
    assert cp_coverage_error == 0.
    summary["preceding_slot_change_next_useful_max_absolute_error"] = cp_coverage_error
    for key in ("integer_serial_relative_error", "serial_vs_useful_operator_relative_error", "adjoint_relative_error", "pilot_template_relative_error"):
        assert summary[key] < 1e-10, (key, summary[key])
    assert tone_error < 1e-10 and not any(summary["noiseless_recovery_errors"])
    assert max(r["operator_nmse"] for r in summary["clean_parameter_fit"]) < 1e-7
    summary["complete"] = True
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
