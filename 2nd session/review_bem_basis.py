"""Review BEM basis candidates without selecting a receiver configuration.

Compare finite-window trajectory representation and the exact TF response of
the known pilot component. Unknown-data ICI, channel estimation and BER are
outside this diagnostic. Function counts below are probes, not selected values.
Only NumPy is required. DPSS are computed by spectral quadrature and checked
against the exact prolate Toeplitz eigenproblem using an FFT product.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pilot_layouts import build_grid, build_masks


def dpss_subspace(length, half_bandwidth, count):
    nodes, weights = np.polynomial.legendre.leggauss(128)
    frequencies = half_bandwidth * nodes
    scale = np.sqrt(half_bandwidth * weights)
    difference = frequencies[None, :] - frequencies[:, None]
    gram = (scale[:, None] * scale[None, :] * length
            * np.sinc(length * difference) / np.sinc(difference)
            * np.exp(1j * np.pi * (length - 1) * difference))
    eigenvalues, vectors = np.linalg.eigh(gram)
    selected = np.argsort(eigenvalues)[::-1][:count]
    eigenvalues = eigenvalues[selected]
    features = (np.exp(2j * np.pi * np.arange(length)[:, None]
                       * frequencies[None, :]) * scale[None, :])
    basis = (features @ vectors[:, selected]) / np.sqrt(eigenvalues)[None, :]
    first_column = 2 * half_bandwidth * np.sinc(2 * half_bandwidth * np.arange(length))
    embedding = np.r_[first_column, first_column[-2:0:-1]]
    product = np.fft.ifft(
        np.fft.fft(embedding)[:, None]
        * np.fft.fft(basis, n=embedding.size, axis=0), axis=0)[:length]
    residual = float(np.linalg.norm(product - basis * eigenvalues[None, :])
                     / np.linalg.norm(basis * eigenvalues[None, :]))
    orthogonality = float(np.linalg.norm(basis.conj().T @ basis - np.eye(count), 2))
    if residual >= 1e-9 or orthogonality >= 1e-8:
        raise ValueError("DPSS quadrature failed the independent prolate-operator check.")
    return basis, {"quadrature_points": 128,
                   "relative_eigen_residual": residual,
                   "orthogonality_error_2norm": orthogonality,
                   "smallest_retained_eigenvalue": float(eigenvalues[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    prefix = settings["prefix_policy"]
    fs = settings["sampling_model"]["sample_rate_hz"]
    useful_length = prefix["useful_samples_per_block"]
    cp = prefix["prefix_samples_per_block"]
    block_length = useful_length + cp
    length = prefix["transmitted_samples_per_frame"]
    blocks = length // block_length
    maximum = settings["channel_model"]["multipath_followup"]["doppler_effect_followup"]["sweep_magnitude_design"]["maximum_residual_doppler_hz"]
    largest_count = 31
    dpss, dpss_check = dpss_subspace(length, maximum / fs, largest_count)
    sample = np.arange(length)
    ce = np.exp(2j * np.pi * sample[:, None] * np.arange(-15, 16)[None, :] / length) / np.sqrt(length)
    polynomials = np.polynomial.legendre.legvander(np.linspace(-1, 1, length), largest_count - 1)
    polynomials = np.linalg.qr(polynomials, mode="reduced")[0]
    useful_indices = (np.arange(blocks)[:, None] * block_length + cp
                      + np.arange(useful_length)[None, :]).reshape(-1)
    times = useful_indices / fs
    probe_frequencies = np.linspace(-maximum, maximum, 401)
    pilot_mask = build_masks("OFDM", scenario=settings)["pilot"]
    pilot_grid = build_grid("OFDM", np.zeros(int((~pilot_mask).sum()), dtype=complex), scenario=settings)
    pilot_time = np.fft.ifft(pilot_grid, axis=0, norm="ortho")
    pilot_serial = np.vstack([pilot_time[-cp:, :], pilot_time]).reshape(-1, order="F")
    # A diagnostic span from zero through CP, not the true three path locations.
    delays = list(range(cp + 1))
    report = {
        "purpose": "Candidate basis review; no basis or function count is selected.",
        "numpy_version": np.__version__,
        "window_blocks": blocks,
        "transmitted_sample_count": length,
        "useful_sample_count": int(useful_indices.size),
        "sample_rate_hz": fs,
        "maximum_design_residual_doppler_hz": maximum,
        "time_bandwidth_product_one_sided": maximum * length / fs,
        "probe_frequency_count": int(probe_frequencies.size),
        "probe_frequency_range_hz": [-maximum, maximum],
        "diagnostic_delay_span_samples": delays,
        "delay_span_status": "Diagnostic assumption from the current CP length; not a newly selected receiver search prior or supplied true path positions.",
        "dpss_numerical_verification": dpss_check,
        "known_pilot_observation_rows": int(pilot_mask.sum()),
        "rank_relative_tolerance": 1e-10,
        "results": [],
        "limits": [
            "Trajectory tests use unit-amplitude complex exponentials as probes, not BER or measured Starlink channels.",
            "Representation errors are best-fit errors using the diagnostic trajectory itself, not coefficients recovered from pilot observations. The true trajectories are never supplied to a production receiver.",
            "All bases are fixed from the design envelope, without using actual per-path Dopplers.",
            "The pilot-component matrix includes the current native transforms, fixed pilots, integer delays and CP clock. It omits unknown-data ICI and observation noise.",
            "Full rank does not establish accurate noisy pilot-based channel estimation; noise gain is for an ideal white-observation-noise model only.",
            "Diagnostic counts 3/5/27/29/31 are not adopted BEM parameters.",
        ],
    }
    families = [("Legendre_polynomial", polynomials, [3, 5, 27, 29, 31]),
                ("DFT_complex_exponential", ce, [27, 29, 31]),
                ("DPSS_Slepian", dpss, [27, 29, 31])]
    for family, full_basis, counts in families:
        pilot_parts = []
        for delay in delays:
            shifted = np.zeros_like(pilot_serial)
            if delay:
                shifted[delay:] = pilot_serial[:-delay]
            else:
                shifted[:] = pilot_serial
            values = (shifted[:, None] * full_basis)[useful_indices, :]
            values = values.reshape((blocks, useful_length, largest_count)).transpose(1, 0, 2)
            pilot_parts.append(np.fft.fft(values, axis=0, norm="ortho")[pilot_mask, :])
        for count in counts:
            first = (largest_count - count) // 2 if family == "DFT_complex_exponential" else 0
            indices = np.arange(first, first + count)
            basis = full_basis[:, indices]
            useful_basis = basis[useful_indices, :]
            orthogonal = np.linalg.qr(useful_basis, mode="reduced")[0]
            worst_error = -1.0
            worst_frequency = None
            squared_errors = []
            for start in range(0, probe_frequencies.size, 32):
                frequencies = probe_frequencies[start:start + 32]
                truth = np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
                fitted = orthogonal @ (orthogonal.conj().T @ truth)
                relative_squared = np.sum(np.abs(truth - fitted)**2, axis=0) / times.size
                squared_errors.extend(relative_squared.tolist())
                peak = int(np.argmax(relative_squared))
                if relative_squared[peak] > worst_error:
                    worst_error = float(relative_squared[peak])
                    worst_frequency = float(frequencies[peak])
            observation_matrix = np.concatenate([part[:, indices] for part in pilot_parts], axis=1)
            _, singular, vh = np.linalg.svd(observation_matrix, full_matrices=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
            inverse_gram = (vh.conj().T * (1 / singular**2)[None, :]) @ vh
            useful_gram = useful_basis.conj().T @ useful_basis
            noise_gain = sum(float(np.trace(useful_gram @ inverse_gram[k*count:(k+1)*count, k*count:(k+1)*count]).real)
                             for k in range(len(delays))) / times.size
            report["results"].append({
                "basis_family": family, "basis_function_count": count,
                "worst_trajectory_relative_L2_error": float(np.sqrt(worst_error)),
                "worst_trajectory_probe_frequency_hz": worst_frequency,
                "mean_trajectory_relative_squared_error": float(np.mean(squared_errors)),
                "known_pilot_matrix_shape": list(observation_matrix.shape),
                "known_pilot_matrix_rank": rank,
                "known_pilot_matrix_condition_2norm": float(singular[0] / singular[-1]),
                "ideal_white_noise_channel_MSE_gain": noise_gain,
            })
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Review report saved to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
