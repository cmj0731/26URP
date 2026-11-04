"""NumPy-only DPSS subspace with an independent prolate-operator check."""

import numpy as np


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
