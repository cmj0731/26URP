"""Numerical solves for the selected estimated-channel linear-MMSE problem.

Local TF/DAFT systems use direct solves. OTFS retains the full implicit frame:
CG for positive measured variance, zero-start LSQR for undamped least squares.
Only receiver inputs already held by PlugInMMSEProblem are consumed.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MMSESolveResult:
    symbols: np.ndarray
    converged: bool
    method: str
    iterations: int
    relative_normal_residual: float
    normal_residual_norm: float
    rhs_norm: float
    reason: str
    diagnostics: dict = field(default_factory=dict)

    def require_converged(self):
        """Prevent a failed numerical solve from entering later bit scoring."""
        if not self.converged:
            raise RuntimeError(f"MMSE solve failed: {self.method}: {self.reason}")
        return self.symbols


def _relative(residual_norm, rhs_norm):
    if rhs_norm == 0:
        return 0.0 if residual_norm == 0 else float("inf")
    return float(residual_norm/rhs_norm)


def _finish(problem, values, method, iterations, tolerance, reason, diagnostics=None):
    values = np.asarray(values, dtype=complex)
    rhs_norm = float(np.linalg.norm(problem.rhs))
    finite = bool(np.isfinite(values).all())
    norm = float(np.linalg.norm(problem.rhs-problem.normal(values))) if finite else float("inf")
    relative = _relative(norm, rhs_norm)
    converged = finite and relative <= tolerance and reason in ("converged", "zero_rhs")
    if reason == "converged" and not converged:
        reason = "true_residual_above_tolerance"
    return MMSESolveResult(values, converged, method, iterations, relative, norm,
                           rhs_norm, reason, diagnostics or {})


def _direct_blocks(problem, tolerance):
    values = np.zeros(problem.data_count, dtype=complex)
    blocks = []
    for block in range(problem.grid_shape[1]):
        selected = problem.model.block_data_slice(block)
        try:
            if problem.regularization > 0:
                normal, rhs = problem.block_system(block)
                estimate = np.linalg.solve(normal, rhs)
                rank = None
            else:
                matrix = problem.model.block_matrix(block)
                estimate, _, rank, _ = np.linalg.lstsq(matrix, problem.observations[:, block], rcond=None)
                normal, rhs = problem.block_system(block)
                rank = int(rank)
        except np.linalg.LinAlgError as error:
            return _finish(problem, values, "block_direct", 0, tolerance, "factorization_failed",
                           {"failed_block": block, "error": str(error), "blocks": blocks})
        values[selected] = estimate
        relative = _relative(float(np.linalg.norm(rhs-normal@estimate)), float(np.linalg.norm(rhs)))
        blocks.append({"block": block, "data_unknowns": selected.stop-selected.start,
                       "relative_normal_residual": relative, "least_squares_rank": rank})
    maximum = max(row["relative_normal_residual"] for row in blocks)
    reason = "converged" if maximum <= tolerance else "block_residual_above_tolerance"
    method = "block_direct" if problem.regularization > 0 else "block_SVD_least_squares"
    return _finish(problem, values, method, 0, tolerance, reason,
                   {"block_count": len(blocks), "max_block_relative_normal_residual": maximum,
                    "blocks": blocks})


def _cg(problem, tolerance, maximum_iterations):
    values = np.zeros(problem.data_count, dtype=complex)
    rhs_norm = float(np.linalg.norm(problem.rhs))
    if rhs_norm == 0:
        return _finish(problem, values, "full_frame_CG", 0, tolerance, "zero_rhs")
    residual = problem.rhs.copy()
    direction = residual.copy()
    squared = float(np.vdot(residual, residual).real)
    for iteration in range(1, maximum_iterations+1):
        action = problem.normal(direction)
        curvature = float(np.vdot(direction, action).real)
        if not np.isfinite(curvature) or curvature <= 0:
            return _finish(problem, values, "full_frame_CG", iteration-1, tolerance,
                           "nonpositive_or_nonfinite_curvature")
        step = squared/curvature
        values += step*direction
        residual -= step*action
        if not np.isfinite(values).all():
            return _finish(problem, values, "full_frame_CG", iteration, tolerance, "nonfinite_iterate")
        # Accept only a freshly computed residual of the original equation.
        true_residual = problem.rhs-problem.normal(values)
        true_norm = float(np.linalg.norm(true_residual))
        if true_norm <= tolerance*rhs_norm:
            return _finish(problem, values, "full_frame_CG", iteration, tolerance, "converged")
        updated = float(np.vdot(residual, residual).real)
        if not np.isfinite(updated):
            return _finish(problem, values, "full_frame_CG", iteration, tolerance, "nonfinite_residual")
        if updated <= (tolerance*rhs_norm)**2:
            # Roundoff can make recursive and true residuals disagree. Restart
            # from the true residual without altering the MMSE equation.
            residual = true_residual
            direction = residual.copy()
            squared = float(np.vdot(residual, residual).real)
        else:
            direction = residual+(updated/squared)*direction
            squared = updated
    return _finish(problem, values, "full_frame_CG", maximum_iterations, tolerance,
                   "maximum_iterations_reached")


def _lsqr(problem, tolerance, maximum_iterations):
    """Complex Golub-Kahan LSQR, zero start and no damping or dense DD matrix."""
    values = np.zeros(problem.data_count, dtype=complex)
    rhs_norm = float(np.linalg.norm(problem.rhs))
    if rhs_norm == 0:
        return _finish(problem, values, "full_frame_LSQR", 0, tolerance, "zero_rhs")
    u = problem.observations.copy()
    beta = float(np.linalg.norm(u))
    u /= beta
    v = problem.model.adjoint(u)
    alpha = float(np.linalg.norm(v))
    if alpha == 0:
        return _finish(problem, values, "full_frame_LSQR", 0, tolerance, "bidiagonalization_breakdown")
    v /= alpha
    direction = v.copy()
    rho_bar, phi_bar = alpha, beta
    for iteration in range(1, maximum_iterations+1):
        u = problem.model.apply(v)-alpha*u
        beta = float(np.linalg.norm(u))
        if beta > 0:
            u /= beta
        v = problem.model.adjoint(u)-beta*v
        alpha = float(np.linalg.norm(v))
        if alpha > 0:
            v /= alpha
        rho = float(np.hypot(rho_bar, beta))
        if not np.isfinite(rho) or rho == 0:
            return _finish(problem, values, "full_frame_LSQR", iteration-1, tolerance,
                           "bidiagonalization_breakdown")
        cosine, sine = rho_bar/rho, beta/rho
        theta = sine*alpha
        rho_bar = -cosine*alpha
        phi, phi_bar = cosine*phi_bar, sine*phi_bar
        values += (phi/rho)*direction
        direction = v-(theta/rho)*direction
        if not np.isfinite(values).all():
            return _finish(problem, values, "full_frame_LSQR", iteration, tolerance, "nonfinite_iterate")
        true_norm = float(np.linalg.norm(problem.rhs-problem.normal(values)))
        if true_norm <= tolerance*rhs_norm:
            return _finish(problem, values, "full_frame_LSQR", iteration, tolerance, "converged")
        if alpha == 0 and beta == 0:
            return _finish(problem, values, "full_frame_LSQR", iteration, tolerance,
                           "bidiagonalization_breakdown")
    return _finish(problem, values, "full_frame_LSQR", maximum_iterations, tolerance,
                   "maximum_iterations_reached")


def solve_mmse_problem(problem, settings):
    """Solve with the agreed fixed public policy; return explicit status/accuracy.

    Output is the raw linear estimate in data-cell order. No output gain
    adjustment, constellation decision, channel update or BER count is made.
    A nonconverged result must not be scored or silently dropped from a BER run.
    """
    policy = settings["receiver_equalization_policy"]["numerical_solver"]
    tolerance = policy["relative_normal_residual_tolerance"]
    maximum = policy["maximum_iterations"]
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or not np.isrealobj(tolerance) or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("A finite positive relative tolerance smaller than one is required.")
    if (not isinstance(maximum, (int, np.integer)) or isinstance(maximum, (bool, np.bool_)) or maximum < 1):
        raise ValueError("A positive integer iteration cap is required.")
    if policy["preconditioner"] != "none" or policy["initial_iterate"] != "zero":
        raise ValueError("The selected zero-start unpreconditioned solver policy is required.")
    if problem.scheme == "OTFS":
        if (policy["positive_noise_methods"]["OTFS"] != "full_frame_CG"
                or policy["zero_noise_methods"]["OTFS"] != "full_frame_LSQR"):
            raise ValueError("The selected full-frame CG/undamped LSQR methods are required.")
        return (_cg(problem, tolerance, maximum) if problem.regularization > 0
                else _lsqr(problem, tolerance, maximum))
    if (policy["positive_noise_methods"][problem.scheme] != "block_direct"
            or policy["zero_noise_methods"][problem.scheme] != "block_SVD_least_squares"):
        raise ValueError("The selected local direct/least-squares methods are required.")
    return _direct_blocks(problem, tolerance)
