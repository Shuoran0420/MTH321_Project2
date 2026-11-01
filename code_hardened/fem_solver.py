"""P1 FEM solver for the microstrip electrostatic boundary-value problem.

Weak form: find phi=V0 on the trace and phi=0 on the ground such that
    integral_Omega epsilon grad(phi) . grad(v) dOmega = 0
for all test functions v vanishing on the Dirichlet boundary.
Zero flux on the other exterior boundaries is the natural FEM condition.

This is a drop-in replacement for code/fem_solver.py (original public API kept).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve

from geometry import GeometryData, build_baseline_geometry
from materials import MaterialConfig, permittivity, relative_permittivity


@dataclass
class MicrostripSolution:
    geometry: GeometryData
    phi: np.ndarray
    matrix_full: csr_matrix
    matrix_free: csr_matrix
    rhs_free: np.ndarray
    free_dofs: np.ndarray
    dirichlet_dofs: np.ndarray
    trace_dofs: np.ndarray
    ground_dofs: np.ndarray
    reaction_full: np.ndarray
    element_gradients: np.ndarray
    element_areas: np.ndarray
    epsilon: np.ndarray
    epsilon_r: np.ndarray
    assembly_time: float
    solve_time: float
    lambda_min: float
    lambda_max: float
    condition_estimate: float
    scaled_lambda_min: float
    scaled_lambda_max: float
    scaled_condition_estimate: float


def assemble_stiffness(
    geom: GeometryData, mat_cfg: MaterialConfig = MaterialConfig()
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble K_ij = sum_T int_T epsilon grad(N_i).grad(N_j) dA.

    Element vertices are indexed as geom.elements[e, local_node]; global node
    order is the order in geom.points. P1 gradients are constant on each T.
    All arrays are vectorised, but assembly stays sparse (COO -> CSR).
    """
    start = time.perf_counter()
    points = np.asarray(geom.points, dtype=float)
    triangles = np.asarray(geom.elements, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("geom.points must have shape (number_of_nodes, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError("geom.elements must have shape (number_of_triangles, 3)")
    if np.any(~np.isfinite(points)):
        raise ValueError("mesh coordinates must be finite")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("triangle references an invalid node")

    xy = points[triangles]
    x0, y0 = xy[:, 0, 0], xy[:, 0, 1]
    x1, y1 = xy[:, 1, 0], xy[:, 1, 1]
    x2, y2 = xy[:, 2, 0], xy[:, 2, 1]
    twice_signed_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(~np.isfinite(twice_signed_area)) or np.any(twice_signed_area == 0):
        raise ValueError("mesh contains a degenerate or non-finite triangle")
    areas = 0.5 * np.abs(twice_signed_area)
    gradients = np.stack(
        (np.stack((y1 - y2, x2 - x1), axis=1),
         np.stack((y2 - y0, x0 - x2), axis=1),
         np.stack((y0 - y1, x1 - x0), axis=1)), axis=1
    ) / twice_signed_area[:, None, None]

    eps = permittivity(geom.element_regions, mat_cfg)
    eps_r = relative_permittivity(geom.element_regions, mat_cfg)
    if eps.shape != (len(triangles),):
        raise ValueError("one permittivity value is required per triangle")

    local = (eps * areas)[:, None, None] * np.einsum(
        "eik,ejk->eij", gradients, gradients, optimize=True
    )
    row = np.broadcast_to(triangles[:, :, None], local.shape).ravel()
    col = np.broadcast_to(triangles[:, None, :], local.shape).ravel()
    K = coo_matrix((local.ravel(), (row, col)), shape=(len(points), len(points))).tocsr()
    K.sum_duplicates()
    return K, areas, gradients, eps, eps_r, time.perf_counter() - start


def _eigen_extremes(K: csr_matrix) -> tuple[float, float, float]:
    if K.shape[0] == 1:
        val = float(K[0, 0])
        if val <= 0:
            raise ValueError("matrix is not positive definite")
        return val, val, 1.0
    lo = float(eigsh(K, k=1, which="SA", return_eigenvectors=False, tol=1e-9)[0])
    hi = float(eigsh(K, k=1, which="LA", return_eigenvectors=False, tol=1e-9)[0])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0):
        raise ValueError("matrix conditioning audit found nonpositive eigenvalue")
    return lo, hi, hi / lo


def solve_microstrip(
    geom: GeometryData | None = None,
    mat_cfg: MaterialConfig = MaterialConfig(),
    compute_conditioning: bool = True,
    *,
    residual_tolerance: float = 1e-8,
) -> MicrostripSolution:
    """Solve the reduced SPD system and check its scaled algebraic residual.

    residual_tolerance is dimensionless: ||K_ff*u_f-b_f||_inf /
    max(||K_ff*u_f||_inf, ||b_f||_inf, tiny). It measures the
    algebraic solve only; it does NOT certify spatial discretisation error.
    """
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    geom = geom if geom is not None else build_baseline_geometry()
    K, areas, grad_basis, eps, eps_r, assembly_time = assemble_stiffness(geom, mat_cfg)
    n = len(geom.points)
    trace = np.unique(np.asarray(geom.trace_nodes, dtype=int))
    ground = np.unique(np.asarray(geom.ground_nodes, dtype=int))
    if not len(trace) or not len(ground):
        raise ValueError("trace and ground must each contain at least one node")
    if np.any(trace < 0) or np.any(trace >= n) or np.any(ground < 0) or np.any(ground >= n):
        raise ValueError("invalid Dirichlet node index")
    if np.intersect1d(trace, ground).size:
        raise ValueError("trace and ground Dirichlet dofs overlap")
    if not np.isfinite(geom.config.V0) or geom.config.V0 == 0:
        raise ValueError("V0 must be finite and nonzero for capacitance extraction")
    fixed = np.union1d(trace, ground)
    free = np.setdiff1d(np.arange(n, dtype=int), fixed, assume_unique=True)
    if not len(free):
        raise ValueError("mesh contains no free degrees of freedom")
    values = np.zeros(n, dtype=float)
    values[trace] = geom.config.V0
    K_ff = K[free][:, free].tocsr()
    rhs = np.asarray(-K[free][:, fixed] @ values[fixed]).ravel()
    diagonal = K_ff.diagonal()
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError("reduced stiffness matrix has non-positive diagonal")

    start = time.perf_counter()
    values[free] = spsolve(K_ff, rhs)
    solve_time = time.perf_counter() - start
    if not np.all(np.isfinite(values)):
        raise RuntimeError("linear solver returned non-finite electrostatic potential")
    lhs = K_ff @ values[free]
    r = lhs - rhs
    scale = max(float(np.linalg.norm(lhs, ord=np.inf)),
                float(np.linalg.norm(rhs, ord=np.inf)), np.finfo(float).tiny)
    scaled_residual = float(np.linalg.norm(r, ord=np.inf) / scale)
    if not np.isfinite(scaled_residual) or scaled_residual > residual_tolerance:
        raise RuntimeError(f"linear solve failed scaled residual check: {scaled_residual:.3e}")

    reaction = np.asarray(K @ values).ravel()
    element_gradients = np.einsum("eik,ei->ek", grad_basis, values[geom.elements])
    if compute_conditioning:
        lambda_min, lambda_max, condition_estimate = _eigen_extremes(K_ff)
        D = diags(1.0 / np.sqrt(diagonal))
        K_scaled = (D @ K_ff @ D).tocsr()
        scaled_lambda_min, scaled_lambda_max, scaled_condition_estimate = _eigen_extremes(K_scaled)
    else:
        lambda_min = lambda_max = condition_estimate = float("nan")
        scaled_lambda_min = scaled_lambda_max = scaled_condition_estimate = float("nan")
    return MicrostripSolution(
        geometry=geom, phi=values, matrix_full=K, matrix_free=K_ff,
        rhs_free=rhs, free_dofs=free, dirichlet_dofs=fixed,
        trace_dofs=trace, ground_dofs=ground, reaction_full=reaction,
        element_gradients=element_gradients, element_areas=areas,
        epsilon=eps, epsilon_r=eps_r, assembly_time=assembly_time,
        solve_time=solve_time, lambda_min=lambda_min, lambda_max=lambda_max,
        condition_estimate=condition_estimate,
        scaled_lambda_min=scaled_lambda_min, scaled_lambda_max=scaled_lambda_max,
        scaled_condition_estimate=scaled_condition_estimate,
    )
