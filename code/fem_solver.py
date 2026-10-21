"""Reusable P1 finite-element solver for the microstrip electrostatic model."""

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


def assemble_stiffness(geom: GeometryData, mat_cfg: MaterialConfig = MaterialConfig()) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    points = geom.points
    elements = geom.elements
    eps = permittivity(geom.element_regions, mat_cfg)
    eps_r = relative_permittivity(geom.element_regions, mat_cfg)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    areas = np.empty(elements.shape[0], dtype=float)
    grad_basis = np.empty((elements.shape[0], 3, 2), dtype=float)

    for e, tri in enumerate(elements):
        xy = points[tri]
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]
        signed_area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        area = abs(signed_area)
        if area <= 0.0:
            raise ValueError("non-positive element area")
        grads = np.array(
            [
                [y1 - y2, x2 - x1],
                [y2 - y0, x0 - x2],
                [y0 - y1, x1 - x0],
            ],
            dtype=float,
        ) / (2.0 * signed_area)
        local = eps[e] * area * (grads @ grads.T)
        areas[e] = area
        grad_basis[e] = grads
        for a in range(3):
            for b in range(3):
                rows.append(int(tri[a]))
                cols.append(int(tri[b]))
                data.append(float(local[a, b]))

    matrix = coo_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0])).tocsr()
    return matrix, areas, grad_basis, eps, eps_r, time.perf_counter() - start


def solve_microstrip(
    geom: GeometryData | None = None,
    mat_cfg: MaterialConfig = MaterialConfig(),
    compute_conditioning: bool = True,
) -> MicrostripSolution:
    geom = geom or build_baseline_geometry()
    matrix, areas, grad_basis, eps, eps_r, assembly_time = assemble_stiffness(geom, mat_cfg)
    n = geom.points.shape[0]
    trace = np.array(sorted(set(geom.trace_nodes.tolist())), dtype=int)
    ground = np.array(sorted(set(geom.ground_nodes.tolist())), dtype=int)
    overlap = np.intersect1d(trace, ground)
    if overlap.size:
        raise ValueError("trace and ground Dirichlet dofs overlap")
    dirichlet = np.array(sorted(set(trace.tolist()) | set(ground.tolist())), dtype=int)
    free = np.setdiff1d(np.arange(n), dirichlet)
    values = np.zeros(n, dtype=float)
    values[trace] = geom.config.V0

    rhs_free = -matrix[free][:, dirichlet] @ values[dirichlet]
    matrix_free = matrix[free][:, free].tocsr()
    start = time.perf_counter()
    values[free] = spsolve(matrix_free, rhs_free)
    solve_time = time.perf_counter() - start
    reaction = matrix @ values
    element_gradients = np.einsum("eik,ei->ek", grad_basis, values[geom.elements])
    if compute_conditioning:
        lambda_min = float(eigsh(matrix_free, k=1, which="SA", return_eigenvectors=False, tol=1e-9)[0])
        lambda_max = float(eigsh(matrix_free, k=1, which="LA", return_eigenvectors=False, tol=1e-9)[0])
        condition_estimate = float(lambda_max / lambda_min)
        diagonal = matrix_free.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("reduced stiffness matrix has non-positive diagonal")
        inv_sqrt_d = diags(1.0 / np.sqrt(diagonal))
        scaled_matrix = (inv_sqrt_d @ matrix_free @ inv_sqrt_d).tocsr()
        scaled_lambda_min = float(eigsh(scaled_matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-9)[0])
        scaled_lambda_max = float(eigsh(scaled_matrix, k=1, which="LA", return_eigenvectors=False, tol=1e-9)[0])
        scaled_condition_estimate = float(scaled_lambda_max / scaled_lambda_min)
    else:
        diagonal = matrix_free.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("reduced stiffness matrix has non-positive diagonal")
        lambda_min = float("nan")
        lambda_max = float("nan")
        condition_estimate = float("nan")
        scaled_lambda_min = float("nan")
        scaled_lambda_max = float("nan")
        scaled_condition_estimate = float("nan")
    return MicrostripSolution(
        geometry=geom,
        phi=values,
        matrix_full=matrix,
        matrix_free=matrix_free,
        rhs_free=rhs_free,
        free_dofs=free,
        dirichlet_dofs=dirichlet,
        trace_dofs=trace,
        ground_dofs=ground,
        reaction_full=reaction,
        element_gradients=element_gradients,
        element_areas=areas,
        epsilon=eps,
        epsilon_r=eps_r,
        assembly_time=assembly_time,
        solve_time=solve_time,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_estimate=condition_estimate,
        scaled_lambda_min=scaled_lambda_min,
        scaled_lambda_max=scaled_lambda_max,
        scaled_condition_estimate=scaled_condition_estimate,
    )
