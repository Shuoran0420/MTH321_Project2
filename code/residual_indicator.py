"""Physical-flux edge residual indicator and deterministic Doerfler marking."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Indicator:
    element_squared: np.ndarray
    edge_jump: np.ndarray
    interior: np.ndarray
    far: np.ndarray
    dirichlet: np.ndarray

    @property
    def eta(self):
        return float(np.sqrt(self.element_squared.sum()))


def flux_indicator(geom, gradients, epsilon):
    """Exact edge integrals for constant P1 gradients; no volume term (f=0)."""
    mesh = geom.mesh
    edges = mesh.facets.T
    adjacent = mesh.f2t.T
    a, b = adjacent[:, 0], adjacent[:, 1]
    tangent = geom.points[edges[:, 1]] - geom.points[edges[:, 0]]
    length = np.linalg.norm(tangent, axis=1)
    # The same normal is used on both sides; its sign is immaterial after squaring.
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / length[:, None]
    flux = epsilon[:, None] * gradients
    interior = b >= 0
    jump = np.einsum("ij,ij->i", flux[a], normal)
    jump[interior] -= np.einsum("ij,ij->i", flux[b[interior]], normal[interior])
    midpoint = geom.points[edges].mean(axis=1)
    c = geom.config
    far = (~interior) & (
        np.isclose(np.abs(midpoint[:, 0]), c.W, rtol=0, atol=1e-13)
        | np.isclose(midpoint[:, 1], c.Hair, rtol=0, atol=1e-13)
    )
    dirichlet = (~interior) & (~far)
    edge_squared = length**2 * jump**2
    eta2 = np.zeros(len(geom.elements))
    np.add.at(eta2, a[interior], 0.5 * edge_squared[interior])
    np.add.at(eta2, b[interior], 0.5 * edge_squared[interior])
    np.add.at(eta2, a[far], edge_squared[far])
    return Indicator(eta2, jump, interior, far, dirichlet)


def doerfler_mark(eta_squared, theta=0.5):
    """Smallest prefix in descending indicator order, with index tie breaking."""
    eta_squared = np.asarray(eta_squared)
    if not 0 < theta <= 1 or np.any(eta_squared < 0) or not np.isfinite(eta_squared).all():
        raise ValueError("invalid marking input")
    if eta_squared.sum() == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((np.arange(len(eta_squared)), -eta_squared))
    n = np.searchsorted(np.cumsum(eta_squared[order]), theta * eta_squared.sum()) + 1
    return np.sort(order[:n])
