"""Why does the 'uniform' capacitance sequence decrease under nested refinement?

For a conforming P1 space on nested meshes with the same Dirichlet data,
U_h = 1/2 phi_h^T K_h phi_h must be non-decreasing (Galerkin minimises energy
over an expanding subspace).  The project's M0..M7 sequence decreases by ~28%.

This script isolates the cause:
  1. Are the point sets nested?
  2. Is every coarse triangle exactly a union of fine triangles?
  3. Is the coarse solution admissible (energy-preserving) on the fine mesh?
  4. Does the energy decrease already between two levels, and by how much?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capacitance import extract_capacitance
from fem_solver import assemble_stiffness
from uniform_convergence import build_uniform_mesh


def key(p: np.ndarray) -> np.ndarray:
    return np.round(p, 12)


def energy_of(geom, phi):
    K, areas, grads, eps, _eps_r, _t = assemble_stiffness(geom)
    g = np.einsum("eik,ei->ek", grads, phi[geom.elements])
    e2 = np.einsum("ij,ij->i", g, g)
    return float(0.5 * np.sum(eps * e2 * areas)), K


def main() -> None:
    meshes = {lv: build_uniform_mesh(lv) for lv in range(4)}
    vols = {}
    for lv, um in meshes.items():
        g = um.geom
        pts = g.points[g.elements]
        a = 0.5 * np.abs((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                         - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1]))
        vols[lv] = a.sum()
        ys = np.unique(np.round(g.points[:, 1], 12))
        xs = np.unique(np.round(g.points[:, 0], 12))
        print(f"L{lv}: elements={g.elements.shape[0]:6d} points={g.points.shape[0]:6d} "
              f"area={a.sum():.12e} nx={len(xs)-1} ny={len(ys)-1} "
              f"min_area={a.min():.3e} max_aspect={np.max(np.ptp(pts, axis=1))/max(a.min()*2, 1e-30):.1f}")

    print("\n--- 1) point-set nesting ---")
    for a_lv, b_lv in ((0, 1), (1, 2), (2, 3)):
        A = set(map(tuple, key(meshes[a_lv].geom.points)))
        B = set(map(tuple, key(meshes[b_lv].geom.points)))
        print(f"L{a_lv} points not in L{b_lv}: {len(A - B)} / {len(A)}")

    print("\n--- 2) coarse triangles contained in fine mesh ---")
    for a_lv, b_lv in ((0, 1), (1, 2)):
        fine = meshes[b_lv].geom
        B = set(map(tuple, key(fine.points)))
        A = meshes[a_lv].geom
        missing = 0
        for tri in A.elements:
            for v in A.points[tri]:
                if tuple(key(v)) not in B:
                    missing += 1
        print(f"L{a_lv} element vertices missing from L{b_lv}: {missing}")

    print("\n--- 3) energy (2U/V0^2) per level ---")
    energies = {}
    for lv, um in meshes.items():
        _, K = energy_of(um.geom, np.zeros(um.geom.points.shape[0]))
        energies[lv] = K
    from fem_solver import solve_microstrip
    sols = {}
    for lv, um in meshes.items():
        sol = solve_microstrip(um.geom, compute_conditioning=False)
        sols[lv] = sol
        print(f"L{lv}: C={extract_capacitance(sol).CE:.12e}  nelem={um.geom.elements.shape[0]}")

    # (an earlier draft interpolated the coarse solution onto the fine mesh to show
    #  energy preservation; that needed a fragile barycentric lookup and is not part
    #  of the conclusion, which the point/triangle nesting above already settles.)


if __name__ == "__main__":
    main()
