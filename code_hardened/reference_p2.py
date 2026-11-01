"""Independent reference capacitance with quadratic (P2) elements.

Motivation
----------
The project's evidence chain is entirely P1: the final value C(M7) is quoted as
"0.156% change from M6", and a Richardson fit on the P1 sequence gives
C_ext = 1.2047e-10 F/m.  Both statements are internal to P1.  If the P1 error is
dominated by the re-entrant corner singularity at the trace edges, the observed
order is not 2 and a two-term Richardson model is fragile.

This module solves the *same* PDE on the *same* meshes with a completely
different implementation (scikit-fem, ElementTriP2, elementwise permittivity,
facet-based Dirichlet data).  Quadratic elements converge much faster on this
problem, so the resulting values give an external check on C_ext and let us
quantify the true discretisation error of the frozen M7 value.

Both capacitance routes are computed:
    energy   : C = 2U/V0^2,  U = 1/2 int eps |grad u|^2
    reaction : C = |sum of K u on the trace dofs| / V0

Run:
    python code/reference_p2.py --levels 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP0, ElementTriP2, Functional,
                   MeshTri, asm, condense, solve)
from skfem.helpers import dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import GeometryConfig  # noqa: E402
from uniform_convergence import build_uniform_mesh  # noqa: E402


@BilinearForm
def stiffness(u, v, w):
    return w["eps"] * dot(grad(u), grad(v))


@Functional
def energy_density(w):
    return w["eps"] * dot(grad(w.uh), grad(w.uh))


def dirichlet_dofs(basis: Basis, cfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ground_dofs, trace_dofs, all) using facet predicates."""
    xmin, xmax = cfg.trace_xmin, cfg.trace_xmax
    ymin, ymax = cfg.trace_ymin, cfg.trace_ymax
    tol = 1e-12 * cfg.H

    def on_trace(p):
        inside_x = (p[0] >= xmin - tol) & (p[0] <= xmax + tol)
        inside_y = (p[1] >= ymin - tol) & (p[1] <= ymax + tol)
        vertical = (np.abs(p[0] - xmin) < tol) | (np.abs(p[0] - xmax) < tol)
        horizontal = (np.abs(p[1] - ymin) < tol) | (np.abs(p[1] - ymax) < tol)
        return (vertical & inside_y) | (horizontal & inside_x)

    def on_ground(p):
        return np.abs(p[1]) < tol

    ground = basis.get_dofs(on_ground).all()
    trace = basis.get_dofs(on_trace).all()
    return ground, trace, np.union1d(ground, trace)


def solve_level(level: int, element_name: str, all_air: bool = False) -> dict:
    started = time.perf_counter()
    um = build_uniform_mesh(level)
    geom = um.geom
    mesh = MeshTri(geom.points.T.astype(float), geom.elements.T.astype(np.int64))

    element = ElementTriP2() if element_name == "P2" else None
    if element is None:
        from skfem import ElementTriP1
        element = ElementTriP1()
    basis = Basis(mesh, element)
    # eps is elementwise constant, so it lives in P0 -- but the P0 basis has to
    # use the *same quadrature* as the solution basis or skfem rejects the form.
    basis0 = Basis(mesh, ElementTriP0(), quadrature=basis.quadrature)

    from materials import permittivity
    eps_elem = permittivity(geom.element_regions)
    if all_air:
        # matched all-air reference: same mesh, same Dirichlet data, eps_r = 1
        from materials import EPSILON_0
        eps_elem = np.full_like(eps_elem, EPSILON_0)
    w_eps = basis0.interpolate(eps_elem)

    t0 = time.perf_counter()
    A = asm(stiffness, basis, eps=w_eps).tocsr()
    b = np.zeros(basis.N)
    assembly_time = time.perf_counter() - t0

    ground, trace, D = dirichlet_dofs(basis, geom.config)
    if np.intersect1d(ground, trace).size:
        raise RuntimeError("ground and trace Dirichlet sets overlap")
    u0 = np.zeros(basis.N)
    u0[trace] = geom.config.V0

    t0 = time.perf_counter()
    uh = solve(*condense(A, b, D=D, x=u0))
    solve_time = time.perf_counter() - t0

    U = float(asm(energy_density, basis, uh=uh, eps=w_eps))   # integral only
    V0 = geom.config.V0
    C_energy = U / V0**2                                       # C = 2*(U/2)/V0^2

    reaction = A @ uh
    q_trace = float(np.sum(reaction[trace]))
    q_ground = float(np.sum(reaction[ground]))
    C_reaction = abs(q_trace) / abs(V0)

    # scaled algebraic residual on the free block (scale from the full system)
    free = np.setdiff1d(np.arange(basis.N), D)
    r = (A @ uh - b)[free]
    scale = max(float(np.linalg.norm(A @ uh, np.inf)), float(np.linalg.norm(b, np.inf)),
                np.finfo(float).tiny)
    scaled_residual = float(np.linalg.norm(r, np.inf) / scale)

    return {
        "level": level,
        "element": element_name,
        "elements": int(geom.elements.shape[0]),
        "P1_nodes": int(geom.points.shape[0]),
        "dofs": int(basis.N),
        "nnz": int(A.nnz),
        "C_energy": C_energy,
        "C_reaction": C_reaction,
        "delta_EQ": abs(C_energy - C_reaction) / C_energy,
        "charge_balance": abs(q_trace + q_ground) / max(abs(q_trace), 1e-300),
        "scaled_residual": scaled_residual,
        "assembly_s": assembly_time,
        "solve_s": solve_time,
        "wall_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--element", default="P2", choices=["P1", "P2"])
    parser.add_argument("--all-air", action="store_true",
                        help="matched all-air reference (eps_r = 1 everywhere)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for level in args.levels:
        row = solve_level(level, args.element, all_air=args.all_air)
        rows.append(row)
        print(json.dumps(row), flush=True)

    print("\n%-4s %-3s %10s %10s %20s %20s %10s %8s" % (
        "lv", "el", "elements", "dofs", "C_energy", "C_reaction", "dEQ", "wall_s"))
    for r in rows:
        print("%-4d %-3s %10d %10d %20.12e %20.12e %10.2e %8.2f" % (
            r["level"], r["element"], r["elements"], r["dofs"],
            r["C_energy"], r["C_reaction"], r["delta_EQ"], r["wall_s"]))

    if len(rows) >= 2:
        print("\nconsecutive relative changes and observed orders (energy route):")
        for i in range(1, len(rows)):
            d = abs(rows[i]["C_energy"] - rows[i - 1]["C_energy"]) / rows[i]["C_energy"]
            line = f"  L{rows[i-1]['level']}->L{rows[i]['level']}: rel change {d:.3e}"
            if i >= 2:
                prev = abs(rows[i - 1]["C_energy"] - rows[i - 2]["C_energy"]) / rows[i - 1]["C_energy"]
                if d > 0:
                    p = np.log(prev / d) / np.log(2.0)
                    line += f"   observed order p={p:.3f}"
            print(line)

    out = args.out or (Path(__file__).resolve().parents[1] / "results" / "audit"
                       / f"reference_{args.element}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
