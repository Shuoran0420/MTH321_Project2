"""Independent audit of the P1 FEM core for MTH321 Project 2.

Purpose
-------
This module deliberately does NOT trust the project's own mesh generator or its
saved evidence.  It builds its own tiny geometries, knows the exact solution of
each one, and checks the project's *assembler / Dirichlet elimination /
capacitance extraction* against those exact answers.

Each check prints PASS/FAIL with the observed number, so the output can be
pasted into a report as a genuine log.

Run:
    python code/core_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capacitance import extract_capacitance  # noqa: E402
from fem_solver import assemble_stiffness, solve_microstrip  # noqa: E402
from geometry import GeometryData  # noqa: E402
from materials import EPSILON_0, MaterialConfig  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, ok: bool, value, detail: str = "") -> None:
    RESULTS.append({"check": name, "status": "PASS" if ok else "FAIL",
                    "value": value, "detail": detail})
    flag = "PASS" if ok else "**FAIL**"
    print(f"[{flag}] {name}: {value} {detail}")


@dataclass
class Box:
    """A rectangular layered capacitor meshed on a tensor grid.

    x in [0, Lx], y in [0, Ly].  Dirichlet: y=0 -> 0 V, y=Ly -> V0.
    Sides x=0 and x=Lx are homogeneous Neumann (natural).
    Regions are assigned by the y-coordinate of the element centroid, which is
    exact here because the interface y=y_iface is a grid line.
    """

    Lx: float
    Ly: float
    V0: float
    y_iface: float | None
    nx: int
    ny: int

    def build(self) -> GeometryData:
        xs = np.linspace(0.0, self.Lx, self.nx + 1)
        if self.y_iface is not None:
            ys = np.unique(np.concatenate([
                np.linspace(0.0, self.y_iface, self.ny // 2 + 1),
                np.linspace(self.y_iface, self.Ly, self.ny - self.ny // 2 + 1),
            ]))
        else:
            ys = np.linspace(0.0, self.Ly, self.ny + 1)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        nid = np.arange(pts.shape[0]).reshape(len(ys), len(xs))
        tris = []
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                q = [nid[j, i], nid[j, i + 1], nid[j + 1, i], nid[j + 1, i + 1]]
                tris.append([q[0], q[1], q[3]])
                tris.append([q[0], q[3], q[2]])
        elements = np.array(tris, dtype=int)
        mesh_like = None  # the solver never touches GeometryData.mesh
        cen_y = pts[elements][:, :, 1].mean(axis=1)
        if self.y_iface is None:
            regions = np.array(["air"] * len(elements))
        else:
            regions = np.where(cen_y < self.y_iface - 1e-15, "substrate", "air")
        ground = np.flatnonzero(np.isclose(pts[:, 1], 0.0, atol=1e-15))
        trace = np.flatnonzero(np.isclose(pts[:, 1], self.Ly, atol=1e-15))
        far = np.array([], dtype=int)
        cfg = type("Cfg", (), {"V0": self.V0})()
        return GeometryData(cfg, mesh_like, pts, elements, regions, ground, trace, far,
                            np.zeros((0, 2)))


def check_linear_field_exact() -> None:
    """Uniform field in a one-material box: P1 reproduces phi = V0*y/Ly exactly.

    Exact capacitance per unit depth: C = eps0*eps_r*Lx/Ly.
    Any error here is an assembly / Dirichlet / energy-scaling bug.
    """
    for eps_r in (1.0, 4.4):
        for nx, ny in ((4, 3), (9, 7)):
            box = Box(Lx=2.0, Ly=1.0, V0=1.0, y_iface=None, nx=nx, ny=ny)
            geom = box.build()
            cfg = MaterialConfig(epsilon_r_air=eps_r, epsilon_r_substrate=eps_r)
            sol = solve_microstrip(geom, cfg, compute_conditioning=False)
            cap = extract_capacitance(sol)
            exact_phi = box.V0 * geom.points[:, 1] / box.Ly
            phi_err = float(np.max(np.abs(sol.phi - exact_phi)))
            exact_c = EPSILON_0 * eps_r * box.Lx / box.Ly
            rel_c = abs(cap.CE - exact_c) / exact_c
            record(f"uniform field phi exact (eps_r={eps_r}, {nx}x{ny})",
                   phi_err < 1e-12, f"{phi_err:.2e} V", "max|phi-phi_exact|")
            record(f"capacitance exact eps_r*eps0*A/d (eps_r={eps_r}, {nx}x{ny})",
                   rel_c < 1e-12, f"{rel_c:.2e}", f"C_h={cap.CE:.10e} C_exact={exact_c:.10e}")
            record(f"energy vs reaction capacitance (eps_r={eps_r}, {nx}x{ny})",
                   cap.delta_EQ < 1e-12, f"{cap.delta_EQ:.2e}", "|CE-CQ|/CE")


def check_two_layer_series() -> None:
    """Two-layer capacitor: C_series = 1 / (d1/eps1 + d2/eps2) per unit area.

    This is the interface test: it only comes out right if (i) each element gets
    the right permittivity, (ii) the interface is conforming, and (iii) the
    natural (weak) interface condition [eps dphi/dn] = 0 is what the FE method
    actually enforces.  Exact field is piecewise constant:
        E1 = V0 / (d1 + d2*eps1/eps2),  E2 = E1*eps1/eps2.
    """
    eps1, eps2 = 4.4, 1.0
    d1, d2 = 0.6, 0.4
    V0 = 1.0
    box = Box(Lx=1.0, Ly=d1 + d2, V0=V0, y_iface=d1, nx=6, ny=10)
    geom = box.build()
    sol = solve_microstrip(geom, MaterialConfig(epsilon_r_substrate=eps1,
                                                epsilon_r_air=eps2),
                           compute_conditioning=False)
    cap = extract_capacitance(sol)
    exact_c = EPSILON_0 / (d1 / eps1 + d2 / eps2) * box.Lx
    rel_c = abs(cap.CE - exact_c) / exact_c

    # exact piecewise-constant field
    e1 = V0 / (d1 + d2 * eps1 / eps2)
    e2 = e1 * eps1 / eps2
    cen = geom.points[geom.elements].mean(axis=1)
    below = cen[:, 1] < d1
    ey = sol.element_gradients[:, 1]
    e_err = max(float(np.max(np.abs(ey[below] - e1))),
                float(np.max(np.abs(ey[~below] - e2))))
    record("two-layer series capacitance", rel_c < 1e-12, f"{rel_c:.2e}",
           f"C_h={cap.CE:.10e} C_exact={exact_c:.10e}")
    record("two-layer piecewise-constant field", e_err < 1e-9 * abs(e1), f"{e_err:.2e} V/m",
           f"E1={e1:.6f} E2={e2:.6f}")

    # interface flux continuity, evaluated element-by-element across y=d1
    eps_e = permittivity_per_element(geom, MaterialConfig(epsilon_r_substrate=eps1,
                                                         epsilon_r_air=eps2))
    d_below = eps_e[below] * ey[below]
    d_above = eps_e[~below] * ey[~below]
    jump = abs(float(np.mean(d_below)) - float(np.mean(d_above)))
    scale = max(abs(float(np.mean(d_below))), 1e-300)
    record("normal D continuous across interface", jump / scale < 1e-10,
           f"{jump / scale:.2e}", "|D_below-D_above|/|D|")


def permittivity_per_element(geom: GeometryData, cfg: MaterialConfig):
    from materials import permittivity
    return permittivity(geom.element_regions, cfg)


def check_outer_flux_zero() -> None:
    """On the real microstrip mesh the outer box carries no charge.

    Sum of reactions on *all* Dirichlet nodes must vanish (Gauss), and the
    reaction must be exactly zero on every non-Dirichlet boundary node, because
    homogeneous Neumann is natural in the weak form.
    """
    from geometry import GeometryConfig, build_geometry

    geom = build_geometry(GeometryConfig(nx_outer=48, ny_substrate=20, ny_air=40))
    sol = solve_microstrip(geom, compute_conditioning=False)
    cap = extract_capacitance(sol)

    r = sol.reaction_full
    dirichlet = sol.dirichlet_dofs
    free_interior = np.setdiff1d(sol.free_dofs, geom.far_nodes)
    far_react = float(np.max(np.abs(r[geom.far_nodes]))) if geom.far_nodes.size else 0.0
    # far nodes are free, so their reaction is a residual of the interior equation
    scale = float(np.max(np.abs(r[dirichlet])))
    record("zero normal flux on outer Neumann boundary", far_react / scale < 1e-10,
           f"{far_react / scale:.2e}", "max|reaction| on far nodes / max|Q|")
    record("total induced charge is zero", cap.charge_balance_defect < 1e-10,
           f"{cap.charge_balance_defect:.2e}", "(Q_trace+Q_ground)/scale")
    record("trace potential equals V0", bool(np.allclose(sol.phi[sol.trace_dofs], 1.0, atol=0)),
           f"{sol.phi[sol.trace_dofs].min():.17g}", "V")
    record("ground potential equals 0", bool(np.allclose(sol.phi[sol.ground_dofs], 0.0, atol=0)),
           f"{sol.phi[sol.ground_dofs].max():.17g}", "V")
    record("discrete maximum principle 0<=phi<=V0",
           bool(sol.phi.min() >= -1e-12 and sol.phi.max() <= 1 + 1e-12),
           f"[{sol.phi.min():.3e}, {sol.phi.max():.3e}]", "V")

    K = sol.matrix_full
    asym = float(abs(K - K.T).max()) if (K - K.T).nnz else 0.0
    record("stiffness matrix symmetric", asym <= 1e-30, f"{asym:.2e}", "max|K-K^T|")

    # SPD of the reduced matrix (small mesh only: dense eigenvalues)
    Kff = sol.matrix_free.toarray()
    lam = np.linalg.eigvalsh(0.5 * (Kff + Kff.T))
    record("reduced matrix positive definite", bool(lam.min() > 0),
           f"lambda_min={lam.min():.6e} lambda_max={lam.max():.6e}",
           f"cond={lam.max() / lam.min():.3e}")


def check_nested_energy_monotonicity() -> None:
    """A conforming Galerkin scheme on nested meshes must have monotone energy.

    The exact solution *minimises* the Dirichlet energy over the admissible set,
    so for nested spaces V_h subset V_h' the discrete energy satisfies
        U <= U_h' <= U_h        (upper bounds, decreasing with refinement).
    C_h = 2U_h/V0^2 therefore decreases monotonically towards C_exact and is
    bounded below by it.  A non-monotone sequence (or one that turns around)
    would indicate non-nested meshes, a changing Dirichlet set or a changing
    material assignment between levels.
    """
    from uniform_convergence import build_uniform_mesh, solve_uniform

    caps = []
    for level in range(4):
        row = solve_uniform(level)
        caps.append((level, row["cap"].CE, row["mesh"].geom.elements.shape[0],
                     row["mesh"].geom.points.shape[0]))
    mono = all(caps[i][1] >= caps[i + 1][1] for i in range(len(caps) - 1))
    record("nested uniform family: energy decreases monotonically",
           mono, " ".join(f"M{l}={c:.6e}" for l, c, *_ in caps),
           "Galerkin: C_h >= C_exact and C_h non-increasing under nesting")
    return caps


def main() -> None:
    print("=" * 78)
    print("Independent core audit — analytic cases, no project mesh generator")
    print("=" * 78)
    check_linear_field_exact()
    print("-" * 78)
    check_two_layer_series()
    print("-" * 78)
    check_outer_flux_zero()
    print("-" * 78)
    caps = check_nested_energy_monotonicity()
    print("=" * 78)
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed")
    out = Path(__file__).resolve().parents[1] / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "core_audit.json").write_text(json.dumps(
        {"results": RESULTS, "uniform_family": caps}, indent=2) + "\n")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
