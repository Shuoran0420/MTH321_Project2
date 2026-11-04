"""Boundary-row audit: hand-check the Dirichlet row on non-zero data.

Tutorial 8, individual challenge, flavour 3 ("verify a boundary").

The project imposes the Dirichlet condition by *elimination*: rows and columns
of the fixed nodes are dropped from K, and the known values are moved to the
right-hand side,

    K_ff u_f = -K_fd g_d                                              (E)

No file in the repository ever checks that this is the same system a textbook
row-replacement would produce,

    A[i,:] = e_i,  b[i] = g_i   for i in D                            (T)

and every existing analytic test uses a *single* non-zero electrode (trace V0,
ground forced to 0), so a sign error in the shift `-K_fd g_d` would cancel out
of most of them.  This module supplies the missing check.

Three independent constructions are compared on the same K:

    E  elimination, written out exactly as fem_solver.solve_microstrip does it
    T  textbook row replacement, assembled as an explicit n x n system
    S  the shipped solver itself, run on (0, V) and shifted by a constant
       (translation invariance), which is the only way to give the shipped
       ground electrode a non-zero value

and, for the layered box, all three are compared with the closed-form
piecewise-linear field of a series capacitor, which P1 reproduces exactly
when the interface is a mesh line.

Four mutants show the test is not vacuous: each breaks one line of (E) and the
error must jump from roundoff to O(scale).

Run:
    python code_hardened/boundary_row_audit.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import diags, identity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core_audit import Box  # noqa: E402
from fem_solver import assemble_stiffness, solve_microstrip  # noqa: E402
from geometry import GeometryConfig, GeometryData, build_geometry  # noqa: E402
from materials import EPSILON_0, MaterialConfig  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, ok: bool, value: str, detail: str = "") -> None:
    RESULTS.append({"check": name, "status": "PASS" if ok else "FAIL",
                    "value": value, "detail": detail})
    print(f"[{'PASS' if ok else '**FAIL**'}] {name}: {value}  {detail}")


# ---------------------------------------------------------------------------
# A. one element, exact rational arithmetic
# ---------------------------------------------------------------------------
def hand_element_stiffness() -> list[list[Fraction]]:
    """Stiffness of the triangle A(0,0), B(1,0), C(2,1) with eps = 1.

    Constant P1 gradients (2A = 1, A = 1/2):
        grad N_A = (-1, 1),  grad N_B = (1, -2),  grad N_C = (0, 1)
    so K_ij = eps * A * grad_i . grad_j is exactly
        [[1, -3/2, 1/2], [-3/2, 5/2, -1], [1/2, -1, 1/2]]
    computed here from the gradients, not typed in, so the derivation is in the
    file that checks it.
    """
    grads = [(Fraction(-1), Fraction(1)),
             (Fraction(1), Fraction(-2)),
             (Fraction(0), Fraction(1))]
    area = Fraction(1, 2)
    return [[area * (gi[0] * gj[0] + gi[1] * gj[1]) for gj in grads] for gi in grads]


def check_single_element() -> dict:
    """Hand-eliminate one element: ground A = 0, trace C = 3, free B."""
    K_hand = hand_element_stiffness()
    elems = np.array([[0, 1, 2]], dtype=int)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])

    cfg = type("Cfg", (), {"V0": 3.0})()
    geom = GeometryData(cfg, None, pts, elems, np.array(["air"]),
                        ground_nodes=np.array([0]), trace_nodes=np.array([2]),
                        far_nodes=np.array([], dtype=int),
                        conductor_polygon=np.zeros((0, 2)))
    mat = MaterialConfig(epsilon_r_air=1.0, epsilon_r_substrate=1.0)
    K, *_ = assemble_stiffness(geom, mat)

    # assemble_stiffness multiplies by the physical permittivity eps0 * eps_r,
    # so the hand matrix (eps = 1) has to be scaled by eps0 before comparison.
    Kref = np.array([[float(x) for x in row] for row in K_hand]) * EPSILON_0
    dK = float(np.max(np.abs(K.toarray() - Kref))) / float(np.max(np.abs(Kref)))
    record("hand element stiffness == assembled (exact fractions)", dK <= 1e-15,
           f"{dK:.1e}", "max_ij |K_code - eps0*K_hand| / max|K|, eps_r=1")

    # hand elimination of the single free node B (index 1), g = (0, _, 3)
    gA, gC = Fraction(0), Fraction(3)
    rhs_hand = -(K_hand[1][0] * gA + K_hand[1][2] * gC)
    uB_hand = rhs_hand / K_hand[1][1]

    sol = solve_microstrip(geom, mat, compute_conditioning=False)
    rhs_code = float(sol.rhs_free[0])
    rhs_ref = float(rhs_hand) * EPSILON_0
    d_rhs = abs(rhs_code - rhs_ref) / abs(rhs_ref)
    record("hand rhs of the free row == code rhs", d_rhs <= 1e-15,
           f"{d_rhs:.1e}", f"hand={rhs_ref:.6e} code={rhs_code:.6e}")
    uB_code = float(sol.phi[1])
    record("hand solution of the free row == code solution",
           abs(uB_code - float(uB_hand)) <= 1e-15, f"{abs(uB_code - float(uB_hand)):.1e}",
           f"u_B = 6/5 = {float(uB_hand):.1f} V, code {uB_code:.15f} V")
    return {"K_hand": [[str(x) for x in row] for row in K_hand],
            "rhs_hand": str(rhs_hand), "u_B_hand": str(uB_hand)}


# ---------------------------------------------------------------------------
# B. general non-zero Dirichlet data: three constructions of the same system
# ---------------------------------------------------------------------------
def textbook_system(K, fixed: np.ndarray, g: np.ndarray):
    """Row-replacement system (T): Dirichlet rows become e_i, b_i = g_i."""
    n = K.shape[0]
    mask = np.zeros(n)
    mask[fixed] = 1.0
    P = diags(mask)
    # The textbook system keeps the columns of the fixed nodes (their values are
    # now known, x_fixed = g), so the free rows keep the original right-hand side
    # (zero here) and it is the *columns* that carry the known data.  Mixing this
    # with the column-eliminated shift -K_fd g would count the data twice.
    A = (identity(n, format="csr") - P) @ K + P          # zero the fixed rows, put 1 on the diagonal
    b = mask * g
    return A.tocsr(), np.asarray(b, dtype=float)


def elimination_system(K, fixed: np.ndarray, g: np.ndarray, free: np.ndarray):
    """Exactly the four lines of fem_solver.solve_microstrip, with general g."""
    K_ff = K[free][:, free].tocsr()
    rhs = np.asarray(-K[free][:, fixed] @ g[fixed]).ravel()
    return K_ff, rhs


def solve_dense(A, b):
    from scipy.sparse.linalg import spsolve
    return spsolve(A.tocsc(), b)


def layered_box(nx: int = 8, ny: int = 12) -> tuple[GeometryData, MaterialConfig, float]:
    """Series capacitor: substrate 4.4 over air, interface at y = 0.4, Ly = 1."""
    box = Box(Lx=2.0, Ly=1.0, V0=1.0, y_iface=0.4, nx=nx, ny=ny)
    return box.build(), MaterialConfig(epsilon_r_substrate=4.4, epsilon_r_air=1.0), 0.4


def exact_series_field(geom: GeometryData, g_bottom: float, g_top: float,
                       d1: float, eps1: float, eps2: float) -> np.ndarray:
    """Closed form of the layered series capacitor, per node."""
    d2 = float(np.max(geom.points[:, 1])) - d1
    dv = g_top - g_bottom
    e1 = dv / (d1 + d2 * eps1 / eps2)
    e2 = e1 * eps1 / eps2
    y = geom.points[:, 1]
    return g_bottom + e1 * np.minimum(y, d1) + e2 * np.maximum(y - d1, 0.0)


def compare_three(name: str, geom: GeometryData, mat: MaterialConfig,
                  g_bottom: float, g_top: float, exact=None, d1=None) -> dict:
    """Run E, T, S on one geometry with non-zero data on both electrodes."""
    K, *_ = assemble_stiffness(geom, mat)
    n = K.shape[0]
    ground = np.asarray(geom.ground_nodes, dtype=int)
    trace = np.asarray(geom.trace_nodes, dtype=int)
    fixed = np.union1d(ground, trace)
    free = np.setdiff1d(np.arange(n, dtype=int), fixed, assume_unique=True)
    g = np.zeros(n)
    g[ground] = g_bottom
    g[trace] = g_top
    scale = max(abs(g_bottom), abs(g_top))

    # E: elimination, verbatim
    K_ff, rhs = elimination_system(K, fixed, g, free)
    phi_E = np.zeros(n)
    phi_E[fixed] = g[fixed]
    phi_E[free] = solve_dense(K_ff, rhs)

    # T: textbook row replacement
    A, b = textbook_system(K, fixed, g)
    phi_T = solve_dense(A, b)

    # S: the shipped solver, run on (0, V) and shifted (ground is hard-zero there)
    shift = g_bottom
    cfg = type("Cfg", (), {"V0": g_top - shift})()
    geom_s = GeometryData(cfg, geom.mesh, geom.points, geom.elements,
                          geom.element_regions, ground, trace, geom.far_nodes,
                          geom.conductor_polygon)
    phi_S = solve_microstrip(geom_s, mat, compute_conditioning=False).phi + shift

    d_ET = float(np.max(np.abs(phi_E - phi_T))) / scale
    d_ST = float(np.max(np.abs(phi_S - phi_T))) / scale
    # free-row residual of E: ||K_ff u_f + K_fd g_d||_inf / ||K u||_inf
    res = float(np.max(np.abs(K_ff @ phi_E[free] + K[free][:, fixed] @ g[fixed])))
    ref = float(np.max(np.abs(K @ phi_T))) or 1.0
    res_r = res / ref

    print(f"--- {name}: n={n}, |D|={fixed.size}, g=({g_bottom:g},{g_top:g})")
    record(f"{name}: elimination == row replacement", d_ET <= 1e-13,
           f"{d_ET:.1e}", "max|phi_E - phi_T| / max|g|")
    record(f"{name}: shipped solver == row replacement", d_ST <= 1e-13,
           f"{d_ST:.1e}", "max|phi_S - phi_T| / max|g|")
    record(f"{name}: free-row residual of the eliminated system", res_r <= 1e-13,
           f"{res_r:.1e}", "max|K_ff u_f + K_fd g_d| / max|K u|")

    out = {"n": int(n), "dirichlet": int(fixed.size), "d_ET": d_ET,
           "d_ST": d_ST, "residual": res_r, "phi_T": phi_T, "K": K, "fixed": fixed,
           "g": g, "free": free, "scale": scale}

    if exact is not None:
        d_ex = float(np.max(np.abs(phi_T - exact))) / scale
        record(f"{name}: row-replacement solution == closed form", d_ex <= 1e-12,
               f"{d_ex:.1e}", "max|phi - phi_exact| / max|g|, layered series field")
        out["d_exact"] = d_ex

    # ---- mutants: break one line of (E) and see the number move ----
    mut = {}
    K_fd = K[free][:, fixed]          # only the free rows enter the reduced system
    g_fixed = g[fixed]

    def solve_mutant(bad_rhs):
        x = np.zeros(n)
        x[fixed] = g[fixed]
        x[free] = solve_dense(K_ff, bad_rhs)
        return float(np.max(np.abs(x - phi_T))) / scale

    mut["rhs sign flipped"] = solve_mutant(K_fd @ g_fixed)
    mut["shift dropped"] = solve_mutant(np.zeros(free.size))
    mut["shift doubled"] = solve_mutant(-2.0 * K_fd @ g_fixed)

    # textbook variant that forgets to replace the Dirichlet row: it keeps the
    # full K row instead of e_i (a one-sided "closure" error)
    mask = np.zeros(n)
    mask[fixed] = 1.0
    P = diags(mask)
    A_bad = A + P @ (K - identity(n, format="csr"))
    b_bad = b.copy()
    phi_bad = solve_dense(A_bad.tocsr(), b_bad)
    mut["Dirichlet row left as K row"] = float(np.max(np.abs(phi_bad - phi_T))) / scale

    # the bug the observation is about: keep only the trace columns of the shift,
    # i.e. silently assume g_ground = 0 (which is exactly what the shipped
    # solver hard-wires, so this error is invisible on the project's own mesh)
    mut["ground columns dropped from shift"] = solve_mutant(
        -K[free][:, np.asarray(geom.trace_nodes, dtype=int)]
        @ g[np.asarray(geom.trace_nodes, dtype=int)])

    for k, v in mut.items():
        print(f"    mutant [{k}]: {v:.3e}")
    out["mutants"] = mut
    return out


def check_layered_box() -> dict:
    """Non-zero data on *both* electrodes, closed form available."""
    geom, mat, d1 = layered_box()
    g_bottom, g_top = 1.0, 3.0
    exact = exact_series_field(geom, g_bottom, g_top, d1, 4.4, 1.0)
    return compare_three("layered box (8x12)", geom, mat, g_bottom, g_top, exact, d1)


def check_real_mesh() -> dict:
    """The submitted microstrip mesh: ground 0, trace 1, plus mutants."""
    geom = build_geometry(GeometryConfig())      # the project's default 12k-node mesh
    mat = MaterialConfig()
    K, *_ = assemble_stiffness(geom, mat)
    n = K.shape[0]
    ground = np.asarray(geom.ground_nodes, dtype=int)
    trace = np.asarray(geom.trace_nodes, dtype=int)
    fixed = np.union1d(ground, trace)
    free = np.setdiff1d(np.arange(n, dtype=int), fixed, assume_unique=True)
    g = np.zeros(n)
    g[trace] = 1.0                      # the one non-zero electrode the project uses
    scale = 1.0

    K_ff, rhs = elimination_system(K, fixed, g, free)
    phi_E = np.zeros(n)
    phi_E[fixed] = g[fixed]
    phi_E[free] = solve_dense(K_ff, rhs)
    A, b = textbook_system(K, fixed, g)
    phi_T = solve_dense(A, b)
    d_ET = float(np.max(np.abs(phi_E - phi_T))) / scale
    res = float(np.max(np.abs(K_ff @ phi_E[free] + K[free][:, fixed] @ g[fixed])))
    ref = float(np.max(np.abs(K @ phi_T))) or 1.0

    print(f"--- microstrip mesh (default 96x40x80): n={n}, |D|={fixed.size}")
    record("microstrip: elimination == row replacement", d_ET <= 1e-13, f"{d_ET:.1e}",
           "max|phi_E - phi_T|, ground 0 trace 1")
    record("microstrip: free-row residual", res / ref <= 1e-13, f"{res / ref:.1e}",
           "max|K_ff u_f + K_fd g_d| / max|K u|")

    K_fd = K[free][:, fixed]
    g_fixed = g[fixed]

    def mutant(bad):
        x = np.zeros(n)
        x[fixed] = g[fixed]
        x[free] = solve_dense(K_ff, bad)
        return float(np.max(np.abs(x - phi_T)))

    mut = {"rhs sign flipped": mutant(K_fd @ g_fixed),
           "shift dropped": mutant(np.zeros(free.size)),
           "shift doubled": mutant(-2.0 * K_fd @ g_fixed)}
    trace_g = -K[free][:, trace] @ g[trace]
    mut_ground = mutant(trace_g)
    mut["ground columns dropped from shift"] = mut_ground
    for k, v in mut.items():
        print(f"    mutant [{k}]: {v:.3e} V")
    record("microstrip: shift mutants are O(1), the check has teeth",
           min(mut[k] for k in ("rhs sign flipped", "shift dropped", "shift doubled")) > 1e-3,
           f"min shift mutant {min(mut[k] for k in ('rhs sign flipped', 'shift dropped', 'shift doubled')):.3e} V",
           "sign flip / drop / double the -K_fd g_d shift")
    record("microstrip: the ground-column bug is invisible here (ground = 0 V)",
           mut_ground <= 1e-12, f"{mut_ground:.1e} V",
           "dropping the ground columns of the shift is undetectable at round-off")
    return {"n": int(n), "dirichlet": int(fixed.size), "d_ET": d_ET,
            "residual": res / ref, "mutants": mut}


def main() -> None:
    print("=" * 78)
    print("Boundary-row audit — is the eliminated Dirichlet row the textbook row?")
    print("=" * 78)
    hand = check_single_element()
    print("-" * 78)
    box = check_layered_box()
    print("-" * 78)
    real = check_real_mesh()
    print("=" * 78)
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed")

    box_out = {k: v for k, v in box.items() if k not in ("phi_T", "K", "fixed", "g", "free")}
    payload = {"results": RESULTS, "single_element": hand, "layered_box": box_out,
               "microstrip": real}
    out = HERE / "results" / "boundary_row_audit.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"written: {out}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
