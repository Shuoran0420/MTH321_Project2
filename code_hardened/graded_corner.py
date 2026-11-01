"""Corner-graded P1 refinement: can we beat M7 at a fraction of its cost?

Uniform refinement is capped by the re-entrant 270-degree dielectric wedge at the
four trace corners: the P1 capacitance converges at the observed order
p ~ 1.16 instead of 2, so each 4x increase in unknowns only buys a factor
2^1.16 ~ 2.2 in error.

Graded refinement attacks exactly that: keep the far field coarse and place
geometrically shrinking elements around the four corners.  The asymptotic rate
in terms of DOFs is unchanged for 0 < beta < 1, but the constant improves by
orders of magnitude, which is what matters at fixed cost.

The reference value is the Richardson limit of an independent P2 computation
(`code/reference_p2.py`), not a P1 number, so the errors below are not circular.

Run:
    python code/graded_corner.py --base 3 --levels 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_mesh import geometry_from_mesh, refine, symmetry_complete  # noqa: E402
from capacitance import extract_capacitance  # noqa: E402
from fem_solver import solve_microstrip  # noqa: E402
from uniform_convergence import build_uniform_mesh  # noqa: E402

# Richardson limit of the independent P2 sequence (M3..M6), see results/audit/.
REFERENCE_C = 1.2046659e-10


def corners(geom) -> np.ndarray:
    c = geom.config
    return np.array([[c.trace_xmax, c.trace_ymax], [c.trace_xmin, c.trace_ymax],
                     [c.trace_xmax, c.H], [c.trace_xmin, c.H]])


def mark_near_corners(geom, radius: float) -> np.ndarray:
    """Elements whose centroid lies within `radius` of any trace corner."""
    cen = geom.points[geom.elements].mean(axis=1)
    dist = np.min(np.linalg.norm(cen[:, None, :] - corners(geom)[None, :, :], axis=2), axis=1)
    return np.flatnonzero(dist <= radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=3, help="uniform parent level")
    parser.add_argument("--levels", type=int, default=8, help="number of corner rounds")
    parser.add_argument("--sigma", type=float, default=0.35, help="geometric grading ratio")
    parser.add_argument("--radius0", type=float, default=1.6e-4,
                        help="first grading radius in metres")
    args = parser.parse_args()

    print(f"# corner-graded P1, base=M{args.base}, sigma={args.sigma}, "
          f"r0={args.radius0:.2e} m, reference C={REFERENCE_C:.10e} F/m")
    print(f"{'round':>5} {'elements':>9} {'dofs':>8} {'C (F/m)':>20} "
          f"{'rel err vs ref':>15} {'dofs/M7':>8} {'wall s':>7}")

    # `refine` mirrors marked elements, so the parent triangulation must be
    # reflection invariant.  The tensor meshes are not (the quad diagonal flips
    # under x -> -x); symmetry_complete() gives the mirror-invariant 8-triangle
    # cells used by the project's own adaptive study.
    geom = symmetry_complete(build_uniform_mesh(args.base).geom)
    rows = []
    started = time.perf_counter()
    for k in range(args.levels + 1):
        radius = args.radius0 * args.sigma ** max(k - 1, 0)
        if k > 0:
            marked = mark_near_corners(geom, radius * 2.0)
            if marked.size == 0:
                print(f"{k:5d} nothing marked at radius {radius:.2e}; stop")
                break
            geom, _, _ = refine(geom, marked)
        t0 = time.perf_counter()
        sol = solve_microstrip(geom, compute_conditioning=False)
        cap = extract_capacitance(sol)
        wall = time.perf_counter() - t0
        err = abs(cap.CE - REFERENCE_C) / REFERENCE_C
        rows.append({"round": k, "elements": int(geom.elements.shape[0]),
                     "dofs": int(geom.points.shape[0]), "C": cap.CE,
                     "rel_err": err, "dofs_over_M7": geom.points.shape[0] / 362112,
                     "solve_wall_s": wall,
                     "delta_EQ": cap.delta_EQ,
                     "charge_defect": cap.charge_balance_defect})
        print(f"{k:5d} {geom.elements.shape[0]:9d} {geom.points.shape[0]:8d} "
              f"{cap.CE:20.12e} {err:15.3e} {geom.points.shape[0]/362112:8.3f} {wall:7.2f}")

    total = time.perf_counter() - started
    print(f"\ntotal wall time {total:.1f} s")
    if len(rows) >= 2:
        print("\nconvergence per round (error should fall geometrically):")
        for i in range(1, len(rows)):
            if rows[i]["rel_err"] > 0:
                r = rows[i - 1]["rel_err"] / rows[i]["rel_err"]
                print(f"  round {i-1}->{i}: error ratio {r:6.2f}   "
                      f"dofs x{rows[i]['dofs']/rows[i-1]['dofs']:.2f}")

    out = Path(__file__).resolve().parents[1] / "results" / "audit" / "graded_corner.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": vars(args), "reference_C": REFERENCE_C, "rows": rows}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
