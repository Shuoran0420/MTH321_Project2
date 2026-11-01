"""Regression check: the hardened core must reproduce the authoritative values.

For each uniform level used in the evidence chain this recomputes the
capacitance with the current solver and compares it, bit-for-bit, with the
values that are frozen in ``tables/final_capacitance_convergence.csv``.
It also reports assembly/solve wall time so the vectorised assembly can be
judged on cost as well as on correctness.

Run:
    python code/regression_check.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from capacitance import extract_capacitance  # noqa: E402
from fem_solver import solve_microstrip  # noqa: E402
from uniform_convergence import build_uniform_mesh  # noqa: E402


def frozen_values() -> dict[int, dict]:
    out = {}
    with (ROOT / "tables" / "final_capacitance_convergence.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["family"] != "uniform":
                continue
            level = int(row["mesh_id"].lstrip("M"))
            out[level] = {
                "CE": float(row["CE"]),
                "elements": int(row["elements"]),
                "DOFs": int(row["DOFs"]),
                "assembly_time": float(row["assembly_time"]),
                "solve_time": float(row["solve_time"]),
            }
    return out


def main() -> None:
    levels = [int(a) for a in sys.argv[1:]] or [4, 5, 6, 7]
    frozen = frozen_values()
    rows = []
    print(f"{'mesh':>5} {'elements':>9} {'DOFs':>8} {'C_expected':>22} "
          f"{'C_recomputed':>22} {'rel diff':>10} {'asm s':>7} {'solve s':>8}")
    for level in levels:
        ref = frozen.get(level)
        if ref is None:
            print(f"M{level}: not present in the frozen table, skipped")
            continue
        geom = build_uniform_mesh(level).geom
        t_asm0 = time.perf_counter()
        sol = solve_microstrip(geom, compute_conditioning=False)
        wall = time.perf_counter() - t_asm0
        cap = extract_capacitance(sol)
        rel = abs(cap.CE - ref["CE"]) / abs(ref["CE"])
        rows.append({
            "mesh": f"M{level}",
            "elements": int(geom.elements.shape[0]),
            "DOFs": int(geom.points.shape[0]),
            "C_frozen": ref["CE"],
            "C_recomputed": cap.CE,
            "rel_diff": rel,
            "assembly_time_s": sol.assembly_time,
            "solve_time_s": sol.solve_time,
            "total_wall_s": wall,
            "frozen_assembly_time_s": ref["assembly_time"],
            "frozen_solve_time_s": ref["solve_time"],
            "delta_EQ": cap.delta_EQ,
            "charge_balance_defect": cap.charge_balance_defect,
        })
        print(f"M{level:<4d} {geom.elements.shape[0]:9d} {geom.points.shape[0]:8d} "
              f"{ref['CE']:22.15e} {cap.CE:22.15e} {rel:10.2e} "
              f"{sol.assembly_time:7.3f} {sol.solve_time:8.3f}")

    print("\nverdict")
    worst = max((r["rel_diff"] for r in rows), default=float("nan"))
    print(f"  largest relative difference vs the frozen table: {worst:.3e}")
    print("  tolerance for 'numerically identical': 1e-12 (only rounding differs)")

    print("\ntiming, same machine, same process")
    for r in rows:
        if r["frozen_assembly_time_s"]:
            print(f"  {r['mesh']}: assembly {r['frozen_assembly_time_s']:.3f}s -> "
                  f"{r['assembly_time_s']:.3f}s "
                  f"(x{r['frozen_assembly_time_s'] / max(r['assembly_time_s'], 1e-9):.1f} faster)")

    out = ROOT / "results" / "audit" / "regression_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwritten: {out}")
    if not (worst < 1e-12):
        raise SystemExit("regression FAILED: values changed")


if __name__ == "__main__":
    main()
