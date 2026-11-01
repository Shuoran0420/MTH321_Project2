"""Measure the corner singularity that limits the P1 capacitance error.

The trace is a 3.0 mm x 0.035 mm conducting rectangle sitting on the substrate.
At its four corners the dielectric occupies a 270-degree wedge, so the local
solution behaves like

    phi - phi_corner ~ r^beta,        |grad phi| ~ r^(beta-1) =: r^(-alpha)

with 0 < beta < 1 determined by the corner angle (and, at the two lower
corners, by the eps jump across y = H).  For a P1 space the interpolation error
in H1 is O(h^min(1,beta)), hence the energy (and therefore the capacitance)
error is O(h^(2*min(1,beta))).  That is what turns the smooth O(h^2) rate into
the observed p ~ 1.16.

This module fits alpha from the computed field, then checks the prediction
     p_predicted = 2*min(1, 1-alpha)
against the observed capacitance order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import GeometryConfig  # noqa: E402
from uniform_convergence import build_uniform_mesh  # noqa: E402


def corner_field_fit(level: int, wedge_deg: float = 12.0) -> dict:
    from fem_solver import solve_microstrip

    um = build_uniform_mesh(level)
    geom = um.geom
    sol = solve_microstrip(geom, compute_conditioning=False)
    cfg: GeometryConfig = geom.config

    centroids = geom.points[geom.elements].mean(axis=1)
    gmag = np.linalg.norm(sol.element_gradients, axis=1)

    corners = {
        "upper_right": (cfg.trace_xmax, cfg.trace_ymax),
        "upper_left": (cfg.trace_xmin, cfg.trace_ymax),
        "lower_right": (cfg.trace_xmax, cfg.H),
        "lower_left": (cfg.trace_xmin, cfg.H),
    }
    out = {}
    h = float(np.min(np.ptp(geom.points[geom.elements], axis=1)))
    for name, (cx, cy) in corners.items():
        dx = centroids[:, 0] - cx
        dy = centroids[:, 1] - cy
        r = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        # sample the wedge in the dielectric (right/up of the metal corner)
        if name.startswith("upper"):
            keep_ang = (np.abs(ang) <= wedge_deg) | (np.abs(ang - 90) <= wedge_deg)
        else:
            keep_ang = (np.abs(ang) <= wedge_deg) | (np.abs(ang + 90) <= wedge_deg)
        sel = keep_ang & (r > 6 * h) & (r < 1.2e-3)
        if sel.sum() < 8:
            out[name] = {"n": int(sel.sum()), "alpha": float("nan")}
            continue
        rr = r[sel]
        gg = gmag[sel]
        # bin by logarithmic radius to avoid weighting dense regions
        edges = np.logspace(np.log10(rr.min()), np.log10(rr.max()), 12)
        idx = np.digitize(rr, edges) - 1
        bx, by = [], []
        for k in range(len(edges) - 1):
            m = idx == k
            if m.sum() >= 3:
                bx.append(np.exp(np.mean(np.log(rr[m]))))
                by.append(np.exp(np.mean(np.log(gg[m]))))
        bx, by = np.array(bx), np.array(by)
        slope, intercept = np.polyfit(np.log(bx), np.log(by), 1)
        resid = float(np.max(np.abs(np.log(by) - (slope * np.log(bx) + intercept))))
        out[name] = {
            "n": int(sel.sum()),
            "alpha": float(-slope),
            "beta": float(1 + slope),
            "fit_points": int(len(bx)),
            "max_log_residual": resid,
            "r_range": [float(bx.min()), float(bx.max())],
        }
    out["_level"] = level
    out["_h_min"] = h
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[5, 6])
    args = parser.parse_args()

    rows = {}
    for level in args.levels:
        fit = corner_field_fit(level)
        rows[str(level)] = fit
        print(f"--- M{level} (h_min={fit['_h_min']:.3e} m) ---")
        for name, v in fit.items():
            if name.startswith("_"):
                continue
            if np.isnan(v.get("alpha", np.nan)):
                print(f"  {name:12s}: insufficient samples ({v['n']})")
                continue
            print(f"  {name:12s}: alpha={v['alpha']:.3f}  beta={v['beta']:.3f}  "
                  f"n={v['n']:5d}  r=[{v['r_range'][0]:.2e},{v['r_range'][1]:.2e}]  "
                  f"resid={v['max_log_residual']:.3f}")

    alphas = [v["alpha"] for lv in rows.values() for k, v in lv.items()
              if not k.startswith("_") and np.isfinite(v.get("alpha", np.nan))]
    if alphas:
        alpha = float(np.median(alphas))
        beta = 1.0 - alpha
        p = 2.0 * min(1.0, beta)
        print(f"\nmedian alpha={alpha:.3f} -> beta={beta:.3f} "
              f"-> predicted capacitance order p={p:.3f}")
        print("observed P1 order from the uniform family: p=1.155 (M5-M6-M7 triple)")
        rows["_summary"] = {"alpha_median": alpha, "beta": beta, "p_predicted": p}

    out = Path(__file__).resolve().parents[1] / "results" / "audit" / "corner_singularity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
