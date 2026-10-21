"""Gate D uniform-refinement convergence and corner-field diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capacitance import extract_capacitance
from fem_solver import solve_microstrip
from geometry import GeometryConfig, GeometryData, _inside_conductor, _on_trace_boundary
from materials import MaterialConfig
from skfem import MeshTri


@dataclass
class UniformMesh:
    geom: GeometryData
    xs: np.ndarray
    ys: np.ndarray
    point_index: dict[tuple[int, int], int]


def _refined_axis(base: list[float], level: int) -> np.ndarray:
    vals = np.array(sorted(set(base)), dtype=float)
    for _ in range(level):
        mids = 0.5 * (vals[:-1] + vals[1:])
        vals = np.unique(np.concatenate([vals, mids]))
        vals.sort()
    return vals


def build_uniform_mesh(level: int) -> UniformMesh:
    cfg = GeometryConfig(W_over_H=8.0, Hair_over_H=8.0)
    xs = _refined_axis([-cfg.W, -3.2e-3, cfg.trace_xmin, 0.0, cfg.trace_xmax, 3.2e-3, cfg.W], level)
    ys = _refined_axis([0.0, cfg.H, cfg.trace_ymax, 2.4e-3, cfg.Hair], level)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    all_points = np.column_stack([xx.ravel(), yy.ravel()])
    all_ids = np.arange(all_points.shape[0]).reshape(len(ys), len(xs))
    tris: list[list[int]] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            quad = [all_ids[j, i], all_ids[j, i + 1], all_ids[j + 1, i], all_ids[j + 1, i + 1]]
            cx = float(all_points[quad, 0].mean())
            cy = float(all_points[quad, 1].mean())
            if _inside_conductor(np.array([cx]), np.array([cy]), cfg)[0]:
                continue
            tris.append([quad[0], quad[1], quad[3]])
            tris.append([quad[0], quad[3], quad[2]])
    used = np.unique(np.array(tris, dtype=int).ravel())
    remap = -np.ones(all_points.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    points = all_points[used]
    elements = remap[np.array(tris, dtype=int)]
    mesh = MeshTri(points.T, elements.T)
    centroids = points[elements].mean(axis=1)
    regions = np.where(centroids[:, 1] < cfg.H - 1e-13, "substrate", "air")
    ground = np.flatnonzero(np.isclose(points[:, 1], 0.0, atol=1e-12))
    trace = np.flatnonzero(_on_trace_boundary(points[:, 0], points[:, 1], cfg))
    far = np.flatnonzero(
        (
            np.isclose(points[:, 0], -cfg.W, atol=1e-12)
            | np.isclose(points[:, 0], cfg.W, atol=1e-12)
            | np.isclose(points[:, 1], cfg.Hair, atol=1e-12)
        )
        & ~np.isin(np.arange(points.shape[0]), ground)
        & ~np.isin(np.arange(points.shape[0]), trace)
    )
    poly = np.array([[cfg.trace_xmin, cfg.H], [cfg.trace_xmax, cfg.H], [cfg.trace_xmax, cfg.trace_ymax], [cfg.trace_xmin, cfg.trace_ymax], [cfg.trace_xmin, cfg.H]])
    point_index = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            old = all_ids[j, i]
            if remap[old] >= 0:
                point_index[(i, j)] = int(remap[old])
    return UniformMesh(GeometryData(cfg, mesh, points, elements, regions, ground, trace, far, poly), xs, ys, point_index)


def element_sizes(geom: GeometryData) -> tuple[float, float, float]:
    pts = geom.points[geom.elements]
    lengths = np.stack([
        np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1),
    ], axis=1)
    diam = lengths.max(axis=1)
    return float(diam.max()), float(diam.min()), float(np.median(diam))


def solve_uniform(level: int) -> dict:
    um = build_uniform_mesh(level)
    sol = solve_microstrip(um.geom, compute_conditioning=False)
    cap = extract_capacitance(sol)
    free_solution = sol.phi[sol.free_dofs]
    residual = sol.matrix_free @ free_solution - sol.rhs_free
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    system_scale = float(np.linalg.norm(sol.matrix_free @ free_solution, ord=np.inf) + np.linalg.norm(sol.rhs_free, ord=np.inf))
    normalized_residual = residual_inf / max(system_scale, 1e-300)
    hmax, hmin, hmed = element_sizes(um.geom)
    max_e_idx = int(np.argmax(np.linalg.norm(sol.element_gradients, axis=1)))
    centroid = um.geom.points[um.geom.elements[max_e_idx]].mean(axis=0)
    return {
        "level": level,
        "mesh": um,
        "solution": sol,
        "cap": cap,
        "residual_inf": residual_inf,
        "normalized_residual": normalized_residual,
        "h_max": hmax,
        "h_min": hmin,
        "h_median": hmed,
        "max_E": float(np.linalg.norm(sol.element_gradients[max_e_idx])),
        "max_E_location": centroid,
    }


def _element_gradient(points: np.ndarray, tri: np.ndarray, values: np.ndarray) -> np.ndarray:
    xy = points[tri]
    x0, y0 = xy[0]; x1, y1 = xy[1]; x2, y2 = xy[2]
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    grads = np.array([[y1-y2, x2-x1], [y2-y0, x0-x2], [y0-y1, x1-x0]]) / area2
    return values[tri] @ grads


def coarse_gradient_on_ref_centroid(coarse: dict, x: float, y: float) -> np.ndarray:
    um: UniformMesh = coarse["mesh"]
    xs, ys = um.xs, um.ys
    i = int(np.searchsorted(xs, x, side="right") - 1)
    j = int(np.searchsorted(ys, y, side="right") - 1)
    i = max(0, min(i, len(xs) - 2)); j = max(0, min(j, len(ys) - 2))
    bl = um.point_index.get((i, j)); br = um.point_index.get((i + 1, j)); tl = um.point_index.get((i, j + 1)); tr = um.point_index.get((i + 1, j + 1))
    if None in (bl, br, tl, tr):
        return np.array([np.nan, np.nan])
    xi = (x - xs[i]) / (xs[i+1] - xs[i])
    eta = (y - ys[j]) / (ys[j+1] - ys[j])
    tri = np.array([bl, br, tr]) if eta <= xi else np.array([bl, tr, tl])
    return _element_gradient(um.geom.points, tri, coarse["solution"].phi)


def energy_error(coarse: dict, ref: dict) -> tuple[float, float]:
    ref_sol = ref["solution"]
    ref_geom = ref["mesh"].geom
    ref_norm_sq = float(np.sum(ref_sol.epsilon * np.einsum("ij,ij->i", ref_sol.element_gradients, ref_sol.element_gradients) * ref_sol.element_areas))
    err_sq = 0.0
    centroids = ref_geom.points[ref_geom.elements].mean(axis=1)
    for e, c in enumerate(centroids):
        g_coarse = coarse_gradient_on_ref_centroid(coarse, float(c[0]), float(c[1]))
        if np.any(~np.isfinite(g_coarse)):
            continue
        diff = ref_sol.element_gradients[e] - g_coarse
        err_sq += float(ref_sol.epsilon[e] * np.dot(diff, diff) * ref_sol.element_areas[e])
    return math.sqrt(err_sq), math.sqrt(err_sq / ref_norm_sq)


def transfer_nodal_check(coarse: dict, ref: dict) -> float:
    # Nested grid: shared coarse nodes should evaluate exactly from the coarse vector.
    coarse_pts = {tuple(np.round(p, 15)): v for p, v in zip(coarse["mesh"].geom.points, coarse["solution"].phi)}
    diffs = []
    for p, v in zip(ref["mesh"].geom.points, ref["solution"].phi):
        key = tuple(np.round(p, 15))
        if key in coarse_pts:
            diffs.append(abs(coarse_pts[key] - coarse["solution"].phi[np.argmin(np.linalg.norm(coarse["mesh"].geom.points - p, axis=1))]))
    return float(max(diffs) if diffs else 0.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def fit_slope(rows: list[dict], fit_levels: list[int], error_key: str = "relative_energy_error") -> tuple[float, float]:
    data = [r for r in rows if int(r["level"]) in fit_levels]
    x = np.log([float(r["h_max_m"]) for r in data])
    y = np.log([float(r[error_key]) for r in data])
    coeff = np.polyfit(x, y, 1)
    residual = float(np.max(np.abs(y - np.polyval(coeff, x))))
    return float(coeff[0]), residual


def plot_outputs(root: Path, conv_rows: list[dict], stability_rows: list[dict], solutions: list[dict], slope: float, fit_levels: list[int]) -> None:
    fig_dir = root / "figures"; fig_dir.mkdir(exist_ok=True)
    hs = np.array([float(r["h_max_m"]) for r in conv_rows])
    errs = np.array([float(r["relative_energy_error"]) for r in conv_rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(hs, errs, "o-", label="measured relative energy error")
    fit_rows = [r for r in conv_rows if int(r["level"]) in fit_levels]
    fit_h = np.array([float(r["h_max_m"]) for r in fit_rows])
    fit_e = float(fit_rows[0]["relative_energy_error"]) * (fit_h / float(fit_rows[0]["h_max_m"])) ** slope
    ax.loglog(fit_h, fit_e, "--", label=f"fit slope {slope:.2f}")
    ax.loglog(hs, errs[0]*(hs/hs[0]), ":", label="smooth P1 energy slope-1 reference")
    ax.set_xlabel("h_max (m)"); ax.set_ylabel("relative energy-norm error")
    ax.set_title("Uniform-refinement energy convergence")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir/"uniform_energy_convergence.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    stab_analysis = [r for r in stability_rows if str(r["analysis_mesh_id"]).startswith("M")]
    stab_h = np.array([float(r["h_max_m"]) for r in stab_analysis])
    for key, label in [("error_using_R0", "reference R0"), ("error_using_R1", "reference R1"), ("error_using_R2", "reference R2")]:
        vals = np.array([float(r[key]) for r in stab_analysis])
        ax.loglog(stab_h, vals, "o-", label=label)
    ax.set_xlabel("h_max (m)")
    ax.set_ylabel("relative energy-norm error")
    ax.set_title("Reference-solution stability")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir/"reference_stability.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    for ax, sol in zip(axs.ravel(), solutions[:4]):
        g = sol["mesh"].geom
        ax.triplot(g.points[:,0]*1e3, g.points[:,1]*1e3, g.elements, lw=0.25, color="0.25")
        ax.set_title(f"M{sol['level']}")
        ax.set_xlim(-2.2,2.2); ax.set_ylim(1.25,2.05); ax.set_aspect("equal")
    fig.suptitle("Uniform mesh sequence near trace")
    fig.tight_layout(); fig.savefig(fig_dir/"uniform_mesh_sequence.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2,4.2))
    ax.plot(hs, [float(r["CE_F_per_m"]) for r in conv_rows], "o-")
    ax.set_xlabel("h_max (m)"); ax.set_ylabel("CE (F/m)")
    ax.set_title("Uniform capacitance convergence")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(fig_dir/"uniform_capacitance_convergence.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2,4.2))
    ax.loglog(hs, [float(r["max_E_V_per_m"]) for r in conv_rows], "o-")
    ax.set_xlabel("h_max (m)"); ax.set_ylabel("max |E_h| (V/m)")
    ax.set_title("Corner peak field under uniform refinement")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout(); fig.savefig(fig_dir/"corner_max_field_vs_h.png", dpi=180); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    for ax, sol in zip(axs.ravel(), solutions[:4]):
        vals = np.linalg.norm(sol["solution"].element_gradients, axis=1)
        g = sol["mesh"].geom
        im = ax.tripcolor(g.points[:,0]*1e3, g.points[:,1]*1e3, g.elements, facecolors=vals, shading="flat")
        ax.set_title(f"M{sol['level']}")
        ax.set_xlim(-1.85,-1.1); ax.set_ylim(1.42,1.82); ax.set_aspect("equal")
    fig.colorbar(im, ax=axs.ravel().tolist(), label="V/m")
    fig.suptitle("Lower trace-corner field under uniform refinement")
    fig.savefig(fig_dir/"corner_field_refinement.png", dpi=180); plt.close(fig)


def run_uniform_convergence(root: Path, quick: bool = False) -> dict:
    analysis_levels = [0, 1] if quick else [0, 1, 2, 3]
    r0_level = 2 if quick else 4
    r1_level = 3 if quick else 5
    r2_level = 4 if quick else 6
    estimate_mesh = build_uniform_mesh(r2_level)
    estimated_elements = int(estimate_mesh.geom.elements.shape[0])
    estimated_dofs = int(estimate_mesh.geom.points.shape[0])
    estimated_free = estimated_dofs - int(len(set(estimate_mesh.geom.trace_nodes.tolist()) | set(estimate_mesh.geom.ground_nodes.tolist())))
    estimated_nnz = int(7 * estimated_free)
    estimated_memory_mb = float((estimated_nnz * (8 + 4) + estimated_dofs * 8 * 10) / 1024**2)
    levels = sorted(set(analysis_levels + [r0_level, r1_level, r2_level]))
    solutions = [solve_uniform(level) for level in levels]
    by_level = {s["level"]: s for s in solutions}
    rows = []
    for level in analysis_levels:
        sol = by_level[level]
        err_abs_r2, err_rel_r2 = energy_error(sol, by_level[r2_level])
        err_abs_r1, err_rel_r1 = energy_error(sol, by_level[r1_level])
        err_abs_r0, err_rel_r0 = energy_error(sol, by_level[r0_level])
        sens_r0_r1 = abs(err_rel_r1 - err_rel_r0) / max(abs(err_rel_r1), 1e-300)
        sens_r1_r2 = abs(err_rel_r2 - err_rel_r1) / max(abs(err_rel_r2), 1e-300)
        row = {
            "level": level, "mesh_id": f"M{level}", "h_max_m": sol["h_max"], "h_min_m": sol["h_min"],
            "elements": sol["mesh"].geom.elements.shape[0], "nodes": sol["mesh"].geom.points.shape[0],
            "DOFs": sol["mesh"].geom.points.shape[0], "free_DOFs": sol["solution"].free_dofs.size,
            "NNZ": sol["solution"].matrix_free.nnz, "assembly_time_s": sol["solution"].assembly_time,
            "solve_time_s": sol["solution"].solve_time, "CE_F_per_m": sol["cap"].CE, "CQ_F_per_m": sol["cap"].CQ,
            "delta_EQ": sol["cap"].delta_EQ, "absolute_energy_error": err_abs_r2, "relative_energy_error": err_rel_r2,
            "reference_id": f"R2=M{r2_level}", "reference_sensitivity": sens_r1_r2,
            "error_using_R1": err_rel_r1, "sensitivity_R0_to_R1": sens_r0_r1, "sensitivity_R1_to_R2": sens_r1_r2,
            "max_E_V_per_m": sol["max_E"], "max_E_location_x_m": sol["max_E_location"][0],
            "max_E_location_y_m": sol["max_E_location"][1],
            "scaled_condition_estimate_if_available": np.nan,
            "residual_inf": sol["residual_inf"], "normalized_residual": sol["normalized_residual"],
            "CE_change_to_next": np.nan, "observed_order_to_next": np.nan,
        }
        rows.append(row)
    for i in range(len(rows)-1):
        rows[i]["CE_change_to_next"] = abs(rows[i]["CE_F_per_m"] - rows[i+1]["CE_F_per_m"]) / abs(rows[i+1]["CE_F_per_m"])
        rows[i]["observed_order_to_next"] = math.log(rows[i]["relative_energy_error"]/rows[i+1]["relative_energy_error"]) / math.log(rows[i]["h_max_m"]/rows[i+1]["h_max_m"])
    candidate_windows = []
    possible = [[0, 1, 2], [1, 2, 3], [0, 1, 2, 3]]
    for window in possible:
        if not all(level in analysis_levels for level in window):
            continue
        data = [r for r in rows if int(r["level"]) in window]
        monotone = all(float(data[i]["relative_energy_error"]) > float(data[i + 1]["relative_energy_error"]) for i in range(len(data) - 1))
        max_sens = max(float(r["reference_sensitivity"]) for r in data)
        slope_r2, dev_r2 = fit_slope(rows, window, "relative_energy_error")
        slope_r1, _ = fit_slope(rows, window, "error_using_R1")
        slope_abs_change = abs(slope_r2 - slope_r1)
        slope_rel_change = slope_abs_change / max(abs(slope_r2), 1e-300)
        candidate_windows.append({
            "window": window,
            "monotone": monotone,
            "max_sensitivity": max_sens,
            "slope_r1": slope_r1,
            "slope_r2": slope_r2,
            "fit_dev": dev_r2,
            "slope_abs_change": slope_abs_change,
            "slope_rel_change": slope_rel_change,
        })
    stable_candidates = [c for c in candidate_windows if c["monotone"] and c["max_sensitivity"] <= 0.03 and c["slope_rel_change"] <= 0.05]
    if stable_candidates:
        chosen = max(stable_candidates, key=lambda c: (len(c["window"]), -c["fit_dev"]))
    else:
        chosen = min(candidate_windows, key=lambda c: (c["max_sensitivity"], c["slope_rel_change"])) if candidate_windows else {
            "window": analysis_levels,
            "slope_r1": float("nan"),
            "slope_r2": float("nan"),
            "fit_dev": float("nan"),
            "slope_abs_change": float("nan"),
            "slope_rel_change": float("nan"),
            "max_sensitivity": float("inf"),
        }
    fit_levels = chosen["window"]
    slope = chosen["slope_r2"]
    fit_dev = chosen["fit_dev"]
    write_csv(root/"tables/uniform_convergence.csv", rows)
    ref_self_abs, ref_self_rel = energy_error(by_level[r0_level], by_level[r1_level])
    ref2_self_abs, ref2_self_rel = energy_error(by_level[r1_level], by_level[r2_level])
    stability = []
    for r in rows:
        level = int(r["level"])
        e0 = energy_error(by_level[level], by_level[r0_level])[1]
        e1 = float(r["error_using_R1"])
        e2 = float(r["relative_energy_error"])
        stability.append({
            "analysis_mesh_id": f"M{level}",
            "analysis_DOFs": r["DOFs"],
            "h_max_m": r["h_max_m"],
            "error_using_R0": e0,
            "error_using_R1": e1,
            "error_using_R2": e2,
            "sensitivity_R0_to_R1": abs(e1 - e0) / max(abs(e1), 1e-300),
            "sensitivity_R1_to_R2": abs(e2 - e1) / max(abs(e2), 1e-300),
        })
    stability.append({"analysis_mesh_id": "R0_summary", "analysis_DOFs": by_level[r0_level]["mesh"].geom.points.shape[0], "h_max_m": by_level[r0_level]["h_max"], "error_using_R0": np.nan, "error_using_R1": np.nan, "error_using_R2": np.nan, "sensitivity_R0_to_R1": np.nan, "sensitivity_R1_to_R2": np.nan})
    stability.append({"analysis_mesh_id": "R1_summary", "analysis_DOFs": by_level[r1_level]["mesh"].geom.points.shape[0], "h_max_m": by_level[r1_level]["h_max"], "error_using_R0": ref_self_abs, "error_using_R1": ref_self_rel, "error_using_R2": np.nan, "sensitivity_R0_to_R1": np.nan, "sensitivity_R1_to_R2": np.nan})
    stability.append({"analysis_mesh_id": "R2_summary", "analysis_DOFs": by_level[r2_level]["mesh"].geom.points.shape[0], "h_max_m": by_level[r2_level]["h_max"], "error_using_R0": np.nan, "error_using_R1": ref2_self_abs, "error_using_R2": ref2_self_rel, "sensitivity_R0_to_R1": np.nan, "sensitivity_R1_to_R2": np.nan})
    write_csv(root/"tables/reference_stability.csv", stability)
    plot_outputs(root, rows, stability, [by_level[l] for l in analysis_levels], slope, fit_levels)
    max_fit_reference_sensitivity = max(float(r["reference_sensitivity"]) for r in rows if int(r["level"]) in fit_levels)
    slope_rel_change = float(chosen["slope_rel_change"])
    gate_pass = (
        (not quick)
        and len(rows) >= 4
        and all(rows[i]["relative_energy_error"] > rows[i+1]["relative_energy_error"] for i in range(len(rows)-1))
        and np.isfinite(slope)
        and max_fit_reference_sensitivity <= 0.03
        and slope_rel_change <= 0.05
    )
    candidate_text = "\n".join(
        f"- {c['window']}: slope_R1={c['slope_r1']:.6f}, slope_R2={c['slope_r2']:.6f}, "
        f"max_sensitivity={c['max_sensitivity']:.6e}, slope_rel_change={c['slope_rel_change']:.6e}, "
        f"max_log_fit_deviation={c['fit_dev']:.3e}, monotone={c['monotone']}"
        for c in candidate_windows
    )
    log = (
        "Gate D uniform mesh convergence\n\n"
        "Fixed domain: W/H=8, Hair/H=8; homogeneous Neumann far boundary.\n"
        "Uniform refinement: every base grid interval is split by midpoint at each level; no local corner refinement is used.\n"
        f"Analysis levels: {analysis_levels}; R0=M{r0_level}; R1=M{r1_level}; R2=M{r2_level}.\n"
        f"R2 feasibility estimate before solve: elements={estimated_elements}, DOFs={estimated_dofs}, approximate_NNZ={estimated_nnz}, approximate_memory_MB={estimated_memory_mb:.2f}.\n"
        f"R2 actual: elements={by_level[r2_level]['mesh'].geom.elements.shape[0]}, DOFs={by_level[r2_level]['mesh'].geom.points.shape[0]}, NNZ={by_level[r2_level]['solution'].matrix_free.nnz}.\n"
        "Cross-mesh transfer: coarse P1 gradients are evaluated on the nested reference mesh elements; coefficient vectors are never subtracted directly.\n"
        f"Reference self-difference ||R1-R0||_E={ref_self_abs:.6e}, relative={ref_self_rel:.6e}.\n"
        f"Reference self-difference ||R2-R1||_E={ref2_self_abs:.6e}, relative={ref2_self_rel:.6e}.\n"
        f"Pairwise observed orders: {[r['observed_order_to_next'] for r in rows[:-1]]}\n"
        f"Candidate fit windows:\n{candidate_text}\n"
        f"Final fitted asymptotic order={slope:.6f}; fit levels={fit_levels}; max log fit deviation={fit_dev:.3e}.\n"
        f"Slope using R1 on final window={chosen['slope_r1']:.6f}; slope using R2={chosen['slope_r2']:.6f}; absolute change={chosen['slope_abs_change']:.6e}; relative change={chosen['slope_rel_change']:.6e}.\n"
        f"Reference sensitivity on fit levels: {[r['reference_sensitivity'] for r in rows if int(r['level']) in fit_levels]}.\n"
        "The final fit window is selected from monotone error reduction, reference stability, and fit stability, not from the largest-looking slope.\n"
        f"Max |E| trend: {[r['max_E_V_per_m'] for r in rows]}.\n"
        f"Gate D status: {'PASS' if gate_pass else 'NOT YET COMPLETE'}\n"
    )
    (root/"logs/gate_D_uniform.txt").write_text(log)
    return {
        "gate_d_status": "PASS" if gate_pass else "NOT YET COMPLETE",
        "fitted_slope": slope,
        "fit_levels": fit_levels,
        "reference_self_relative": ref2_self_rel,
        "r2_elements": int(by_level[r2_level]["mesh"].geom.elements.shape[0]),
        "r2_dofs": int(by_level[r2_level]["mesh"].geom.points.shape[0]),
        "r2_nnz": int(by_level[r2_level]["solution"].matrix_free.nnz),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_uniform_convergence(args.project_root, quick=args.quick))


if __name__ == "__main__":
    main()
