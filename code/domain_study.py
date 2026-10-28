"""Gate C controlled domain-truncation study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skfem import MeshTri

from baseline import gate_b_metrics
from capacitance import extract_capacitance
from fem_solver import solve_microstrip
from geometry import GeometryConfig, GeometryData, _inside_conductor, _on_trace_boundary

DOMAIN_RATIOS = [3.0, 5.0, 8.0, 12.0]
CORE_X_LIMIT = 3.2e-3
CORE_Y_LIMIT = 2.4e-3
CORE_NX = 64
CORE_NY_R0 = 72
CORE_NY_R1 = 108


@dataclass(frozen=True)
class DomainRun:
    ratio: float
    resolution_id: str
    near_y: int


def _axis_with_segments(start: float, core_min: float, core_max: float, stop: float, core_n: int, outer_step: float, required: list[float]) -> np.ndarray:
    vals = [np.linspace(core_min, core_max, core_n + 1)]
    if start < core_min:
        vals.append(np.arange(core_min - outer_step, start - 0.5 * outer_step, -outer_step))
    if stop > core_max:
        vals.append(np.arange(core_max + outer_step, stop + 0.5 * outer_step, outer_step))
    arr = np.unique(np.round(np.concatenate(vals + [np.array(required, dtype=float), np.array([start, stop])]), 15))
    arr = arr[(arr >= start - 1e-15) & (arr <= stop + 1e-15)]
    arr.sort()
    return arr


def build_domain_geometry(ratio: float, resolution_id: str = "R0") -> GeometryData:
    ny_core = CORE_NY_R1 if resolution_id == "R1" else CORE_NY_R0
    cfg = GeometryConfig(W_over_H=ratio, Hair_over_H=ratio)
    outer_step = 0.32e-3 if resolution_id == "R0" else 0.28e-3
    xs = _axis_with_segments(-cfg.W, -CORE_X_LIMIT, CORE_X_LIMIT, cfg.W, CORE_NX, outer_step, [0.0, cfg.trace_xmin, cfg.trace_xmax])
    ys = _axis_with_segments(0.0, 0.0, CORE_Y_LIMIT, cfg.Hair, ny_core, outer_step, [cfg.H, cfg.trace_ymax])
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points_all = np.column_stack([xx.ravel(), yy.ravel()])
    node_id = np.arange(points_all.shape[0]).reshape(len(ys), len(xs))
    tris: list[list[int]] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            quad = [node_id[j, i], node_id[j, i + 1], node_id[j + 1, i], node_id[j + 1, i + 1]]
            cx = float(points_all[quad, 0].mean())
            cy = float(points_all[quad, 1].mean())
            if _inside_conductor(np.array([cx]), np.array([cy]), cfg)[0]:
                continue
            tris.append([quad[0], quad[1], quad[3]])
            tris.append([quad[0], quad[3], quad[2]])
    used = np.unique(np.array(tris, dtype=int).ravel())
    remap = -np.ones(points_all.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    points = points_all[used]
    elements = remap[np.array(tris, dtype=int)]
    mesh = MeshTri(points.T, elements.T)
    centroids = points[elements].mean(axis=1)
    element_regions = np.where(centroids[:, 1] < cfg.H - 1e-13, "substrate", "air")
    x = points[:, 0]
    y = points[:, 1]
    ground_nodes = np.flatnonzero(np.isclose(y, 0.0, atol=1e-12))
    trace_nodes = np.flatnonzero(_on_trace_boundary(x, y, cfg))
    far_nodes = np.flatnonzero(
        (
            np.isclose(x, -cfg.W, atol=1e-12)
            | np.isclose(x, cfg.W, atol=1e-12)
            | np.isclose(y, cfg.Hair, atol=1e-12)
        )
        & ~np.isin(np.arange(points.shape[0]), ground_nodes)
        & ~np.isin(np.arange(points.shape[0]), trace_nodes)
    )
    conductor_polygon = np.array(
        [
            [cfg.trace_xmin, cfg.trace_ymin],
            [cfg.trace_xmax, cfg.trace_ymin],
            [cfg.trace_xmax, cfg.trace_ymax],
            [cfg.trace_xmin, cfg.trace_ymax],
            [cfg.trace_xmin, cfg.trace_ymin],
        ]
    )
    return GeometryData(cfg, mesh, points, elements, element_regions, ground_nodes, trace_nodes, far_nodes, conductor_polygon)


def near_field_signature(geom: GeometryData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = geom.points
    elems = geom.elements
    in_core_nodes = (np.abs(pts[:, 0]) <= CORE_X_LIMIT + 1e-14) & (pts[:, 1] <= CORE_Y_LIMIT + 1e-14)
    core_coords = np.array(sorted(map(tuple, np.round(pts[in_core_nodes], 15))))
    centroids = pts[elems].mean(axis=1)
    in_core_elems = (np.abs(centroids[:, 0]) <= CORE_X_LIMIT + 1e-14) & (centroids[:, 1] <= CORE_Y_LIMIT + 1e-14)
    elem_centroids = np.array(sorted(map(tuple, np.round(centroids[in_core_elems], 15))))
    trace_coords = np.array(sorted(map(tuple, np.round(pts[geom.trace_nodes], 15))))
    return core_coords, elem_centroids, trace_coords


def near_field_equivalence(geoms: list[GeometryData]) -> tuple[bool, str]:
    ref = near_field_signature(geoms[0])
    for geom in geoms[1:]:
        sig = near_field_signature(geom)
        for a, b, name in zip(ref, sig, ["core node coordinates", "core element pattern", "trace dof coordinates"]):
            if a.shape != b.shape or not np.array_equal(a, b):
                return False, f"Mismatch in {name} for W/H={geom.config.W_over_H}"
        if not np.isclose(geom.config.H, geoms[0].config.H) or not np.isclose(geom.config.w, geoms[0].config.w) or not np.isclose(geom.config.t, geoms[0].config.t):
            return False, "Trace geometry mismatch"
    return True, "Near-field coordinates, local element centroids, trace DOFs, and interface geometry match exactly."


def solve_domain_run(run: DomainRun) -> dict[str, float | str]:
    geom = build_domain_geometry(run.ratio, run.resolution_id)
    sol = solve_microstrip(geom)
    cap = extract_capacitance(sol)
    metrics = gate_b_metrics(sol, cap)
    return {
        "W_over_H": run.ratio,
        "Hair_over_H": run.ratio,
        "W_m": geom.config.W,
        "Hair_m": geom.config.Hair,
        "full_width_2W_m": 2.0 * geom.config.W,
        "resolution_id": run.resolution_id,
        "elements": geom.elements.shape[0],
        "nodes": geom.points.shape[0],
        "DOFs": geom.points.shape[0],
        "free_DOFs": sol.free_dofs.size,
        "NNZ": sol.matrix_free.nnz,
        "assembly_time_s": sol.assembly_time,
        "solve_time_s": sol.solve_time,
        "We_J_per_m": cap.energy,
        "CE_F_per_m": cap.CE,
        "CQ_F_per_m": cap.CQ,
        "delta_EQ": cap.delta_EQ,
        "Q_trace_C_per_m": cap.Q_trace,
        "Q_ground_C_per_m": cap.Q_ground,
        "charge_balance_defect": cap.charge_balance_defect,
        "min_phi": metrics["min_phi"],
        "max_phi": metrics["max_phi"],
        "symmetry_relative_error": metrics["symmetry_relative_error"],
        "matrix_symmetry_error": metrics["matrix_symmetry_error"],
        "residual_inf": metrics["residual_inf"],
        "lambda_min": metrics["lambda_min"],
        "lambda_max": metrics["lambda_max"],
        "condition_estimate": metrics["condition_estimate"],
        "scaled_condition_estimate": metrics["scaled_condition_estimate"],
        "relative_to_C12": np.nan,
        "percent_to_C12": np.nan,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_domain_figures(root: Path, rows: list[dict]) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    ratios = [float(r["W_over_H"]) for r in rows if r["resolution_id"] == "R0"]
    rel = [float(r["relative_to_C12"]) for r in rows if r["resolution_id"] == "R0"]
    ce = [float(r["CE_F_per_m"]) for r in rows if r["resolution_id"] == "R0"]
    dofs = [float(r["DOFs"]) for r in rows if r["resolution_id"] == "R0"]
    solve = [float(r["solve_time_s"]) for r in rows if r["resolution_id"] == "R0"]
    nnz = [float(r["NNZ"]) for r in rows if r["resolution_id"] == "R0"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    max_w_mm = max(ratios) * GeometryConfig().H * 1e3
    max_h_mm = max(ratios) * GeometryConfig().H * 1e3
    for ratio in ratios:
        cfg = GeometryConfig(W_over_H=ratio, Hair_over_H=ratio)
        ax.add_patch(plt.Rectangle((-cfg.W * 1e3, 0), 2 * cfg.W * 1e3, cfg.Hair * 1e3, fill=False, lw=1.3, label=f"{ratio:g}H domain"))
    ax.add_patch(plt.Rectangle((-1.5, 1.6), 3.0, 0.035, facecolor="white", edgecolor="crimson", lw=1.5, label="common signal trace"))
    ax.annotate("2W full horizontal span", xy=(0, max_h_mm * 0.96), ha="center", fontsize=9)
    ax.set_xlim(-max_w_mm * 1.04, max_w_mm * 1.04)
    ax.set_ylim(0, max_h_mm * 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Controlled domain growth: full computational boxes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_mesh_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ratio in ratios:
        geom = build_domain_geometry(ratio, "R0")
        core = geom.points[(np.abs(geom.points[:, 0]) <= CORE_X_LIMIT + 1e-14) & (geom.points[:, 1] <= CORE_Y_LIMIT + 1e-14)]
        ax.plot(core[:, 0] * 1e3, core[:, 1] * 1e3, ".", ms=0.5, alpha=0.35, label=f"{ratio:g}H core")
    ax.add_patch(plt.Rectangle((-1.5, 1.6), 3.0, 0.035, fill=False, edgecolor="crimson", lw=1.4, label="common trace"))
    ax.axhline(1.6, color="navy", ls="--", lw=1.0, label="interface y=H")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(0, 2.55)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Near-field mesh coordinates overlay")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_near_field_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ratios, rel, "o-")
    ax.axhline(0.01, color="crimson", ls="--", label="1% threshold")
    ax.set_xlabel("W/H = Hair/H")
    ax.set_ylabel("relative difference to tested C12")
    ax.set_title("Domain truncation relative to 12H reference")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_truncation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ratios, ce, "o-")
    ax.set_xlabel("W/H = Hair/H")
    ax.set_ylabel("CE (F/m)")
    ax.set_title("Energy capacitance vs tested domain size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "capacitance_vs_domain.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    line1, = ax1.plot(dofs, solve, "o-", label="solve time")
    ax1.set_xlabel("DOFs")
    ax1.set_ylabel("solve time (s)")
    ax2 = ax1.twinx()
    line2, = ax2.plot(dofs, nnz, "s--", color="tab:orange", label="NNZ")
    ax2.set_ylabel("NNZ")
    ax1.set_title("Domain-study cost")
    ax1.grid(alpha=0.3)
    ax1.legend([line1, line2], ["solve time", "NNZ"], loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_cost.png", dpi=180)
    plt.close(fig)


def run_domain_study(root: Path, quick: bool = False) -> dict[str, str | float]:
    ratios = [3.0, 5.0] if quick else DOMAIN_RATIOS
    runs = [DomainRun(r, "R0", CORE_NY_R0) for r in ratios]
    if not quick:
        runs.append(DomainRun(12.0, "R1", CORE_NY_R1))
    rows = [solve_domain_run(run) for run in runs]
    main = [r for r in rows if r["resolution_id"] == "R0"]
    c12 = next(float(r["CE_F_per_m"]) for r in main if float(r["W_over_H"]) == 12.0) if not quick else float(main[-1]["CE_F_per_m"])
    for r in rows:
        r["relative_to_C12"] = abs(float(r["CE_F_per_m"]) - c12) / abs(c12)
        r["percent_to_C12"] = 100.0 * float(r["relative_to_C12"])
    if not quick:
        c12_r0 = next(float(r["CE_F_per_m"]) for r in rows if r["resolution_id"] == "R0" and float(r["W_over_H"]) == 12.0)
        c12_r1 = next(float(r["CE_F_per_m"]) for r in rows if r["resolution_id"] == "R1")
        mesh_sens = abs(c12_r1 - c12_r0) / abs(c12_r1)
    else:
        c12_r0 = c12_r1 = mesh_sens = float("nan")
    _write_csv(root / "tables" / "domain_truncation.csv", rows)
    if not quick:
        adequacy = [
            {**{k: r[k] for k in ["W_over_H", "resolution_id", "elements", "DOFs", "NNZ", "CE_F_per_m", "assembly_time_s", "solve_time_s"]}}
            for r in rows if float(r["W_over_H"]) == 12.0
        ]
        for r in adequacy:
            r["domain"] = "12H"
            r["difference_from_finer"] = abs(float(r["CE_F_per_m"]) - c12_r1)
            r["relative_difference"] = abs(float(r["CE_F_per_m"]) - c12_r1) / abs(c12_r1)
            r["interpretation"] = "fine reference" if r["resolution_id"] == "R1" else ("below 0.1% target" if r["relative_difference"] < 0.001 else "above 0.1% target")
        _write_csv(root / "tables" / "domain_mesh_adequacy.csv", adequacy)
        plot_domain_figures(root, rows)
    geoms = [build_domain_geometry(r, "R0") for r in DOMAIN_RATIOS]
    near_ok, near_msg = near_field_equivalence(geoms)
    production_below = [r for r in main if float(r["relative_to_C12"]) < 0.01]
    smallest = min(float(r["W_over_H"]) for r in production_below) if production_below else float("nan")
    trend_vals = [float(r["CE_F_per_m"]) for r in main]
    trend = "monotone increasing" if all(np.diff(trend_vals) > 0) else ("monotone decreasing" if all(np.diff(trend_vals) < 0) else "non-monotone")
    refined_rel = {float(r["W_over_H"]): abs(float(r["CE_F_per_m"]) - c12_r1) / abs(c12_r1) for r in main} if not quick else {}
    refined_below = [k for k, v in refined_rel.items() if v < 0.01]
    refined_smallest = min(refined_below) if refined_below else float("nan")
    conclusion_changes = bool(not quick and refined_smallest != smallest)
    gate_pass = (not quick) and near_ok and mesh_sens < 0.001 and not conclusion_changes and all(float(r["delta_EQ"]) < 1e-10 for r in rows)
    log = root / "logs" / "gate_C_domain.txt"
    log.write_text(
        "Gate C controlled domain-truncation study\n\n"
        "Artificial BC: homogeneous Neumann on top and side far boundaries for all runs.\n"
        f"Near-field core: x in [-{CORE_X_LIMIT:g}, {CORE_X_LIMIT:g}] m, y in [0, {CORE_Y_LIMIT:g}] m.\n"
        f"Preserved coordinate lines include x=-w/2, 0, w/2 and y=H, H+t; core resolution CORE_NX={CORE_NX}, CORE_NY_R0={CORE_NY_R0}.\n"
        f"Near-field equivalence: {'PASS' if near_ok else 'FAIL'} - {near_msg}\n\n"
        + "\n".join(f"C{int(r['W_over_H'])}={float(r['CE_F_per_m']):.16e} F/m, rel_to_C12={float(r['relative_to_C12']):.6e}" for r in main)
        + f"\n\nsmallest_tested_domain_below_1_percent = {smallest:g}H\n"
        + f"C12_R0={c12_r0:.16e}\nC12_R1={c12_r1:.16e}\nmesh_sensitivity={mesh_sens:.6e}\n"
        + f"R1_relative_to_production_C12={abs(c12_r1 - c12_r0) / abs(c12_r0):.6e}\n"
        + f"refined_reference_smallest_below_1_percent = {refined_smallest:g}H\nconclusion_changes_under_refined_reference = {conclusion_changes}\n"
        + f"capacitance trend = {trend}\n"
        + f"Gate C status: {'PASS' if gate_pass else 'NOT YET COMPLETE'}\n"
    )
    return {"gate_c_status": "PASS" if gate_pass else "NOT YET COMPLETE", "smallest": smallest, "mesh_sensitivity": mesh_sens, "trend": trend}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_domain_study(args.project_root, quick=args.quick))


if __name__ == "__main__":
    main()
