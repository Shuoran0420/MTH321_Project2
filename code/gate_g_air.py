"""Gate G matched all-air solve on the frozen Gate-F engineering mesh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from skfem import MeshTri

from capacitance import CapacitanceResult, extract_capacitance
from fem_solver import MicrostripSolution, solve_microstrip
from geometry import GeometryConfig, GeometryData
from materials import MaterialConfig
from uniform_convergence import solve_uniform

FINAL_MESH_ID = "uniform_M7_8H_neumann_P1_v1"
GATE_F_CE = 1.206197736939107e-10
REPLAY_RTOL = 1e-10


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_text(value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return value
    return str(value.tolist() if getattr(value, "shape", ()) == () else value)


def load_final_dielectric_row(root: Path) -> dict:
    path = root / "tables/final_dielectric_capacitance.csv"
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    if row["gate_f_status"] != "PASS":
        raise ValueError("Gate F final dielectric CSV is not marked PASS")
    if row["FINAL_MESH_ID"] != FINAL_MESH_ID:
        raise ValueError("Gate F FINAL_MESH_ID does not match Gate G expectation")
    return row


def load_frozen_geometry(root: Path) -> tuple[GeometryData, dict[str, np.ndarray], str]:
    relative_path = "data/processed/final_engineering_mesh.npz"
    path = root / relative_path
    data = np.load(path, allow_pickle=False)
    required = [
        "FINAL_MESH_ID",
        "status",
        "points",
        "elements",
        "element_regions",
        "ground_nodes",
        "trace_nodes",
        "far_nodes",
        "conductor_polygon",
        "W_over_H",
        "Hair_over_H",
        "H",
        "w",
        "t",
        "V0",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"frozen mesh artifact is missing keys: {missing}")
    if str(data["status"]) != "FINAL":
        raise ValueError("frozen mesh artifact is not marked FINAL")
    if str(data["FINAL_MESH_ID"]) != FINAL_MESH_ID:
        raise ValueError("frozen mesh artifact FINAL_MESH_ID mismatch")

    cfg = GeometryConfig(
        H=float(data["H"]),
        w=float(data["w"]),
        t=float(data["t"]),
        W_over_H=float(data["W_over_H"]),
        Hair_over_H=float(data["Hair_over_H"]),
        V0=float(data["V0"]),
    )
    if not np.isclose(cfg.W_over_H, 8.0) or not np.isclose(cfg.Hair_over_H, 8.0):
        raise ValueError("Gate G requires the frozen 8H x 8H final mesh")
    if not np.isclose(cfg.H, 1.6e-3) or not np.isclose(cfg.w, 3.0e-3) or not np.isclose(cfg.t, 0.035e-3):
        raise ValueError("frozen mesh physical dimensions disagree with Gate F")
    if not np.isclose(cfg.V0, 1.0):
        raise ValueError("frozen mesh V0 disagrees with Gate F")

    points = np.array(data["points"], dtype=float)
    elements = np.array(data["elements"], dtype=int)
    geom = GeometryData(
        config=cfg,
        mesh=MeshTri(points.T, elements.T),
        points=points,
        elements=elements,
        element_regions=np.array(data["element_regions"]),
        ground_nodes=np.array(data["ground_nodes"], dtype=int),
        trace_nodes=np.array(data["trace_nodes"], dtype=int),
        far_nodes=np.array(data["far_nodes"], dtype=int),
        conductor_polygon=np.array(data["conductor_polygon"], dtype=float),
    )
    return geom, {key: data[key] for key in data.files}, relative_path


def final_mesh_fingerprint(data: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in [
        "points",
        "elements",
        "element_regions",
        "ground_nodes",
        "trace_nodes",
        "far_nodes",
        "conductor_polygon",
    ]:
        arr = np.ascontiguousarray(data[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps(arr.shape).encode("ascii"))
        h.update(arr.tobytes(order="C"))
    metadata = {
        "W_over_H": float(data["W_over_H"]),
        "Hair_over_H": float(data["Hair_over_H"]),
        "H": float(data["H"]),
        "w": float(data["w"]),
        "t": float(data["t"]),
        "V0": float(data["V0"]),
        "FINAL_MESH_ID": str(data["FINAL_MESH_ID"]),
    }
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _symmetry_error(phi: np.ndarray, points: np.ndarray) -> float:
    lookup = {(round(float(x), 15), round(float(y), 15)): i for i, (x, y) in enumerate(points)}
    diffs: list[float] = []
    refs: list[float] = []
    for i, (x, y) in enumerate(points):
        j = lookup.get((round(float(-x), 15), round(float(y), 15)))
        if j is not None:
            diffs.append(float(phi[i] - phi[j]))
            refs.append(float(phi[i]))
    return float(np.linalg.norm(diffs) / max(np.linalg.norm(refs), 1e-300))


def _matrix_symmetry_error(matrix: csr_matrix) -> float:
    diff = matrix - matrix.T
    return float(np.linalg.norm(diff.data) / max(np.linalg.norm(matrix.data), 1e-300))


def _solution_metrics(sol: MicrostripSolution, cap: CapacitanceResult) -> dict:
    residual = sol.matrix_free @ sol.phi[sol.free_dofs] - sol.rhs_free
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    system_scale = float(np.linalg.norm(sol.matrix_free @ sol.phi[sol.free_dofs], ord=np.inf) + np.linalg.norm(sol.rhs_free, ord=np.inf))
    return {
        "elements": sol.geometry.elements.shape[0],
        "DOFs": sol.geometry.points.shape[0],
        "free_DOFs": sol.free_dofs.size,
        "NNZ": sol.matrix_free.nnz,
        "CE": cap.CE,
        "CQ": cap.CQ,
        "delta_EQ": cap.delta_EQ,
        "We": cap.energy,
        "Q_trace": cap.Q_trace,
        "Q_ground": cap.Q_ground,
        "charge_balance": cap.charge_balance_defect,
        "min_phi": float(np.min(sol.phi)),
        "max_phi": float(np.max(sol.phi)),
        "symmetry_error": _symmetry_error(sol.phi, sol.geometry.points),
        "matrix_symmetry_error": _matrix_symmetry_error(sol.matrix_free),
        "residual_inf": residual_inf,
        "normalized_residual": residual_inf / max(system_scale, 1e-300),
        "assembly_time": sol.assembly_time,
        "solve_time": sol.solve_time,
    }


def solve_case(geom: GeometryData, material_case: str) -> tuple[MicrostripSolution, CapacitanceResult, dict]:
    if material_case == "dielectric":
        mat = MaterialConfig()
    elif material_case == "air":
        mat = MaterialConfig(epsilon_r_substrate=1.0, epsilon_r_air=1.0)
    else:
        raise ValueError(f"unknown material case: {material_case}")
    sol = solve_microstrip(geom, mat_cfg=mat, compute_conditioning=False)
    cap = extract_capacitance(sol)
    return sol, cap, _solution_metrics(sol, cap)


def _same_pattern(a: csr_matrix, b: csr_matrix) -> bool:
    return bool(np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices))


def _row_for_case(mesh_id: str, material_case: str, sol: MicrostripSolution, cap: CapacitanceResult, metrics: dict, fingerprint: str | None = None) -> dict:
    return {
        "mesh_id": mesh_id,
        "material_case": material_case,
        "FINAL_MESH_ID": FINAL_MESH_ID if mesh_id == "M7" else "",
        "FINAL_MESH_SHA256": fingerprint if mesh_id == "M7" else "",
        **metrics,
    }


def _plot_air_outputs(root: Path, air_sol: MicrostripSolution, dielectric_sol: MicrostripSolution, eps_rows: list[dict]) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    geom = air_sol.geometry
    cfg = geom.config

    fig, ax = plt.subplots(figsize=(8, 4.8))
    contour = ax.tricontourf(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, air_sol.phi, levels=30)
    ax.plot(geom.conductor_polygon[:, 0] * 1e3, geom.conductor_polygon[:, 1] * 1e3, color="white", lw=1.0)
    ax.axhline(cfg.H * 1e3, color="white", lw=0.9, ls="--", label="former substrate-interface mesh line")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("All-air potential on frozen final mesh")
    ax.legend(fontsize=8)
    fig.colorbar(contour, ax=ax, label="V")
    fig.tight_layout()
    fig.savefig(fig_dir / "air_potential.png", dpi=180)
    plt.close(fig)

    e_air = np.linalg.norm(air_sol.element_gradients, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    artist = ax.tripcolor(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, facecolors=e_air, shading="flat")
    ax.plot(geom.conductor_polygon[:, 0] * 1e3, geom.conductor_polygon[:, 1] * 1e3, color="crimson", lw=1.0)
    ax.axhline(cfg.H * 1e3, color="white", lw=0.9, ls="--")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("All-air electric-field magnitude |E|")
    fig.colorbar(artist, ax=ax, label="V/m")
    fig.tight_layout()
    fig.savefig(fig_dir / "air_electric_field.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r["DOFs"] for r in eps_rows], [r["epsilon_eff"] for r in eps_rows], "o-")
    for r in eps_rows:
        ax.annotate(r["mesh_id"], (r["DOFs"], r["epsilon_eff"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("DOFs")
    ax.set_ylabel("effective relative permittivity")
    ax.set_title("Matched dielectric/all-air epsilon_eff convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "epsilon_eff_convergence.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, sol, title in [(axs[0], dielectric_sol, "dielectric"), (axs[1], air_sol, "all air")]:
        c = ax.tricontourf(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, sol.phi, levels=24, vmin=0.0, vmax=1.0)
        ax.plot(geom.conductor_polygon[:, 0] * 1e3, geom.conductor_polygon[:, 1] * 1e3, color="white", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(0.0, 3.0)
    axs[0].set_ylabel("y (mm)")
    fig.colorbar(c, ax=axs, label="V")
    fig.suptitle("Matched potential comparison near trace")
    fig.savefig(fig_dir / "dielectric_vs_air_potential.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_gate_g(root: Path, quick: bool = False) -> dict:
    gate_f_row = load_final_dielectric_row(root)
    frozen_geom, frozen_data, mesh_path = load_frozen_geometry(root)
    fingerprint = final_mesh_fingerprint(frozen_data)

    dielectric_sol, dielectric_cap, dielectric_metrics = solve_case(frozen_geom, "dielectric")
    gate_f_ce = float(gate_f_row["CE"])
    replay_rel = abs(dielectric_cap.CE - gate_f_ce) / abs(gate_f_ce)
    if replay_rel > REPLAY_RTOL:
        raise RuntimeError(f"dielectric replay mismatch: {replay_rel:.6e}")

    air_sol, air_cap, air_metrics = solve_case(frozen_geom, "air")
    same_mesh = np.array_equal(dielectric_sol.geometry.points, air_sol.geometry.points) and np.array_equal(dielectric_sol.geometry.elements, air_sol.geometry.elements)
    same_dofs = dielectric_sol.phi.size == air_sol.phi.size and np.array_equal(dielectric_sol.free_dofs, air_sol.free_dofs)
    same_boundaries = (
        np.array_equal(dielectric_sol.trace_dofs, air_sol.trace_dofs)
        and np.array_equal(dielectric_sol.ground_dofs, air_sol.ground_dofs)
        and np.array_equal(dielectric_sol.geometry.far_nodes, air_sol.geometry.far_nodes)
    )
    same_sparsity = (
        dielectric_sol.matrix_free.shape == air_sol.matrix_free.shape
        and dielectric_sol.matrix_free.nnz == air_sol.matrix_free.nnz
        and _same_pattern(dielectric_sol.matrix_free, air_sol.matrix_free)
    )
    physical_ordering = gate_f_ce > air_cap.CE > 0.0
    epsilon_eff = gate_f_ce / air_cap.CE
    physical_range = 1.0 < epsilon_eff < 4.4

    pair_levels = [5, 6]
    pair_rows: list[dict] = []
    eps_rows: list[dict] = []
    for level in pair_levels:
        solved = solve_uniform(level)
        geom = solved["mesh"].geom
        d_sol, d_cap, d_metrics = solve_case(geom, "dielectric")
        a_sol, a_cap, a_metrics = solve_case(geom, "air")
        pair_rows.append(_row_for_case(f"M{level}", "dielectric", d_sol, d_cap, d_metrics))
        pair_rows.append(_row_for_case(f"M{level}", "air", a_sol, a_cap, a_metrics))
        eps_rows.append({
            "mesh_id": f"M{level}",
            "elements": geom.elements.shape[0],
            "DOFs": geom.points.shape[0],
            "NNZ": d_sol.matrix_free.nnz,
            "C_dielectric": d_cap.CE,
            "C_air": a_cap.CE,
            "epsilon_eff": d_cap.CE / a_cap.CE,
            "relative_change_from_previous": np.nan,
            "dielectric_delta_EQ": d_cap.delta_EQ,
            "air_delta_EQ": a_cap.delta_EQ,
            "dielectric_charge_balance": d_cap.charge_balance_defect,
            "air_charge_balance": a_cap.charge_balance_defect,
            "same_mesh_pair_PASS": np.array_equal(d_sol.geometry.points, a_sol.geometry.points) and np.array_equal(d_sol.geometry.elements, a_sol.geometry.elements),
        })

    pair_rows.append(_row_for_case("M7", "dielectric", dielectric_sol, dielectric_cap, dielectric_metrics, fingerprint))
    pair_rows.append(_row_for_case("M7", "air", air_sol, air_cap, air_metrics, fingerprint))
    eps_rows.append({
        "mesh_id": "M7",
        "elements": frozen_geom.elements.shape[0],
        "DOFs": frozen_geom.points.shape[0],
        "NNZ": dielectric_sol.matrix_free.nnz,
        "C_dielectric": gate_f_ce,
        "C_air": air_cap.CE,
        "epsilon_eff": epsilon_eff,
        "relative_change_from_previous": np.nan,
        "dielectric_delta_EQ": dielectric_cap.delta_EQ,
        "air_delta_EQ": air_cap.delta_EQ,
        "dielectric_charge_balance": dielectric_cap.charge_balance_defect,
        "air_charge_balance": air_cap.charge_balance_defect,
        "same_mesh_pair_PASS": same_mesh,
    })
    for i in range(1, len(eps_rows)):
        eps_rows[i]["relative_change_from_previous"] = abs(eps_rows[i]["epsilon_eff"] - eps_rows[i - 1]["epsilon_eff"]) / abs(eps_rows[i]["epsilon_eff"])

    checks = {
        "same_mesh_PASS": same_mesh,
        "same_DOFs_PASS": same_dofs,
        "same_boundary_sets_PASS": same_boundaries,
        "same_sparsity_pattern_PASS": same_sparsity,
        "physical_ordering_PASS": physical_ordering,
        "physical_range_PASS": physical_range,
        "air_bounds_PASS": air_metrics["min_phi"] >= -1e-10 and air_metrics["max_phi"] <= 1.0 + 1e-10,
        "air_delta_EQ_PASS": air_cap.delta_EQ < 1e-9,
        "air_charge_balance_PASS": air_cap.charge_balance_defect < 1e-9,
        "air_residual_PASS": air_metrics["normalized_residual"] < 1e-9,
        "air_symmetry_PASS": air_metrics["symmetry_error"] < 1e-9,
        "dielectric_replay_PASS": replay_rel <= REPLAY_RTOL,
    }
    gate_g_pass = all(checks.values())

    write_csv(root / "tables/matched_dielectric_air.csv", pair_rows)
    write_csv(root / "tables/epsilon_eff_convergence.csv", eps_rows)
    write_csv(root / "tables/final_air_capacitance.csv", [{
        "FINAL_MESH_ID": FINAL_MESH_ID,
        "FINAL_MESH_SHA256": fingerprint,
        "material_case": "all_air",
        "elements": frozen_geom.elements.shape[0],
        "DOFs": frozen_geom.points.shape[0],
        "NNZ": air_sol.matrix_free.nnz,
        "CE_air": air_cap.CE,
        "CQ_air": air_cap.CQ,
        "delta_EQ_air": air_cap.delta_EQ,
        "We_air": air_cap.energy,
        "Q_trace_air": air_cap.Q_trace,
        "Q_ground_air": air_cap.Q_ground,
        "charge_balance_air": air_cap.charge_balance_defect,
        "min_phi": air_metrics["min_phi"],
        "max_phi": air_metrics["max_phi"],
        "symmetry_error": air_metrics["symmetry_error"],
        "residual": air_metrics["residual_inf"],
        "status": "FINAL_MATCHED_AIR" if gate_g_pass else "NOT_YET_COMPLETE",
    }])
    write_csv(root / "tables/final_effective_permittivity.csv", [{
        "FINAL_MESH_ID": FINAL_MESH_ID,
        "FINAL_MESH_SHA256": fingerprint,
        "C_dielectric": gate_f_ce,
        "C_air": air_cap.CE,
        "epsilon_eff": epsilon_eff,
        "dielectric_replay_CE": dielectric_cap.CE,
        "dielectric_replay_relative_difference": replay_rel,
        "dielectric_CQ": dielectric_cap.CQ,
        "air_CQ": air_cap.CQ,
        "dielectric_delta_EQ": dielectric_cap.delta_EQ,
        "air_delta_EQ": air_cap.delta_EQ,
        "dielectric_charge_balance": dielectric_cap.charge_balance_defect,
        "air_charge_balance": air_cap.charge_balance_defect,
        "dielectric_residual": dielectric_metrics["residual_inf"],
        "air_residual": air_metrics["residual_inf"],
        "dielectric_symmetry": dielectric_metrics["symmetry_error"],
        "air_symmetry": air_metrics["symmetry_error"],
        **checks,
        "epsilon_eff_M5": eps_rows[0]["epsilon_eff"],
        "epsilon_eff_M6": eps_rows[1]["epsilon_eff"],
        "epsilon_eff_M7": eps_rows[2]["epsilon_eff"],
        "epsilon_eff_M6_change": eps_rows[1]["relative_change_from_previous"],
        "epsilon_eff_M7_change": eps_rows[2]["relative_change_from_previous"],
        "Gate_G_status": "PASS" if gate_g_pass else "NOT YET COMPLETE",
    }])

    _plot_air_outputs(root, air_sol, dielectric_sol, eps_rows)

    (root / "logs/gate_G_mesh_identity.txt").write_text(
        "Gate G frozen mesh identity checks\n\n"
        f"FINAL_MESH_ID: {FINAL_MESH_ID}\n"
        f"FINAL_MESH_SHA256: {fingerprint}\n"
        f"mesh_artifact: {mesh_path}\n"
        f"points shape: {frozen_geom.points.shape}\n"
        f"elements shape: {frozen_geom.elements.shape}\n"
        f"exact_points_equal: {np.array_equal(dielectric_sol.geometry.points, air_sol.geometry.points)}\n"
        f"exact_elements_equal: {np.array_equal(dielectric_sol.geometry.elements, air_sol.geometry.elements)}\n"
        f"trace_nodes_equal: {np.array_equal(dielectric_sol.trace_dofs, air_sol.trace_dofs)}\n"
        f"ground_nodes_equal: {np.array_equal(dielectric_sol.ground_dofs, air_sol.ground_dofs)}\n"
        f"far_nodes_equal: {np.array_equal(dielectric_sol.geometry.far_nodes, air_sol.geometry.far_nodes)}\n"
        f"total_DOFs_equal: {dielectric_sol.phi.size == air_sol.phi.size}\n"
        f"free_DOFs_equal: {np.array_equal(dielectric_sol.free_dofs, air_sol.free_dofs)}\n"
        f"matrix_shape_equal: {dielectric_sol.matrix_free.shape == air_sol.matrix_free.shape}\n"
        f"matrix_NNZ_equal: {dielectric_sol.matrix_free.nnz == air_sol.matrix_free.nnz}\n"
        f"matrix_indptr_equal: {np.array_equal(dielectric_sol.matrix_free.indptr, air_sol.matrix_free.indptr)}\n"
        f"matrix_indices_equal: {np.array_equal(dielectric_sol.matrix_free.indices, air_sol.matrix_free.indices)}\n"
        f"MATCHED_MESH_IDENTITY: {'PASS' if same_mesh and same_dofs and same_boundaries and same_sparsity else 'FAIL'}\n"
    )
    (root / "logs/gate_G_air.txt").write_text(
        "Gate G matched all-air solve and effective permittivity\n\n"
        f"Gate F status: {gate_f_row['gate_f_status']}; acceptance_path={gate_f_row['acceptance_path']}\n"
        f"FINAL_MESH_ID: {FINAL_MESH_ID}\n"
        f"FINAL_MESH_SHA256: {fingerprint}\n"
        f"dielectric replay CE: {dielectric_cap.CE:.16e}\n"
        f"authoritative Gate-F CE: {gate_f_ce:.16e}\n"
        f"dielectric replay relative difference: {replay_rel:.16e}\n"
        f"C_dielectric: {gate_f_ce:.16e}\n"
        f"C_air: {air_cap.CE:.16e}\n"
        f"epsilon_eff: {epsilon_eff:.16e}\n"
        f"M5 epsilon_eff: {eps_rows[0]['epsilon_eff']:.16e}\n"
        f"M6 epsilon_eff: {eps_rows[1]['epsilon_eff']:.16e}\n"
        f"M7 epsilon_eff: {eps_rows[2]['epsilon_eff']:.16e}\n"
        f"successive epsilon_eff changes: M5->M6={eps_rows[1]['relative_change_from_previous']:.16e}, M6->M7={eps_rows[2]['relative_change_from_previous']:.16e}\n"
        f"air CE/CQ consistency delta_EQ: {air_cap.delta_EQ:.16e}\n"
        f"air charge-balance defect: {air_cap.charge_balance_defect:.16e}\n"
        f"air residual inf: {air_metrics['residual_inf']:.16e}; normalized={air_metrics['normalized_residual']:.16e}\n"
        f"air symmetry error: {air_metrics['symmetry_error']:.16e}\n"
        f"same-mesh result: {same_mesh}\n"
        f"same-DOF result: {same_dofs}\n"
        f"same-boundary result: {same_boundaries}\n"
        f"same-sparsity result: {same_sparsity}\n"
        f"physical ordering result: {physical_ordering}\n"
        f"physical epsilon_eff range result: {physical_range}\n"
        "Limitations: Gate F used the M7 asymptotic/resource fallback; epsilon_eff is a matched FEM ratio, not an exact material constant.\n"
        f"Gate G status: {'PASS' if gate_g_pass else 'NOT YET COMPLETE'}\n"
    )
    return {
        "gate_g_status": "PASS" if gate_g_pass else "NOT YET COMPLETE",
        "FINAL_MESH_ID": FINAL_MESH_ID,
        "FINAL_MESH_SHA256": fingerprint,
        "elements": frozen_geom.elements.shape[0],
        "DOFs": frozen_geom.points.shape[0],
        "NNZ": dielectric_sol.matrix_free.nnz,
        "CE_replay": dielectric_cap.CE,
        "replay_relative_difference": replay_rel,
        "C_dielectric": gate_f_ce,
        "C_air": air_cap.CE,
        "epsilon_eff": epsilon_eff,
        "epsilon_eff_M5": eps_rows[0]["epsilon_eff"],
        "epsilon_eff_M6": eps_rows[1]["epsilon_eff"],
        "epsilon_eff_M7": eps_rows[2]["epsilon_eff"],
        "epsilon_eff_M6_change": eps_rows[1]["relative_change_from_previous"],
        "epsilon_eff_M7_change": eps_rows[2]["relative_change_from_previous"],
        "air_CE": air_cap.CE,
        "air_CQ": air_cap.CQ,
        "air_delta_EQ": air_cap.delta_EQ,
        "air_Q_trace": air_cap.Q_trace,
        "air_Q_ground": air_cap.Q_ground,
        "air_charge_balance": air_cap.charge_balance_defect,
        "air_residual": air_metrics["residual_inf"],
        "air_symmetry_error": air_metrics["symmetry_error"],
        "air_min_phi": air_metrics["min_phi"],
        "air_max_phi": air_metrics["max_phi"],
        **checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_gate_g(args.project_root, quick=args.quick))


if __name__ == "__main__":
    main()
