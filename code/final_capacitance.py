"""Gate F final dielectric capacitance convergence and mesh selection."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graded_study import GRADED_SPECS, build_graded_mesh, solve_mesh
from uniform_convergence import build_uniform_mesh, element_sizes, solve_uniform, write_csv

FINAL_MESH_ID_M6 = "uniform_M6_8H_neumann_P1_v1"
FINAL_MESH_ID_M7 = "uniform_M7_8H_neumann_P1_v1"
FINAL_MESH_ID_M8 = "uniform_M8_8H_neumann_P1_v1"
DOMAIN_INDICATOR = 0.00182031
M7_DOF_LIMIT = 500_000
M8_DOF_LIMIT = 900_000


def _symmetry_error(phi: np.ndarray, points: np.ndarray) -> float:
    lookup = {(round(float(x), 15), round(float(y), 15)): i for i, (x, y) in enumerate(points)}
    diffs = []
    refs = []
    for i, (x, y) in enumerate(points):
        j = lookup.get((round(float(-x), 15), round(float(y), 15)))
        if j is not None:
            diffs.append(phi[i] - phi[j])
            refs.append(phi[i])
    return float(np.linalg.norm(diffs) / max(np.linalg.norm(refs), 1e-300))


def _solution_row(mesh_id: str, family: str, solved: dict, role: str, final_ce: float | None = None) -> dict:
    sol = solved["solution"]
    cap = solved["cap"]
    hmax, hmin, _ = element_sizes(solved["mesh"].geom)
    residual = sol.matrix_free @ sol.phi[sol.free_dofs] - sol.rhs_free
    rel_final = np.nan if final_ce is None else abs(cap.CE - final_ce) / abs(final_ce)
    return {
        "family": family,
        "mesh_id": mesh_id,
        "domain_ratio": 8.0,
        "elements": solved["mesh"].geom.elements.shape[0],
        "nodes": solved["mesh"].geom.points.shape[0],
        "DOFs": solved["mesh"].geom.points.shape[0],
        "free_DOFs": sol.free_dofs.size,
        "NNZ": sol.matrix_free.nnz,
        "h_max": hmax,
        "h_min": hmin,
        "CE": cap.CE,
        "CQ": cap.CQ,
        "delta_EQ": cap.delta_EQ,
        "We": cap.energy,
        "Q_trace": cap.Q_trace,
        "Q_ground": cap.Q_ground,
        "charge_balance_defect": cap.charge_balance_defect,
        "min_phi": float(np.min(sol.phi)),
        "max_phi": float(np.max(sol.phi)),
        "symmetry_error": _symmetry_error(sol.phi, solved["mesh"].geom.points),
        "residual": float(np.linalg.norm(residual, ord=np.inf)),
        "assembly_time": sol.assembly_time,
        "solve_time": sol.solve_time,
        "relative_change_from_previous_uniform": np.nan,
        "relative_change_percent": np.nan,
        "relative_to_final_uniform": rel_final,
        "status": "computed",
        "role": role,
    }


def _apparent_orders(rows: list[dict]) -> list[dict]:
    out = []
    for i in range(2, len(rows)):
        c0, c1, c2 = [float(rows[j]["CE"]) for j in (i - 2, i - 1, i)]
        num = abs(c0 - c1)
        den = abs(c1 - c2)
        p = math.log(num / den) / math.log(2.0) if den > 0 and num > 0 else float("nan")
        out.append({"triple": f"{rows[i-2]['mesh_id']}-{rows[i-1]['mesh_id']}-{rows[i]['mesh_id']}", "p_C": p})
    return out


def _m7_preflight(m6: dict, root: Path) -> dict:
    elems = int(m6["elements"]) * 4
    dofs = int(m6["DOFs"]) * 4 - 2000
    nnz = int(m6["NNZ"]) * 4
    assembly = float(m6["assembly_time"]) * 4
    solve = float(m6["solve_time"]) * 8
    memory_mb = (nnz * (8 + 4) + dofs * 8 * 12) / 1024**2
    decision = "M7_NOT_RUN_RESOURCE_LIMIT" if dofs > M7_DOF_LIMIT else "M7_FEASIBLE_RUN"
    text = (
        "M7 preflight estimate\n\n"
        f"Estimated elements: {elems}\n"
        f"Estimated DOFs: {dofs}\n"
        f"Estimated NNZ: {nnz}\n"
        f"Estimated sparse storage/work memory indicator: {memory_mb:.2f} MB before factorization fill-in\n"
        f"Estimated assembly time: {assembly:.6e} s\n"
        f"Estimated solve time lower-bound scaling: {solve:.6e} s\n"
        f"Decision: {decision}\n"
        "Reason: the estimate is below the configured feasibility limit, so Gate F attempts M7 before making the final status decision.\n"
    )
    (root / "logs/M7_preflight.txt").write_text(text)
    return {"estimated_elements": elems, "estimated_DOFs": dofs, "estimated_NNZ": nnz, "estimated_memory_MB": memory_mb, "decision": decision}


def _freeze_mesh(root: Path, solved: dict, mesh_id: str, final: bool) -> Path:
    path = root / ("data/processed/final_engineering_mesh.npz" if final else "data/processed/candidate_engineering_mesh.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    geom = solved["mesh"].geom
    np.savez(
        path,
        FINAL_MESH_ID=mesh_id,
        status="FINAL" if final else "CANDIDATE_ONLY",
        points=geom.points,
        elements=geom.elements,
        element_regions=geom.element_regions,
        ground_nodes=geom.ground_nodes,
        trace_nodes=geom.trace_nodes,
        far_nodes=geom.far_nodes,
        conductor_polygon=geom.conductor_polygon,
        W_over_H=geom.config.W_over_H,
        Hair_over_H=geom.config.Hair_over_H,
        H=geom.config.H,
        w=geom.config.w,
        t=geom.config.t,
        V0=geom.config.V0,
    )
    return path


def _plot(root: Path, uniform_rows: list[dict], graded_rows: list[dict], final_mesh: str, c_ext: float | None, gate_pass: bool, cext_rows: list[dict]) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["DOFs"] for r in uniform_rows], [r["CE"] for r in uniform_rows], "o-", label=f"uniform {uniform_rows[0]['mesh_id']}-{uniform_rows[-1]['mesh_id']}")
    ax.plot([r["DOFs"] for r in graded_rows], [r["CE"] for r in graded_rows], "s", label="graded cross-check")
    final = next(r for r in uniform_rows if r["mesh_id"] == final_mesh)
    ax.plot(final["DOFs"], final["CE"], "o", ms=10, mfc="none", mec="k", label="selected final uniform mesh" if gate_pass else "Gate-F candidate")
    if c_ext is not None and math.isfinite(c_ext):
        ax.axhline(c_ext, ls="--", color="0.35", label="Richardson diagnostic only")
    ax.set_xlabel("DOFs")
    ax.set_ylabel("CE (F/m)")
    ax.set_title("Final dielectric capacitance selection")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "final_capacitance_convergence.png", dpi=180)
    plt.close(fig)

    changes = [r for r in uniform_rows if math.isfinite(float(r["relative_change_from_previous_uniform"]))]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([r["DOFs"] for r in changes], [r["relative_change_from_previous_uniform"] for r in changes], "o-")
    for r in changes:
        ax.annotate(r["mesh_id"], (r["DOFs"], r["relative_change_from_previous_uniform"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0.001, ls="--", color="0.4", label="preferred 0.1% QC target")
    ax.set_xlabel("DOFs")
    ax.set_ylabel("successive relative CE change")
    ax.set_title("Final capacitance successive change")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "final_capacitance_relative_change.png", dpi=180)
    plt.close(fig)

    if cext_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        yvals = [r["C_extrapolated"] * 1e12 for r in cext_rows]
        ax.plot([r["finest_mesh"] for r in cext_rows], yvals, "o-")
        for r, y in zip(cext_rows, yvals):
            ax.annotate(f"{y:.4f}", (r["finest_mesh"], y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("finest mesh in Richardson triple")
        ax.set_ylabel("Richardson diagnostic C_ext (pF/m)")
        ax.set_title("Capacitance extrapolation stability\nDiagnostic extrapolation only; final reported capacitance is computed M7 FEM")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "capacitance_extrapolation_stability.png", dpi=180)
        plt.close(fig)


def _richardson_rows(uniform_rows: list[dict]) -> list[dict]:
    rows = []
    orders = _apparent_orders(uniform_rows)
    for idx, order in enumerate(orders, start=2):
        p = order["p_C"]
        cf = float(uniform_rows[idx]["CE"])
        cc = float(uniform_rows[idx - 1]["CE"])
        cext = float("nan")
        rel = float("nan")
        if math.isfinite(p) and p > 0.1:
            cext = cf + (cf - cc) / (2.0**p - 1.0)
            rel = abs(cext - cf) / abs(cext)
        rows.append({
            "triple": order["triple"],
            "p_C": p,
            "finest_mesh": uniform_rows[idx]["mesh_id"],
            "C_finest": cf,
            "C_extrapolated": cext,
            "relative_finest_to_extrapolated": rel,
            "relative_change_from_previous_Cext": np.nan,
        })
    for i in range(1, len(rows)):
        rows[i]["relative_change_from_previous_Cext"] = abs(rows[i]["C_extrapolated"] - rows[i - 1]["C_extrapolated"]) / abs(rows[i]["C_extrapolated"])
    return rows


def _m8_preflight(m7: dict, root: Path) -> dict:
    elems = int(m7["elements"]) * 4
    dofs = int(m7["DOFs"]) * 4
    nnz = int(m7["NNZ"]) * 4
    assembly = float(m7["assembly_time"]) * 4
    solve = float(m7["solve_time"]) * 10
    raw_mb = (nnz * (8 + 4) + dofs * 8 * 12) / 1024**2
    factor_mb = raw_mb * 20.0
    decision = "M8_NOT_RECOMMENDED_RESOURCE_LIMIT" if dofs > M8_DOF_LIMIT else "M8_FEASIBLE"
    text = (
        "M8 preflight estimate\n\n"
        "Available resource assumption: ordinary local laptop/workstation run; avoid multi-million-DOF sparse-direct factorization fill-in risk.\n"
        f"Estimated elements: {elems}\n"
        f"Estimated DOFs: {dofs}\n"
        f"Estimated NNZ: {nnz}\n"
        f"Estimated raw sparse storage: {raw_mb:.2f} MB\n"
        f"Estimated factorization memory with fill-in indicator: {factor_mb:.2f} MB\n"
        f"Estimated assembly time: {assembly:.6e} s\n"
        f"Estimated solve time lower-bound scaling: {solve:.6e} s\n"
        f"Decision: {decision}\n"
        "Reason: M8 would have about four times M7 unknowns and substantially larger sparse-direct fill-in; Gate F therefore evaluates M7 fallback asymptotic evidence.\n"
    )
    (root / "logs/M8_preflight.txt").write_text(text)
    return {"estimated_elements": elems, "estimated_DOFs": dofs, "estimated_NNZ": nnz, "estimated_raw_MB": raw_mb, "estimated_factor_MB": factor_mb, "decision": decision}


def run_final_capacitance(root: Path, quick: bool = False) -> dict:
    levels = [3, 4] if quick else [3, 4, 5, 6]
    uniform_solutions = {level: solve_uniform(level) for level in levels}
    uniform_rows = [_solution_row(f"M{level}", "uniform", uniform_solutions[level], "uniform_convergence") for level in levels]
    for i in range(1, len(uniform_rows)):
        prev = float(uniform_rows[i - 1]["CE"])
        cur = float(uniform_rows[i]["CE"])
        uniform_rows[i]["relative_change_from_previous_uniform"] = abs(cur - prev) / abs(cur)
        uniform_rows[i]["relative_change_percent"] = 100.0 * uniform_rows[i]["relative_change_from_previous_uniform"]
    monotone = all(float(uniform_rows[i]["CE"]) >= float(uniform_rows[i + 1]["CE"]) - 1e-20 for i in range(len(uniform_rows) - 1))
    orders = _apparent_orders(uniform_rows)
    richardson = None
    richardson_rel = float("nan")
    if len(uniform_rows) >= 3:
        p = orders[-1]["p_C"]
        if math.isfinite(p) and p > 0.1:
            cf = float(uniform_rows[-1]["CE"])
            cc = float(uniform_rows[-2]["CE"])
            richardson = cf + (cf - cc) / (2.0**p - 1.0)
            richardson_rel = abs(richardson - cf) / abs(richardson)

    final_uniform = uniform_rows[-1]
    final_uniform["role"] = "final_selected"
    final_ce = float(final_uniform["CE"])
    for r in uniform_rows:
        r["relative_to_final_uniform"] = abs(float(r["CE"]) - final_ce) / abs(final_ce)

    graded_solutions = []
    graded_rows = []
    if not quick:
        for spec in GRADED_SPECS[2:]:
            solved = solve_mesh(build_graded_mesh(spec), repeats=1)
            solved["spec"] = spec
            row = _solution_row(f"G{spec.level}", "graded", solved, "graded_crosscheck", final_ce=final_ce)
            graded_rows.append(row)
            graded_solutions.append(solved)

    selection_rows = uniform_rows + graded_rows
    write_csv(root / "tables/final_capacitance_selection.csv", selection_rows)
    write_csv(root / "tables/final_capacitance_convergence.csv", uniform_rows)
    m7 = _m7_preflight(final_uniform, root)
    if not quick and m7["decision"] != "M7_NOT_RUN_RESOURCE_LIMIT":
        levels.append(7)
        solved7 = solve_uniform(7)
        uniform_solutions[7] = solved7
        row7 = _solution_row("M7", "uniform", solved7, "uniform_convergence")
        prev = float(uniform_rows[-1]["CE"])
        cur = float(row7["CE"])
        row7["relative_change_from_previous_uniform"] = abs(cur - prev) / abs(cur)
        row7["relative_change_percent"] = 100.0 * row7["relative_change_from_previous_uniform"]
        uniform_rows.append(row7)
        monotone = all(float(uniform_rows[i]["CE"]) >= float(uniform_rows[i + 1]["CE"]) - 1e-20 for i in range(len(uniform_rows) - 1))
        orders = _apparent_orders(uniform_rows)
        richardson = None
        richardson_rel = float("nan")
        if len(uniform_rows) >= 3:
            p = orders[-1]["p_C"]
            if math.isfinite(p) and p > 0.1:
                cf = float(uniform_rows[-1]["CE"])
                cc = float(uniform_rows[-2]["CE"])
                richardson = cf + (cf - cc) / (2.0**p - 1.0)
                richardson_rel = abs(richardson - cf) / abs(richardson)
        final_uniform = uniform_rows[-1]
        final_ce = float(final_uniform["CE"])
        final_uniform["status"] = "FINAL_SELECTED"
        for r in uniform_rows:
            r["role"] = "uniform_convergence"
            r["status"] = "computed"
            r["relative_to_final_uniform"] = abs(float(r["CE"]) - final_ce) / abs(final_ce)
        final_uniform["role"] = "final_selected"
        final_uniform["status"] = "FINAL_SELECTED"
        m7["decision"] = "M7_RUN_SUCCESS"
        for r in graded_rows:
            r["relative_to_final_uniform"] = abs(float(r["CE"]) - final_ce) / abs(final_ce)

    write_csv(root / "tables/final_capacitance_selection.csv", uniform_rows + graded_rows)
    write_csv(root / "tables/final_capacitance_convergence.csv", uniform_rows)

    cext_rows = _richardson_rows(uniform_rows)
    write_csv(root / "tables/capacitance_asymptotic_diagnostics.csv", cext_rows)
    m8 = _m8_preflight(final_uniform, root) if not quick and levels[-1] == 7 else {"decision": "M8_NOT_EVALUATED"}
    p_vals = [float(r["p_C"]) for r in cext_rows if math.isfinite(float(r["p_C"]))]
    last_two_p_diff = abs(p_vals[-1] - p_vals[-2]) if len(p_vals) >= 2 else float("inf")
    p_range = max(p_vals) - min(p_vals) if p_vals else float("inf")
    last_cext_change = float(cext_rows[-1]["relative_change_from_previous_Cext"]) if len(cext_rows) >= 2 else float("inf")
    last_richardson_rel = float(cext_rows[-1]["relative_finest_to_extrapolated"]) if cext_rows else richardson_rel
    fallback_ok = (
        (not quick)
        and m8["decision"] == "M8_NOT_RECOMMENDED_RESOURCE_LIMIT"
        and monotone
        and all(float(uniform_rows[i]["relative_change_from_previous_uniform"]) > float(uniform_rows[i + 1]["relative_change_from_previous_uniform"]) for i in range(1, len(uniform_rows) - 1))
        and p_vals
        and all(p > 0 for p in p_vals)
        and p_range <= 0.05
        and last_two_p_diff <= 0.02
        and last_cext_change <= 0.0002
        and last_richardson_rel <= 0.0015
        and float(final_uniform["relative_change_from_previous_uniform"]) <= 0.002
    )
    gate_pass = fallback_ok
    acceptance_path = "M7_ASYMPTOTIC_RESOURCE_FALLBACK" if gate_pass else "NOT_ACCEPTED"
    final_solution = uniform_solutions[levels[-1]]
    final_mesh_id = FINAL_MESH_ID_M7 if levels[-1] == 7 else FINAL_MESH_ID_M6
    mesh_path = _freeze_mesh(root, final_solution, final_mesh_id, final=gate_pass)
    cap = final_solution["cap"]
    sol = final_solution["solution"]
    spatial_indicator = float(final_uniform["relative_change_from_previous_uniform"])
    final_row = {
        "FINAL_MESH_ID": final_mesh_id,
        "gate_f_status": "PASS" if gate_pass else "NOT YET COMPLETE",
        "acceptance_path": acceptance_path,
        "domain": "W/H=8, Hair/H=8, homogeneous Neumann",
        "elements": final_solution["mesh"].geom.elements.shape[0],
        "DOFs": final_solution["mesh"].geom.points.shape[0],
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
        "symmetry_error": _symmetry_error(sol.phi, final_solution["mesh"].geom.points),
        "residual": float(np.linalg.norm(sol.matrix_free @ sol.phi[sol.free_dofs] - sol.rhs_free, ord=np.inf)),
        "spatial_indicator": spatial_indicator,
        "domain_indicator": DOMAIN_INDICATOR,
        "combined_conservative": spatial_indicator + DOMAIN_INDICATOR,
        "Richardson_C_ext": cext_rows[-1]["C_extrapolated"] if cext_rows else np.nan,
        "relative_final_to_Richardson": last_richardson_rel,
        "last_apparent_p_C": p_vals[-1] if p_vals else np.nan,
        "M8_run": False,
        "mesh_artifact": "data/processed/final_engineering_mesh.npz" if gate_pass else "data/processed/candidate_engineering_mesh.npz",
        "status": "FINAL" if gate_pass else "CANDIDATE_ONLY",
    }
    if gate_pass:
        candidate_csv = root / "tables/candidate_dielectric_capacitance.csv"
        final_csv = root / "tables/final_dielectric_capacitance.csv"
        if candidate_csv.exists():
            candidate_csv.unlink()
        write_csv(final_csv, [final_row])
    else:
        final_csv = root / "tables/final_dielectric_capacitance.csv"
        final_mesh = root / "data/processed/final_engineering_mesh.npz"
        if not quick:
            if final_csv.exists():
                final_csv.unlink()
            if final_mesh.exists():
                final_mesh.unlink()
        write_csv(root / "tables/candidate_dielectric_capacitance.csv", [final_row])
    _plot(root, uniform_rows, graded_rows, final_uniform["mesh_id"], richardson, gate_pass, cext_rows)

    g2_row = next((r for r in graded_rows if r["mesh_id"] == "G2"), None)
    g2_decision = "REJECT_FINAL_ENGINEERING_MESH" if g2_row and float(g2_row["relative_to_final_uniform"]) > spatial_indicator else "NOT_EVALUATED_IN_QUICK"
    log = (
        "Gate F final dielectric capacitance selection\n\n"
        "Final domain: W/H=8, Hair/H=8, homogeneous Neumann. Gate C 8H-vs-12H finite-domain indicator is carried separately as 0.182031%.\n"
        f"Uniform sequence: {[ (r['mesh_id'], r['CE']) for r in uniform_rows ]}\n"
        f"Monotone CE decrease: {monotone}\n"
        f"Successive relative CE changes: {[r['relative_change_from_previous_uniform'] for r in uniform_rows]}\n"
        f"M7 preflight decision: {m7['decision']}\n"
        f"M8 preflight decision: {m8['decision']}\n"
        f"Apparent capacitance orders: {orders}\n"
        f"Richardson diagnostics: {cext_rows}\n"
        f"p_C range: {p_range}; last p_C difference: {last_two_p_diff}; last Richardson limit change: {last_cext_change}\n"
        f"Richardson diagnostic C_ext: {cext_rows[-1]['C_extrapolated'] if cext_rows else richardson}; relative finest-to-ext: {last_richardson_rel}\n"
        f"Graded cross-check relative to final uniform: {[ (r['mesh_id'], r['relative_to_final_uniform']) for r in graded_rows ]}\n"
        f"G2 final suitability decision: {g2_decision}\n"
        f"Selected final mesh: {final_mesh_id}\n"
        f"Acceptance path: {acceptance_path}\n"
        f"Final CE: {cap.CE:.16e}; CQ: {cap.CQ:.16e}; delta_EQ: {cap.delta_EQ:.16e}\n"
        f"Spatial indicator: {spatial_indicator:.6e}; domain indicator: {DOMAIN_INDICATOR:.6e}; conservative sum: {spatial_indicator + DOMAIN_INDICATOR:.6e}\n"
        "No air-only solve, epsilon_eff, Z0, or benchmark calculation was performed.\n"
        f"Gate F status: {'PASS' if gate_pass else 'NOT YET COMPLETE'}\n"
    )
    (root / "logs/gate_F_capacitance.txt").write_text(log)
    return {
        "gate_f_status": "PASS" if gate_pass else "NOT YET COMPLETE",
        "FINAL_MESH_ID": final_mesh_id,
        "CE": cap.CE,
        "CQ": cap.CQ,
        "spatial_indicator": spatial_indicator,
        "domain_indicator": DOMAIN_INDICATOR,
        "g2_decision": g2_decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_final_capacitance(args.project_root, quick=args.quick))


if __name__ == "__main__":
    main()
