"""Baseline microstrip solve, Gate B diagnostics, and CSV/log generation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from capacitance import CapacitanceResult, extract_capacitance
from fem_solver import MicrostripSolution, solve_microstrip
from geometry import build_baseline_geometry
from plotting import write_baseline_figures


def symmetry_metrics(sol: MicrostripSolution) -> tuple[float, float]:
    pts = sol.geometry.points
    values = sol.phi
    lookup = {(round(float(x), 15), round(float(y), 15)): i for i, (x, y) in enumerate(pts)}
    diffs = []
    refs = []
    for i, (x, y) in enumerate(pts):
        j = lookup.get((round(float(-x), 15), round(float(y), 15)))
        if j is not None:
            diffs.append(values[i] - values[j])
            refs.append(values[i])
    diffs_arr = np.array(diffs, dtype=float)
    refs_arr = np.array(refs, dtype=float)
    rel = float(np.linalg.norm(diffs_arr) / max(np.linalg.norm(refs_arr), 1e-300))
    max_abs = float(np.max(np.abs(diffs_arr))) if diffs_arr.size else float("nan")
    return rel, max_abs


def gate_b_metrics(sol: MicrostripSolution, cap: CapacitanceResult) -> dict[str, float | int | str]:
    free_solution = sol.phi[sol.free_dofs]
    residual = sol.matrix_free @ free_solution - sol.rhs_free
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    system_scale = float(np.linalg.norm(sol.matrix_free @ free_solution, ord=np.inf) + np.linalg.norm(sol.rhs_free, ord=np.inf))
    normalized_residual = residual_inf / max(system_scale, 1e-300)
    asym = sol.matrix_free - sol.matrix_free.T
    matrix_symmetry_error = float(np.linalg.norm(asym.toarray(), ord="fro") / max(np.linalg.norm(sol.matrix_free.toarray(), ord="fro"), 1e-300))
    symmetry_relative_error, symmetry_max_error = symmetry_metrics(sol)
    min_phi = float(np.min(sol.phi))
    max_phi = float(np.max(sol.phi))
    tol = 1e-10
    bound_violation = float(max(0.0, -min_phi, max_phi - sol.geometry.config.V0))
    return {
        "domain": f"W/H={sol.geometry.config.W_over_H}, Hair/H={sol.geometry.config.Hair_over_H}",
        "min_phi": min_phi,
        "max_phi": max_phi,
        "potential_bound_violation": bound_violation,
        "symmetry_relative_error": symmetry_relative_error,
        "symmetry_max_error": symmetry_max_error,
        "matrix_symmetry_error": matrix_symmetry_error,
        "lambda_min": sol.lambda_min,
        "lambda_max": sol.lambda_max,
        "condition_estimate": sol.condition_estimate,
        "scaled_lambda_min": sol.scaled_lambda_min,
        "scaled_lambda_max": sol.scaled_lambda_max,
        "scaled_condition_estimate": sol.scaled_condition_estimate,
        "residual_inf": residual_inf,
        "normalized_residual": normalized_residual,
        "elements": int(sol.geometry.elements.shape[0]),
        "nodes": int(sol.geometry.points.shape[0]),
        "DOFs": int(sol.geometry.points.shape[0]),
        "NNZ": int(sol.matrix_free.nnz),
        "assembly_time": sol.assembly_time,
        "solve_time": sol.solve_time,
        "CE": cap.CE,
        "CQ": cap.CQ,
        "delta_EQ": cap.delta_EQ,
        "charge_balance_defect": cap.charge_balance_defect,
    }


def write_single_row_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_capacitance_csv(path: Path, sol: MicrostripSolution, cap: CapacitanceResult) -> None:
    row = {
        "domain": f"W/H={sol.geometry.config.W_over_H}, Hair/H={sol.geometry.config.Hair_over_H}",
        "elements": int(sol.geometry.elements.shape[0]),
        "DOFs": int(sol.geometry.points.shape[0]),
        "CE": cap.CE,
        "CQ": cap.CQ,
        "delta_EQ": cap.delta_EQ,
        "Q_trace": cap.Q_trace,
        "Q_ground": cap.Q_ground,
        "energy": cap.energy,
        "solve_time": sol.solve_time,
    }
    write_single_row_csv(path, row)


def pass_fail_lines(metrics: dict[str, float | int | str], cap: CapacitanceResult) -> list[str]:
    checks = [
        ("min_phi", metrics["min_phi"], ">= -1e-10 V, discrete maximum principle tolerance", float(metrics["min_phi"]) >= -1e-10),
        ("max_phi", metrics["max_phi"], "<= 1+1e-10 V, discrete maximum principle tolerance", float(metrics["max_phi"]) <= 1.0 + 1e-10),
        ("symmetry_relative_error", metrics["symmetry_relative_error"], "< 1e-10 for symmetric structured mesh and direct solve", float(metrics["symmetry_relative_error"]) < 1e-10),
        ("matrix_symmetry_error", metrics["matrix_symmetry_error"], "< 1e-13, reduced free-DOF matrix should be symmetric to roundoff", float(metrics["matrix_symmetry_error"]) < 1e-13),
        ("lambda_min", metrics["lambda_min"], "> 0, reduced Dirichlet system is SPD", float(metrics["lambda_min"]) > 0.0),
        ("lambda_max", metrics["lambda_max"], "> lambda_min, finite extremal eigenvalue estimate", np.isfinite(float(metrics["lambda_max"])) and float(metrics["lambda_max"]) > float(metrics["lambda_min"])),
        ("condition_estimate", metrics["condition_estimate"], "finite and > 1; ratio is more informative than small SI-scaled lambda_min", np.isfinite(float(metrics["condition_estimate"])) and float(metrics["condition_estimate"]) > 1.0),
        ("scaled_condition_estimate", metrics["scaled_condition_estimate"], "finite and > 1 for diagonally scaled reduced matrix", np.isfinite(float(metrics["scaled_condition_estimate"])) and float(metrics["scaled_condition_estimate"]) > 1.0),
        ("residual_inf", metrics["residual_inf"], "< 1e-18 absolute free-DOF residual for direct solve at this scale", float(metrics["residual_inf"]) < 1e-18),
        ("CE", cap.CE, "> 0 and finite", np.isfinite(cap.CE) and cap.CE > 0.0),
        ("CQ", cap.CQ, "> 0 and finite", np.isfinite(cap.CQ) and cap.CQ > 0.0),
        ("delta_EQ", cap.delta_EQ, "< 1e-10 for reaction-energy algebraic consistency", cap.delta_EQ < 1e-10),
        ("charge_balance_defect", cap.charge_balance_defect, "< 1e-10 for trace plus ground reactions with homogeneous Neumann far boundary", cap.charge_balance_defect < 1e-10),
    ]
    lines = []
    for name, value, criterion, ok in checks:
        lines.append(f"{name}: value={value:.16e} criterion={criterion} status={'PASS' if ok else 'FAIL'}")
    return lines


def write_gate_b_log(path: Path, metrics: dict[str, float | int | str], cap: CapacitanceResult, sol: MicrostripSolution) -> bool:
    lines = pass_fail_lines(metrics, cap)
    passed = all("status=PASS" in line for line in lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("Gate B baseline microstrip checks\n")
        stream.write("Diagnosed matrix: correctly Dirichlet-eliminated free-DOF stiffness matrix.\n")
        stream.write("Charge extraction: full pre-elimination stiffness reaction K*u, summed on trace Dirichlet dofs.\n")
        stream.write(f"Trace dof count: {sol.trace_dofs.size}; ground dof count: {sol.ground_dofs.size}; free dof count: {sol.free_dofs.size}\n\n")
        stream.write("Conditioning note: the physical stiffness matrix includes epsilon_0 in SI units, so absolute eigenvalues are small.\n")
        stream.write("For SPD, the sign of lambda_min is the key positivity check; for refinement comparisons, use the condition ratios.\n")
        stream.write("The scaled condition estimate uses D^{-1/2} A D^{-1/2} and should be interpreted as a diagonal-scaled diagnostic, not the physical matrix condition number.\n\n")
        for line in lines:
            stream.write(line + "\n")
        stream.write(f"\nGate B status: {'PASS' if passed else 'FAIL'}\n")
        stream.write("Interpretation: CE/CQ consistency checks implementation wiring only; it does not prove spatial convergence.\n")
    return passed


def run_baseline(project_root: Path) -> tuple[MicrostripSolution, CapacitanceResult, dict[str, float | int | str], bool]:
    geom = build_baseline_geometry(project_root)
    sol = solve_microstrip(geom)
    cap = extract_capacitance(sol)
    metrics = gate_b_metrics(sol, cap)
    write_baseline_figures(sol, project_root)
    write_single_row_csv(project_root / "tables" / "baseline_checks.csv", metrics)
    write_capacitance_csv(project_root / "tables" / "capacitance_extraction.csv", sol, cap)
    passed = write_gate_b_log(project_root / "logs" / "gate_B_checks.txt", metrics, cap, sol)
    return sol, cap, metrics, passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sol, cap, metrics, passed = run_baseline(args.project_root)
    print(f"Gate B status: {'PASS' if passed else 'FAIL'}")
    print(f"mesh: elements={metrics['elements']} nodes={metrics['nodes']} free_dofs={sol.free_dofs.size} nnz={metrics['NNZ']}")
    print(f"phi range: [{metrics['min_phi']:.16e}, {metrics['max_phi']:.16e}]")
    print(f"symmetry_relative_error={metrics['symmetry_relative_error']:.16e}")
    print(f"matrix_symmetry_error={metrics['matrix_symmetry_error']:.16e}")
    print(f"lambda_min={metrics['lambda_min']:.16e}")
    print(f"lambda_max={metrics['lambda_max']:.16e}")
    print(f"condition_estimate={metrics['condition_estimate']:.16e}")
    print(f"scaled_condition_estimate={metrics['scaled_condition_estimate']:.16e}")
    print(f"residual_inf={metrics['residual_inf']:.16e}")
    print(f"CE={cap.CE:.16e} F/m CQ={cap.CQ:.16e} F/m delta_EQ={cap.delta_EQ:.16e}")
    print(f"charge_balance_defect={cap.charge_balance_defect:.16e}")


if __name__ == "__main__":
    main()
