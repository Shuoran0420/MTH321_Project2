"""Regenerate the Project 2 numerical evidence chain."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from baseline import run_baseline
from domain_study import run_domain_study
from final_capacitance import run_final_capacitance
from gate_g_air import final_mesh_fingerprint, load_frozen_geometry
from gate_g_air import run_gate_g
from graded_study import run_graded_study
from impedance import final_impedance_quantities, hammerstad_jensen_microstrip, ipc2141_microstrip, load_authoritative_inputs
from impedance import run_gate_h
from uniform_convergence import run_uniform_convergence
from verification import main as verification_main


def _read_first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def run_quick_smoke(root: Path) -> dict[str, object]:
    """Run non-destructive infrastructure checks without overwriting final evidence."""
    start = time.perf_counter()
    log_lines: list[str] = [
        "Project 2 quick smoke validation",
        "",
        "Mode: quick",
        "Scope: non-destructive checks only; authoritative full-resolution tables, figures, mesh artifacts and logs are not regenerated.",
    ]

    final_csv = root / "tables/final_dielectric_capacitance.csv"
    air_csv = root / "tables/final_air_capacitance.csv"
    eff_csv = root / "tables/final_effective_permittivity.csv"
    z0_csv = root / "tables/final_characteristic_impedance.csv"
    for path in [final_csv, air_csv, eff_csv, z0_csv, root / "data/processed/final_engineering_mesh.npz"]:
        if not path.exists():
            raise FileNotFoundError(f"required authoritative artifact is missing: {path.relative_to(root)}")
        log_lines.append(f"FOUND: {path.relative_to(root)}")

    geom, mesh_data, mesh_path = load_frozen_geometry(root)
    mesh_sha = final_mesh_fingerprint(mesh_data)
    if mesh_sha != "06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75":
        raise RuntimeError(f"final mesh SHA mismatch in quick smoke: {mesh_sha}")
    log_lines.append(f"FINAL_MESH_ID: {str(mesh_data['FINAL_MESH_ID'])}")
    log_lines.append(f"FINAL_MESH_SHA256: {mesh_sha}")
    log_lines.append(f"mesh source: {mesh_path}")
    log_lines.append(f"mesh size: elements={geom.elements.shape[0]}, DOFs={geom.points.shape[0]}")

    final_row = _read_first_csv_row(final_csv)
    eff_row = _read_first_csv_row(eff_csv)
    z0_row = _read_first_csv_row(z0_csv)
    if final_row["gate_f_status"] != "PASS" or eff_row["Gate_G_status"] != "PASS" or z0_row["Gate_H_status"] != "PASS":
        raise RuntimeError("Gate F/G/H authoritative CSV status is not PASS")

    inputs = load_authoritative_inputs(root)
    q = final_impedance_quantities(inputs["C_dielectric"], inputs["C_air"], inputs["epsilon_eff"])
    if abs(q["Z0_FEM_ohm"] - float(z0_row["Z0_FEM_ohm"])) / abs(q["Z0_FEM_ohm"]) > 1e-12:
        raise RuntimeError("quick smoke Z0 recomputation disagrees with authoritative Gate H CSV")
    hj = hammerstad_jensen_microstrip(4.4, 3.0e-3, 1.6e-3, 0.035e-3)
    ipc = ipc2141_microstrip(4.4, 3.0e-3, 1.6e-3, 0.035e-3)
    if hj.Z0_ohm <= 0.0 or ipc.Z0_ohm <= 0.0:
        raise RuntimeError("benchmark smoke check produced non-positive impedance")

    elapsed = time.perf_counter() - start
    log_lines.extend(
        [
            f"C_dielectric={inputs['C_dielectric']:.16e} F/m",
            f"C_air={inputs['C_air']:.16e} F/m",
            f"epsilon_eff={inputs['epsilon_eff']:.16e}",
            f"Z0_FEM={q['Z0_FEM_ohm']:.16e} ohm",
            f"Hammerstad_Jensen_Z0={hj.Z0_ohm:.16e} ohm",
            f"IPC_2141_Z0={ipc.Z0_ohm:.16e} ohm",
            f"quick_runtime_s={elapsed:.6f}",
            "Quick smoke status: PASS",
        ]
    )
    log_path = root / "logs/run_all_quick_smoke.txt"
    log_path.write_text("\n".join(log_lines) + "\n")
    (root / "results_summary_quick.txt").write_text(
        "Project 2 quick smoke summary\n"
        "This file is intentionally separate from results_summary.txt and does not replace full-resolution evidence.\n"
        f"status=PASS\nruntime_s={elapsed:.6f}\nFINAL_MESH_SHA256={mesh_sha}\n"
    )
    return {"status": "PASS", "runtime_s": elapsed, "FINAL_MESH_SHA256": mesh_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run lightweight smoke evidence")
    parser.add_argument("--full", action="store_true", help="run final-coursework evidence")
    args = parser.parse_args()
    if args.quick and args.full:
        raise SystemExit("Choose only one of --quick or --full.")
    mode = "full" if args.full else "quick"
    root = Path(__file__).resolve().parents[1]
    if mode == "quick":
        result = run_quick_smoke(root)
        print(result)
        return

    levels = ["2", "3", "4", "5"] if mode == "quick" else ["2", "3", "4", "5", "6"]

    old_argv = sys.argv
    try:
        sys.argv = ["verification.py", "--project-root", str(root), "--levels", *levels]
        verification_main()
    finally:
        sys.argv = old_argv

    sol, cap, metrics, gate_b_passed = run_baseline(root)
    gate_c = run_domain_study(root, quick=(mode == "quick"))
    gate_d = run_uniform_convergence(root, quick=(mode == "quick"))
    gate_e = run_graded_study(root, quick=(mode == "quick"))
    gate_f = run_final_capacitance(root, quick=(mode == "quick"))
    gate_g = run_gate_g(root, quick=(mode == "quick"))
    gate_h = run_gate_h(root, quick=(mode == "quick"))

    summary = root / "results_summary.txt"
    summary.write_text(
        "Project 2 Microstrip FEM results summary\n"
        f"Mode: {mode}\n"
        "Gate A: PASS; see logs/gate_A_verification.txt\n"
        f"Gate B: {'PASS' if gate_b_passed else 'FAIL'}; see logs/gate_B_checks.txt\n"
        f"Baseline mesh: elements={metrics['elements']} nodes={metrics['nodes']} free_dofs={sol.free_dofs.size} NNZ={metrics['NNZ']}\n"
        f"Potential range: min={metrics['min_phi']:.16e} max={metrics['max_phi']:.16e}\n"
        f"Symmetry relative error: {metrics['symmetry_relative_error']:.16e}\n"
        f"Matrix symmetry error: {metrics['matrix_symmetry_error']:.16e}\n"
        f"Smallest eigenvalue estimate: {metrics['lambda_min']:.16e}\n"
        f"Largest eigenvalue estimate: {metrics['lambda_max']:.16e}\n"
        f"Physical matrix condition estimate: {metrics['condition_estimate']:.16e}\n"
        f"Scaled matrix condition estimate: {metrics['scaled_condition_estimate']:.16e}\n"
        f"Free residual infinity norm: {metrics['residual_inf']:.16e}\n"
        f"CE={cap.CE:.16e} F/m CQ={cap.CQ:.16e} F/m delta_EQ={cap.delta_EQ:.16e}\n"
        f"Charge-balance defect: {cap.charge_balance_defect:.16e}\n"
        f"Gate C: {gate_c['gate_c_status']}; smallest below 1%={gate_c['smallest']}H; mesh_sensitivity={gate_c['mesh_sensitivity']:.6e}; trend={gate_c['trend']}\n"
        f"Gate D: {gate_d['gate_d_status']}; fitted_slope={gate_d['fitted_slope']:.6f}; fit_levels={gate_d['fit_levels']}; reference_self_relative={gate_d['reference_self_relative']:.6e}\n"
        f"Gate E: {gate_e['gate_e_status']}; production_mesh_candidate={gate_e['production_mesh_candidate']}\n"
        f"Gate F: {gate_f['gate_f_status']}; FINAL_MESH_ID={gate_f['FINAL_MESH_ID']}; CE={gate_f['CE']:.16e}; spatial_indicator={gate_f['spatial_indicator']:.6e}; domain_indicator={gate_f['domain_indicator']:.6e}\n"
        f"Gate G: {gate_g['gate_g_status']}; FINAL_MESH_ID={gate_g['FINAL_MESH_ID']}; C_dielectric={gate_g['C_dielectric']:.16e}; C_air={gate_g['C_air']:.16e}; epsilon_eff={gate_g['epsilon_eff']:.16e}; same_mesh={gate_g['same_mesh_PASS']}\n"
        f"Gate G epsilon_eff sequence: M5={gate_g['epsilon_eff_M5']:.16e}; M6={gate_g['epsilon_eff_M6']:.16e}; M7={gate_g['epsilon_eff_M7']:.16e}\n"
        f"Gate H: {gate_h['gate_h_status']}; Z0_FEM={gate_h['Z0_FEM_ohm']:.16e} ohm; HJ={gate_h['HJ_Z0']:.16e} ohm; IPC={gate_h['IPC_Z0']:.16e} ohm\n"
        f"Gate H Z0 convergence: M5={gate_h['Z0_M5']:.16e}; M6={gate_h['Z0_M6']:.16e}; M7={gate_h['Z0_M7']:.16e}\n"
        f"Gate H benchmark differences: HJ={gate_h['HJ_percent_difference']:.6e}%; IPC={gate_h['IPC_percent_difference']:.6e}%\n"
        "Gates I-J: IN PROGRESS\n"
        "Overall status: NOT YET COMPLETE\n"
    )


if __name__ == "__main__":
    main()
