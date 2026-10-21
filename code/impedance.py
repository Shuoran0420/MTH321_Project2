"""Gate H characteristic impedance and engineering benchmark validation."""

from __future__ import annotations

import argparse
import csv
import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C0 = 299_792_458.0


@dataclass(frozen=True)
class HammerstadJensenResult:
    epsilon_eff: float
    Z0_ohm: float
    u: float
    tau: float
    delta_u_1: float
    delta_u_r: float
    u_1: float
    u_r: float
    epsilon_eff_zero_thickness_u_r: float
    Z_air_u_1: float
    Z_air_u_r: float


@dataclass(frozen=True)
class IPC2141Result:
    Z0_ohm: float
    w_over_H: float
    log_argument: float


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_hj_discrepancies(rows: list[dict]) -> list[dict]:
    """Append the Appendix-E discrepancy without replacing FEM-denominator fields."""
    z_hj = float(next(r for r in rows if r["method"] == "Hammerstad_Jensen")["Z0_ohm"])
    return [
        {
            **row,
            "relative_discrepancy_Z0_vs_HJ": abs(float(row["Z0_ohm"]) - z_hj) / abs(z_hj),
            "percent_discrepancy_Z0_vs_HJ": 100.0 * abs(float(row["Z0_ohm"]) - z_hj) / abs(z_hj),
        }
        for row in rows
    ]


def load_authoritative_inputs(root: Path) -> dict:
    dielectric_path = root / "tables/final_dielectric_capacitance.csv"
    eps_path = root / "tables/final_effective_permittivity.csv"
    dielectric = next(csv.DictReader(dielectric_path.open()))
    effective = next(csv.DictReader(eps_path.open()))
    if dielectric["gate_f_status"] != "PASS":
        raise ValueError("Gate F dielectric input is not PASS")
    if effective["Gate_G_status"] != "PASS":
        raise ValueError("Gate G effective-permittivity input is not PASS")
    if dielectric["FINAL_MESH_ID"] != effective["FINAL_MESH_ID"]:
        raise ValueError("Gate F and Gate G FINAL_MESH_ID values disagree")
    c_dielectric = float(dielectric["CE"])
    c_dielectric_gate_g = float(effective["C_dielectric"])
    if abs(c_dielectric - c_dielectric_gate_g) / abs(c_dielectric) > 1e-12:
        raise ValueError("Gate F and Gate G dielectric capacitance values disagree")
    return {
        "FINAL_MESH_ID": dielectric["FINAL_MESH_ID"],
        "C_dielectric": c_dielectric,
        "C_air": float(effective["C_air"]),
        "epsilon_eff": float(effective["epsilon_eff"]),
        "gate_f_status": dielectric["gate_f_status"],
        "gate_g_status": effective["Gate_G_status"],
        "spatial_indicator": float(dielectric["spatial_indicator"]),
        "domain_indicator": float(dielectric["domain_indicator"]),
    }


def final_impedance_quantities(c_dielectric: float, c_air: float, epsilon_eff: float, c0: float = C0) -> dict:
    z0_fem = 1.0 / (c0 * math.sqrt(c_dielectric * c_air))
    z0_air = 1.0 / (c0 * c_air)
    z0_from_epsilon = z0_air / math.sqrt(epsilon_eff)
    inductance = 1.0 / (c0**2 * c_air)
    z0_from_lc = math.sqrt(inductance / c_dielectric)
    phase_velocity = c0 / math.sqrt(epsilon_eff)
    phase_velocity_from_lc = 1.0 / math.sqrt(inductance * c_dielectric)
    return {
        "c0": c0,
        "L_H_per_m": inductance,
        "Z0_air_ohm": z0_air,
        "Z0_FEM_ohm": z0_fem,
        "Z0_from_epsilon_ohm": z0_from_epsilon,
        "Z0_from_LC_ohm": z0_from_lc,
        "phase_velocity_m_per_s": phase_velocity,
        "phase_velocity_over_c0": phase_velocity / c0,
        "internal_Z0_consistency_error": max(
            abs(z0_fem - z0_from_epsilon) / abs(z0_fem),
            abs(z0_fem - z0_from_lc) / abs(z0_fem),
        ),
        "internal_velocity_consistency_error": abs(phase_velocity - phase_velocity_from_lc) / abs(phase_velocity),
    }


def _hj_a(s: float) -> float:
    return 1.0 + (1.0 / 49.0) * math.log((s**4 + (s / 52.0) ** 2) / (s**4 + 0.432)) + (1.0 / 18.7) * math.log(1.0 + (s / 18.1) ** 3)


def _hj_b(epsilon_r: float) -> float:
    return 0.564 * ((epsilon_r - 0.9) / (epsilon_r + 3.0)) ** 0.053


def _hj_epsilon(epsilon_r: float, s: float) -> float:
    return (epsilon_r + 1.0) / 2.0 + (epsilon_r - 1.0) / 2.0 * (1.0 + 10.0 / s) ** (-_hj_a(s) * _hj_b(epsilon_r))


def _hj_f(s: float) -> float:
    return 6.0 + (2.0 * math.pi - 6.0) * math.exp(-(30.666 / s) ** 0.7528)


def _hj_z_air(s: float) -> float:
    return 60.0 * math.log(_hj_f(s) / s + math.sqrt(1.0 + (2.0 / s) ** 2))


def hammerstad_jensen_microstrip(epsilon_r: float, w: float, H: float, t: float) -> HammerstadJensenResult:
    """Hammerstad-Jensen finite-thickness quasistatic microstrip formula.

    Inputs are only geometry and substrate relative permittivity. All lengths may
    use any common unit because the formula uses the ratios w/H and t/H.
    """
    if epsilon_r <= 1.0 or min(w, H) <= 0.0 or t < 0.0:
        raise ValueError("invalid Hammerstad-Jensen inputs")
    u = w / H
    tau = t / H
    if tau > 0.0:
        delta_u_1 = (tau / math.pi) * math.log(1.0 + (4.0 * math.e / tau) * math.tanh(math.sqrt(6.517 * u)) ** 2)
        delta_u_r = delta_u_1 * (1.0 + 1.0 / math.cosh(math.sqrt(epsilon_r - 1.0))) / 2.0
    else:
        delta_u_1 = 0.0
        delta_u_r = 0.0
    u_1 = u + delta_u_1
    u_r = u + delta_u_r
    eps_ur = _hj_epsilon(epsilon_r, u_r)
    z_air_u_1 = _hj_z_air(u_1)
    z_air_u_r = _hj_z_air(u_r)
    epsilon_eff = eps_ur * (z_air_u_1 / z_air_u_r) ** 2
    z0 = z_air_u_1 / math.sqrt(epsilon_eff)
    return HammerstadJensenResult(
        epsilon_eff=epsilon_eff,
        Z0_ohm=z0,
        u=u,
        tau=tau,
        delta_u_1=delta_u_1,
        delta_u_r=delta_u_r,
        u_1=u_1,
        u_r=u_r,
        epsilon_eff_zero_thickness_u_r=eps_ur,
        Z_air_u_1=z_air_u_1,
        Z_air_u_r=z_air_u_r,
    )


def ipc2141_microstrip(epsilon_r: float, w: float, H: float, t: float) -> IPC2141Result:
    """IPC-2141 compact external-microstrip approximation using natural log."""
    if epsilon_r <= 1.0 or min(w, H) <= 0.0 or t < 0.0:
        raise ValueError("invalid IPC-2141 inputs")
    argument = 5.98 * H / (0.8 * w + t)
    if argument <= 1.0:
        raise ValueError("IPC-2141 log argument is outside the expected positive-impedance range")
    z0 = 87.0 / math.sqrt(epsilon_r + 1.41) * math.log(argument)
    return IPC2141Result(Z0_ohm=z0, w_over_H=w / H, log_argument=argument)


def _benchmark_functions_are_independent() -> bool:
    banned = {"C_dielectric", "C_air", "epsilon_eff_FEM", "Z0_FEM", "c_dielectric", "c_air", "z0_fem"}
    for fn in [hammerstad_jensen_microstrip, ipc2141_microstrip]:
        if banned.intersection(inspect.signature(fn).parameters):
            return False
    return True


def _benchmark_rows(inputs: dict, fem: dict, epsilon_r: float, w: float, H: float, t: float) -> tuple[list[dict], HammerstadJensenResult, IPC2141Result]:
    hj = hammerstad_jensen_microstrip(epsilon_r, w, H, t)
    ipc = ipc2141_microstrip(epsilon_r, w, H, t)
    z_fem = fem["Z0_FEM_ohm"]
    eps_fem = inputs["epsilon_eff"]
    rows = []
    for method, eps_eff, z0, variant, source, notes in [
        ("FEM", eps_fem, z_fem, "Matched capacitance pair, Gate G", "Gate F/G FEM outputs", "Authoritative FEM result; benchmark differences to itself are zero."),
        ("Hammerstad_Jensen", hj.epsilon_eff, hj.Z0_ohm, "Hammerstad-Jensen finite strip-thickness correction, Capacitor2d Appendix D", "Capacitor2d.pdf Appendix D; Hammerstad and Jensen 1980", "Independent quasistatic closed-form approximation using only geometry and epsilon_r."),
        ("IPC_2141", np.nan, ipc.Z0_ohm, "IPC-2141 compact external microstrip expression, natural logarithm", "Capacitor2d.pdf Appendix C; IPC-2141", "Coarse empirical engineering cross-check; no epsilon_eff value is supplied by this compact formula."),
    ]:
        rel_z = abs(z0 - z_fem) / abs(z_fem)
        rel_eps = np.nan if not math.isfinite(float(eps_eff)) else abs(float(eps_eff) - eps_fem) / abs(eps_fem)
        rows.append({
            "method": method,
            "epsilon_r_input": epsilon_r,
            "w_m": w,
            "H_m": H,
            "t_m": t,
            "w_over_H": w / H,
            "epsilon_eff": eps_eff,
            "Z0_ohm": z0,
            "relative_difference_Z0_vs_FEM": rel_z,
            "percent_difference_Z0_vs_FEM": 100.0 * rel_z,
            "relative_difference_epsilon_eff_vs_FEM": rel_eps,
            "percent_difference_epsilon_eff_vs_FEM": np.nan if not math.isfinite(float(rel_eps)) else 100.0 * rel_eps,
            "formula_variant": variant,
            "source_reference": source,
            "notes": notes,
        })
    return add_hj_discrepancies(rows), hj, ipc


def _z0_convergence_rows(root: Path, c0: float = C0) -> list[dict]:
    rows = list(csv.DictReader((root / "tables/epsilon_eff_convergence.csv").open()))
    out: list[dict] = []
    for r in rows:
        cd = float(r["C_dielectric"])
        ca = float(r["C_air"])
        z0 = 1.0 / (c0 * math.sqrt(cd * ca))
        out.append({
            "mesh_id": r["mesh_id"],
            "DOFs": int(r["DOFs"]),
            "C_dielectric": cd,
            "C_air": ca,
            "epsilon_eff": float(r["epsilon_eff"]),
            "Z0": z0,
            "relative_change_Z0_from_previous": np.nan,
        })
    for i in range(1, len(out)):
        out[i]["relative_change_Z0_from_previous"] = abs(out[i]["Z0"] - out[i - 1]["Z0"]) / abs(out[i]["Z0"])
    return out


def _plot_outputs(root: Path, fem: dict, benchmark_rows: list[dict], z0_rows: list[dict]) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)

    labels = ["FEM", "Hammerstad-Jensen", "IPC-2141"]
    lookup = {r["method"]: r for r in benchmark_rows}
    values = [lookup["FEM"]["Z0_ohm"], lookup["Hammerstad_Jensen"]["Z0_ohm"], lookup["IPC_2141"]["Z0_ohm"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("characteristic impedance (ohm)")
    ax.set_title("FEM characteristic impedance versus engineering formulas")
    ax.set_ylim(0.0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "Z0_benchmark_comparison.png", dpi=180)
    plt.close(fig)

    diff_labels = ["Hammerstad-Jensen", "IPC-2141"]
    diff_values = [lookup["Hammerstad_Jensen"]["percent_difference_Z0_vs_FEM"], lookup["IPC_2141"]["percent_difference_Z0_vs_FEM"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(diff_labels, diff_values, color=["#f58518", "#54a24b"])
    ax.set_ylabel("absolute difference from FEM (%)")
    ax.set_title("Engineering formula Z0 difference from FEM")
    ax.set_ylim(0.0, max(diff_values) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, diff_values):
        ax.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "Z0_benchmark_relative_difference.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    eps_labels = ["FEM", "Hammerstad-Jensen"]
    eps_values = [lookup["FEM"]["epsilon_eff"], lookup["Hammerstad_Jensen"]["epsilon_eff"]]
    bars = ax.bar(eps_labels, eps_values, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("effective relative permittivity")
    ax.set_title("Effective permittivity comparison")
    ax.set_ylim(0.0, max(eps_values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, eps_values):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "epsilon_eff_benchmark_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([r["DOFs"] for r in z0_rows], [r["Z0"] for r in z0_rows], "o-")
    for r in z0_rows:
        ax.annotate(r["mesh_id"], (r["DOFs"], r["Z0"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("DOFs")
    ax.set_ylabel("Z0 (ohm)")
    ax.set_title("Matched-pair FEM Z0 convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "Z0_convergence.png", dpi=180)
    plt.close(fig)


def run_gate_h(root: Path, quick: bool = False) -> dict:
    inputs = load_authoritative_inputs(root)
    c_dielectric = inputs["C_dielectric"]
    c_air = inputs["C_air"]
    epsilon_eff = inputs["epsilon_eff"]
    recomputed_eps = c_dielectric / c_air
    if abs(recomputed_eps - epsilon_eff) / abs(epsilon_eff) > 1e-12:
        raise ValueError("epsilon_eff is inconsistent with the loaded capacitance pair")
    fem = final_impedance_quantities(c_dielectric, c_air, epsilon_eff)
    if fem["internal_Z0_consistency_error"] > 1e-12 or fem["internal_velocity_consistency_error"] > 1e-12:
        raise ValueError("internal quasi-TEM identities are inconsistent")

    epsilon_r = 4.4
    H = 1.6e-3
    w = 3.0e-3
    t = 0.035e-3
    benchmarks, hj, ipc = _benchmark_rows(inputs, fem, epsilon_r, w, H, t)
    z0_rows = _z0_convergence_rows(root)
    final_z0 = z0_rows[-1]["Z0"]
    if abs(final_z0 - fem["Z0_FEM_ohm"]) / abs(fem["Z0_FEM_ohm"]) > 1e-12:
        raise ValueError("final Z0 does not match the M7 matched-pair Z0")

    relative_c_dielectric_change = abs(z0_rows[-1]["C_dielectric"] - z0_rows[-2]["C_dielectric"]) / abs(z0_rows[-1]["C_dielectric"])
    relative_c_air_change = abs(z0_rows[-1]["C_air"] - z0_rows[-2]["C_air"]) / abs(z0_rows[-1]["C_air"])
    z0_pair_sensitivity_indicator = 0.5 * (relative_c_dielectric_change + relative_c_air_change)
    difference_from_50 = fem["Z0_FEM_ohm"] - 50.0
    percent_difference_from_50 = 100.0 * abs(difference_from_50) / 50.0
    hj_row = next(r for r in benchmarks if r["method"] == "Hammerstad_Jensen")
    ipc_row = next(r for r in benchmarks if r["method"] == "IPC_2141")
    gate_h_pass = (
        inputs["gate_g_status"] == "PASS"
        and fem["Z0_FEM_ohm"] > 0.0
        and fem["L_H_per_m"] > 0.0
        and 0.0 < fem["phase_velocity_m_per_s"] < C0
        and all(float(r["Z0_ohm"]) > 0.0 for r in benchmarks)
        and _benchmark_functions_are_independent()
    )

    write_csv(root / "tables/final_characteristic_impedance.csv", [{
        "FINAL_MESH_ID": inputs["FINAL_MESH_ID"],
        "C_dielectric": c_dielectric,
        "C_air": c_air,
        "epsilon_eff": epsilon_eff,
        **fem,
        "Gate_H_status": "PASS" if gate_h_pass else "NOT YET COMPLETE",
    }])
    write_csv(root / "tables/engineering_benchmarks.csv", benchmarks)
    write_csv(root / "tables/Z0_convergence.csv", z0_rows)
    write_csv(root / "tables/final_engineering_results.csv", [{
        "FINAL_MESH_ID": inputs["FINAL_MESH_ID"],
        "w_m": w,
        "H_m": H,
        "t_m": t,
        "epsilon_r": epsilon_r,
        "C_dielectric_F_per_m": c_dielectric,
        "C_air_F_per_m": c_air,
        "epsilon_eff": epsilon_eff,
        "L_H_per_m": fem["L_H_per_m"],
        "phase_velocity_m_per_s": fem["phase_velocity_m_per_s"],
        "Z0_FEM_ohm": fem["Z0_FEM_ohm"],
        "Z0_M6_to_M7_relative_change": z0_rows[-1]["relative_change_Z0_from_previous"],
        "Z0_Hammerstad_ohm": hj.Z0_ohm,
        "Z0_IPC_ohm": ipc.Z0_ohm,
        "HJ_percent_difference": hj_row["percent_difference_Z0_vs_FEM"],
        "IPC_percent_difference": ipc_row["percent_difference_Z0_vs_FEM"],
        "difference_from_50ohm": difference_from_50,
        "percent_difference_from_50ohm": percent_difference_from_50,
        "Gate_H_status": "PASS" if gate_h_pass else "NOT YET COMPLETE",
    }])
    _plot_outputs(root, fem, benchmarks, z0_rows)

    log = (
        "Gate H final characteristic impedance and engineering validation\n\n"
        f"Gate G status: {inputs['gate_g_status']}\n"
        f"Input provenance: C from tables/final_dielectric_capacitance.csv; C_air and epsilon_eff from tables/final_effective_permittivity.csv.\n"
        f"FINAL_MESH_ID: {inputs['FINAL_MESH_ID']}\n"
        f"C_dielectric: {c_dielectric:.16e} F/m\n"
        f"C_air: {c_air:.16e} F/m\n"
        f"epsilon_eff: {epsilon_eff:.16e}\n"
        f"c0: {C0:.16e} m/s\n"
        f"L: {fem['L_H_per_m']:.16e} H/m\n"
        f"phase velocity: {fem['phase_velocity_m_per_s']:.16e} m/s; vp/c0={fem['phase_velocity_over_c0']:.16e}\n"
        f"Z0_FEM: {fem['Z0_FEM_ohm']:.16e} ohm\n"
        f"Z0_from_epsilon: {fem['Z0_from_epsilon_ohm']:.16e} ohm\n"
        f"Z0_from_LC: {fem['Z0_from_LC_ohm']:.16e} ohm\n"
        f"internal Z0 consistency error: {fem['internal_Z0_consistency_error']:.16e}\n"
        f"internal velocity consistency error: {fem['internal_velocity_consistency_error']:.16e}\n"
        f"M5/M6/M7 Z0 sequence: {[(r['mesh_id'], r['Z0'], r['relative_change_Z0_from_previous']) for r in z0_rows]}\n"
        f"Hammerstad-Jensen variant: finite strip-thickness correction from Capacitor2d Appendix D; epsilon_eff={hj.epsilon_eff:.16e}; Z0={hj.Z0_ohm:.16e}; percent difference from FEM={hj_row['percent_difference_Z0_vs_FEM']:.16e}\n"
        f"IPC variant: IPC-2141 compact external microstrip expression with natural logarithm; Z0={ipc.Z0_ohm:.16e}; percent difference from FEM={ipc_row['percent_difference_Z0_vs_FEM']:.16e}\n"
        f"50-ohm design comparison: difference={difference_from_50:.16e} ohm; percent={percent_difference_from_50:.16e}\n"
        f"Gate-F spatial capacitance indicator: {inputs['spatial_indicator']:.16e}\n"
        f"Gate-C domain indicator: {inputs['domain_indicator']:.16e}\n"
        f"Gate-G epsilon_eff M6-to-M7 change: {abs(z0_rows[-1]['epsilon_eff'] - z0_rows[-2]['epsilon_eff']) / abs(z0_rows[-1]['epsilon_eff']):.16e}\n"
        f"Z0 pair sensitivity indicator, not an error bar: {z0_pair_sensitivity_indicator:.16e}\n"
        "Limitations: quasistatic/quasi-TEM model, no frequency dispersion, no dielectric/conductor loss, no roughness, perfect conductor trace, finite computational domain, Gate F M7 resource fallback, and approximate engineering formulas.\n"
        f"Gate H status: {'PASS' if gate_h_pass else 'NOT YET COMPLETE'}\n"
    )
    (root / "logs/gate_H_impedance.txt").write_text(log)
    return {
        "gate_h_status": "PASS" if gate_h_pass else "NOT YET COMPLETE",
        "FINAL_MESH_ID": inputs["FINAL_MESH_ID"],
        "C_dielectric": c_dielectric,
        "C_air": c_air,
        "epsilon_eff": epsilon_eff,
        "L_H_per_m": fem["L_H_per_m"],
        "phase_velocity_m_per_s": fem["phase_velocity_m_per_s"],
        "Z0_FEM_ohm": fem["Z0_FEM_ohm"],
        "Z0_from_epsilon_ohm": fem["Z0_from_epsilon_ohm"],
        "Z0_from_LC_ohm": fem["Z0_from_LC_ohm"],
        "internal_Z0_consistency_error": fem["internal_Z0_consistency_error"],
        "internal_velocity_consistency_error": fem["internal_velocity_consistency_error"],
        "Z0_M5": z0_rows[0]["Z0"],
        "Z0_M6": z0_rows[1]["Z0"],
        "Z0_M7": z0_rows[2]["Z0"],
        "Z0_M6_change": z0_rows[1]["relative_change_Z0_from_previous"],
        "Z0_M7_change": z0_rows[2]["relative_change_Z0_from_previous"],
        "HJ_epsilon_eff": hj.epsilon_eff,
        "HJ_Z0": hj.Z0_ohm,
        "HJ_percent_difference": hj_row["percent_difference_Z0_vs_FEM"],
        "IPC_Z0": ipc.Z0_ohm,
        "IPC_percent_difference": ipc_row["percent_difference_Z0_vs_FEM"],
        "difference_from_50ohm": difference_from_50,
        "percent_difference_from_50ohm": percent_difference_from_50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_gate_h(args.project_root, quick=args.quick))


if __name__ == "__main__":
    main()
