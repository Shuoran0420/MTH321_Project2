import csv
import inspect
import math
from pathlib import Path

import numpy as np

from impedance import (
    C0,
    add_hj_discrepancies,
    final_impedance_quantities,
    hammerstad_jensen_microstrip,
    ipc2141_microstrip,
    load_authoritative_inputs,
)


def test_specification_discrepancy_uses_hj_denominator_and_keeps_old_fields():
    rows = [
        {"method": "FEM", "Z0_ohm": 60.0, "percent_difference_Z0_vs_FEM": 0.0},
        {"method": "Hammerstad_Jensen", "Z0_ohm": 50.0, "percent_difference_Z0_vs_FEM": 100.0 / 6.0},
        {"method": "IPC_2141", "Z0_ohm": 45.0, "percent_difference_Z0_vs_FEM": 25.0},
    ]
    corrected = add_hj_discrepancies(rows)
    assert [r["percent_discrepancy_Z0_vs_HJ"] for r in corrected] == [20.0, 0.0, 10.0]
    assert [r["percent_difference_Z0_vs_FEM"] for r in corrected] == [r["percent_difference_Z0_vs_FEM"] for r in rows]


def test_authoritative_gate_g_inputs_load_correctly():
    inputs = load_authoritative_inputs(Path("."))
    assert inputs["gate_f_status"] == "PASS"
    assert inputs["gate_g_status"] == "PASS"
    assert inputs["FINAL_MESH_ID"] == "uniform_M7_8H_neumann_P1_v1"
    assert inputs["C_dielectric"] > inputs["C_air"] > 0.0
    assert 1.0 < inputs["epsilon_eff"] < 4.4


def test_quasi_tem_impedance_identities_agree():
    inputs = load_authoritative_inputs(Path("."))
    q = final_impedance_quantities(inputs["C_dielectric"], inputs["C_air"], inputs["epsilon_eff"])
    assert q["Z0_FEM_ohm"] > 0.0
    assert q["L_H_per_m"] > 0.0
    assert 0.0 < q["phase_velocity_m_per_s"] < C0
    assert q["internal_Z0_consistency_error"] < 1e-12
    assert q["internal_velocity_consistency_error"] < 1e-12


def test_hammerstad_jensen_formula_is_finite_physical_and_scale_invariant():
    er = 4.4
    w = 3.0e-3
    H = 1.6e-3
    t = 0.035e-3
    a = hammerstad_jensen_microstrip(er, w, H, t)
    b = hammerstad_jensen_microstrip(er, w * 1000.0, H * 1000.0, t * 1000.0)
    assert a.Z0_ohm > 0.0
    assert 1.0 < a.epsilon_eff < er
    assert np.isclose(a.Z0_ohm, b.Z0_ohm)
    assert np.isclose(a.epsilon_eff, b.epsilon_eff)
    assert np.isclose(a.u, 1.875)


def test_ipc2141_formula_is_finite_and_scale_invariant():
    er = 4.4
    w = 3.0e-3
    H = 1.6e-3
    t = 0.035e-3
    a = ipc2141_microstrip(er, w, H, t)
    b = ipc2141_microstrip(er, w * 1000.0, H * 1000.0, t * 1000.0)
    assert a.Z0_ohm > 0.0
    assert a.log_argument > 1.0
    assert math.isclose(a.Z0_ohm, b.Z0_ohm)


def test_benchmark_function_signatures_do_not_accept_fem_outputs():
    banned = {"C_dielectric", "C_air", "epsilon_eff_FEM", "Z0_FEM", "c_dielectric", "c_air", "z0_fem"}
    for fn in [hammerstad_jensen_microstrip, ipc2141_microstrip]:
        assert not banned.intersection(inspect.signature(fn).parameters)


def test_gate_h_outputs_if_present_are_consistent():
    path = Path("tables/final_characteristic_impedance.csv")
    if not path.exists():
        return
    row = next(csv.DictReader(path.open()))
    assert row["Gate_H_status"] == "PASS"
    z0 = float(row["Z0_FEM_ohm"])
    assert z0 > 0.0
    assert float(row["internal_Z0_consistency_error"]) < 1e-12
    assert float(row["internal_velocity_consistency_error"]) < 1e-12
    assert math.isclose(float(row["Z0_from_epsilon_ohm"]), z0)
    assert math.isclose(float(row["Z0_from_LC_ohm"]), z0)
    assert 0.0 < float(row["phase_velocity_over_c0"]) < 1.0
    for required in [
        "tables/Z0_convergence.csv",
        "tables/engineering_benchmarks.csv",
        "tables/final_engineering_results.csv",
    ]:
        assert Path(required).exists()
    conv = list(csv.DictReader(Path("tables/Z0_convergence.csv").open()))
    assert [r["mesh_id"] for r in conv] == ["M5", "M6", "M7"]
    assert math.isclose(float(conv[-1]["Z0"]), z0)
    bench = list(csv.DictReader(Path("tables/engineering_benchmarks.csv").open()))
    assert {r["method"] for r in bench} == {"FEM", "Hammerstad_Jensen", "IPC_2141"}
    assert all(float(r["Z0_ohm"]) > 0.0 for r in bench)
