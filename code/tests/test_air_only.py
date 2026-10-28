import csv
from pathlib import Path

import numpy as np

from gate_g_air import FINAL_MESH_ID, final_mesh_fingerprint, load_frozen_geometry


def test_gate_f_final_mesh_artifact_is_frozen_and_fingerprint_reproducible():
    mesh_path = Path("data/processed/final_engineering_mesh.npz")
    csv_path = Path("tables/final_dielectric_capacitance.csv")
    assert mesh_path.exists()
    assert csv_path.exists()
    row = next(csv.DictReader(csv_path.open()))
    geom, data, _ = load_frozen_geometry(Path("."))
    assert str(data["status"]) == "FINAL"
    assert str(data["FINAL_MESH_ID"]) == row["FINAL_MESH_ID"] == FINAL_MESH_ID
    assert geom.elements.shape[0] == int(row["elements"])
    assert geom.points.shape[0] == int(row["DOFs"])
    assert final_mesh_fingerprint(data) == final_mesh_fingerprint(data)


def test_gate_g_outputs_if_present_are_matched_and_physical():
    eff_path = Path("tables/final_effective_permittivity.csv")
    air_path = Path("tables/final_air_capacitance.csv")
    conv_path = Path("tables/epsilon_eff_convergence.csv")
    matched_path = Path("tables/matched_dielectric_air.csv")
    if not eff_path.exists():
        return
    assert air_path.exists()
    assert conv_path.exists()
    assert matched_path.exists()
    row = next(csv.DictReader(eff_path.open()))
    assert row["FINAL_MESH_ID"] == FINAL_MESH_ID
    assert row["Gate_G_status"] == "PASS"
    assert float(row["dielectric_replay_relative_difference"]) <= 1e-10
    assert row["same_mesh_PASS"] == "True"
    assert row["same_DOFs_PASS"] == "True"
    assert row["same_boundary_sets_PASS"] == "True"
    assert row["same_sparsity_pattern_PASS"] == "True"
    assert row["physical_ordering_PASS"] == "True"
    assert row["physical_range_PASS"] == "True"
    assert float(row["C_dielectric"]) > float(row["C_air"]) > 0.0
    assert 1.0 < float(row["epsilon_eff"]) < 4.4
    assert float(row["air_delta_EQ"]) < 1e-9

    air = next(csv.DictReader(air_path.open()))
    assert air["status"] == "FINAL_MATCHED_AIR"
    assert float(air["CE_air"]) > 0.0
    assert float(air["CQ_air"]) > 0.0

    eps_rows = list(csv.DictReader(conv_path.open()))
    assert [r["mesh_id"] for r in eps_rows] == ["M5", "M6", "M7"]
    assert all(r["same_mesh_pair_PASS"] == "True" for r in eps_rows)
    assert all(np.isfinite(float(r["epsilon_eff"])) for r in eps_rows)


def test_no_gate_h_outputs_are_created_by_gate_g():
    assert not Path("tables/engineering_validation.csv").exists()
