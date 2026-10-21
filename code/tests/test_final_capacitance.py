import csv
from pathlib import Path

import numpy as np


def test_final_capacitance_outputs_if_present_are_consistent():
    final_csv = Path("tables/final_dielectric_capacitance.csv")
    mesh_path = Path("data/processed/final_engineering_mesh.npz")
    if not final_csv.exists() or not mesh_path.exists():
        return
    row = next(csv.DictReader(final_csv.open()))
    assert row["FINAL_MESH_ID"].startswith("uniform_M")
    assert row["gate_f_status"] == "PASS"
    assert row["acceptance_path"] in {"M8_PREFERRED_QC", "M7_ASYMPTOTIC_RESOURCE_FALLBACK"}
    assert row["status"] == "FINAL"
    assert not Path(row["mesh_artifact"]).is_absolute()
    assert not row["mesh_artifact"].startswith("/Users/")
    assert ":" not in Path(row["mesh_artifact"]).parts[0]
    assert Path(row["mesh_artifact"]).exists()
    assert row["domain"] == "W/H=8, Hair/H=8, homogeneous Neumann"
    assert float(row["CE"]) > 0.0
    assert float(row["CQ"]) > 0.0
    assert float(row["delta_EQ"]) < 1e-10
    assert float(row["spatial_indicator"]) > 0.0
    assert float(row["domain_indicator"]) > 0.0
    assert abs(float(row["CE"]) - float(row["Richardson_C_ext"])) > 0.0
    data = np.load(mesh_path, allow_pickle=False)
    assert str(data["FINAL_MESH_ID"]) == row["FINAL_MESH_ID"]
    assert str(data["status"]) == "FINAL"
    for key in ["points","elements","element_regions","ground_nodes","trace_nodes","far_nodes","conductor_polygon","W_over_H","Hair_over_H","H","w","t","V0"]:
        assert key in data.files
    assert data["points"].shape[0] == int(row["DOFs"])
    assert data["elements"].shape[0] == int(row["elements"])
    assert data["ground_nodes"].size > 0
    assert data["trace_nodes"].size > 0
    assert data["far_nodes"].size > 0
    assert np.isclose(float(data["W_over_H"]), 8.0)
    assert np.isclose(float(data["Hair_over_H"]), 8.0)


def test_candidate_and_final_artifact_state_is_consistent():
    final_csv = Path("tables/final_dielectric_capacitance.csv")
    candidate_csv = Path("tables/candidate_dielectric_capacitance.csv")
    final_mesh = Path("data/processed/final_engineering_mesh.npz")
    candidate_mesh = Path("data/processed/candidate_engineering_mesh.npz")
    if final_csv.exists():
        row = next(csv.DictReader(final_csv.open()))
        assert row["gate_f_status"] == "PASS"
        assert final_mesh.exists()
        assert not candidate_csv.exists()
    if candidate_csv.exists():
        row = next(csv.DictReader(candidate_csv.open()))
        assert row["status"] == "CANDIDATE_ONLY"
        assert row["gate_f_status"] != "PASS"
        assert candidate_mesh.exists()


def test_final_capacitance_sequence_is_monotone_if_present():
    path = Path("tables/final_capacitance_convergence.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    ces = [float(r["CE"]) for r in rows if r["family"] == "uniform"]
    assert len(ces) >= 4
    assert all(ces[i] >= ces[i + 1] for i in range(len(ces) - 1))
    m7 = [r for r in rows if r["mesh_id"] == "M7"][0]
    assert m7["role"] == "final_selected"
    assert m7["status"] == "FINAL_SELECTED"


def test_gate_h_not_started_outputs_absent():
    assert not Path("tables/engineering_validation.csv").exists()
