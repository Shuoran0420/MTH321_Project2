import csv
import math
from pathlib import Path

import numpy as np

from graded_study import GRADED_SPECS, build_graded_mesh, common_refinement_check, common_refinement_diagnostics, local_size_diagnostics
from uniform_convergence import build_uniform_mesh


def test_gate_e_fixed_domain_geometry_and_symmetry():
    mesh = build_graded_mesh(GRADED_SPECS[2])
    cfg = mesh.geom.config
    assert cfg.W_over_H == 8.0
    assert cfg.Hair_over_H == 8.0
    assert np.isclose(cfg.trace_xmax - cfg.trace_xmin, cfg.w)
    assert np.isclose(cfg.trace_ymax - cfg.trace_ymin, cfg.t)
    pts = {(round(float(x), 15), round(float(y), 15)) for x, y in mesh.geom.points}
    for x, y in mesh.geom.points:
        assert (round(float(-x), 15), round(float(y), 15)) in pts


def test_gate_e_refinement_is_local_but_coordinate_nesting_is_not_triangle_nesting():
    r2 = build_uniform_mesh(6)
    meshes = [build_graded_mesh(spec) for spec in GRADED_SPECS]
    dofs = [m.geom.points.shape[0] for m in meshes]
    nnz_proxy = [m.geom.elements.shape[0] for m in meshes]
    corner_sizes = [local_size_diagnostics(m)[0] for m in meshes]
    far_sizes = [local_size_diagnostics(m)[1] for m in meshes]
    assert all(dofs[i + 1] > dofs[i] for i in range(len(dofs) - 1))
    assert all(nnz_proxy[i + 1] > nnz_proxy[i] for i in range(len(nnz_proxy) - 1))
    assert corner_sizes[-1] < corner_sizes[0]
    assert far_sizes[-1] < far_sizes[0]
    assert corner_sizes[-1] < far_sizes[-1]
    checks = [common_refinement_diagnostics(m, r2) for m in meshes]
    assert all(c["coordinate_subset"] for c in checks)
    assert all(c["crossed_triangles"] > 0 for c in checks)
    assert all(not common_refinement_check(m, r2) for m in meshes)


def test_common_refinement_check_accepts_genuine_uniform_refinement():
    assert common_refinement_check(build_uniform_mesh(0), build_uniform_mesh(2))


def test_invalid_common_reference_is_rejected_before_any_fem_solve(monkeypatch, tmp_path):
    import graded_study
    import pytest

    def forbidden_solve(*args, **kwargs):
        pytest.fail("A solve was attempted before common-refinement validation")

    monkeypatch.setattr(graded_study, "solve_uniform", forbidden_solve)
    with pytest.raises(RuntimeError, match="common-overlay"):
        graded_study.run_graded_study(tmp_path, quick=True)


def test_gate_e_outputs_if_present_are_consistent():
    graded = Path("tables/graded_convergence.csv")
    combined = Path("tables/uniform_vs_graded.csv")
    targets = Path("tables/efficiency_targets.csv")
    if not graded.exists() or not combined.exists() or not targets.exists():
        return
    rows = list(csv.DictReader(graded.open()))
    assert len(rows) >= 4
    assert all(float(r["relative_energy_error"]) > 0.0 and math.isfinite(float(r["relative_energy_error"])) for r in rows)
    assert all(0.0 <= float(r["potential_min"]) <= 1.0 for r in rows)
    assert all(0.0 <= float(r["potential_max"]) <= 1.0 for r in rows)
    assert all(float(r["CE_F_per_m"]) > 0.0 and float(r["CQ_F_per_m"]) > 0.0 for r in rows)
    assert all(float(r["delta_EQ"]) < 1e-10 for r in rows)
    assert all(float(r["residual_inf"]) < 1e-20 for r in rows)
    assert all(math.isfinite(float(r["reference_sensitivity"])) for r in rows)
    combo = list(csv.DictReader(combined.open()))
    assert {r["method"] for r in combo} == {"uniform", "graded"}
    target_rows = list(csv.DictReader(targets.open()))
    combo_by_method = {
        method: [float(r["relative_energy_error"]) for r in combo if r["method"] == method]
        for method in {"uniform", "graded"}
    }
    for row in target_rows:
        if row.get("integration_validation_status") == "PENDING_COMMON_OVERLAY":
            assert row["formal_claim_allowed_current"] == "False"
        target = float(row["target_relative_error"])
        if not math.isfinite(target):
            continue
        for method in {"uniform", "graded"}:
            vals = combo_by_method[method]
            assert min(vals) <= target <= max(vals)
