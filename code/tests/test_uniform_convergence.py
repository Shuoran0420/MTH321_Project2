import csv
import inspect
import math
from pathlib import Path

import numpy as np

from uniform_convergence import build_uniform_mesh, energy_error, fit_slope, solve_uniform


def test_gate_d_fixed_domain_and_reference_hierarchy():
    meshes = [build_uniform_mesh(level) for level in range(5)]
    cfg = meshes[0].geom.config
    assert cfg.W_over_H == 8.0
    assert cfg.Hair_over_H == 8.0
    assert np.isclose(cfg.W, 8.0 * cfg.H)
    assert np.isclose(cfg.Hair, 8.0 * cfg.H)
    assert np.isclose(cfg.trace_xmax - cfg.trace_xmin, cfg.w)
    assert np.isclose(cfg.trace_ymax - cfg.trace_ymin, cfg.t)
    for mesh in meshes[1:]:
        other = mesh.geom.config
        assert other.W_over_H == cfg.W_over_H
        assert other.Hair_over_H == cfg.Hair_over_H
        assert np.isclose(other.trace_xmin, cfg.trace_xmin)
        assert np.isclose(other.trace_xmax, cfg.trace_xmax)
        assert np.isclose(other.trace_ymin, cfg.trace_ymin)
        assert np.isclose(other.trace_ymax, cfg.trace_ymax)
    elements = [mesh.geom.elements.shape[0] for mesh in meshes]
    dofs = [mesh.geom.points.shape[0] for mesh in meshes]
    assert all(elements[i + 1] > elements[i] for i in range(len(elements) - 1))
    assert all(dofs[i + 1] > dofs[i] for i in range(len(dofs) - 1))


def test_cross_mesh_energy_error_is_finite_on_nested_meshes():
    coarse = solve_uniform(0)
    ref = solve_uniform(2)
    err_abs, err_rel = energy_error(coarse, ref)
    assert math.isfinite(err_abs) and err_abs > 0.0
    assert math.isfinite(err_rel) and err_rel > 0.0


def test_no_direct_different_mesh_coefficient_vector_subtraction():
    source = inspect.getsource(energy_error)
    assert "phi -" not in source
    assert "- phi" not in source
    assert "element_gradients[e] - g_coarse" in source


def test_uniform_outputs_if_present_are_consistent():
    conv = Path("tables/uniform_convergence.csv")
    stab = Path("tables/reference_stability.csv")
    if not conv.exists() or not stab.exists():
        return
    rows = list(csv.DictReader(conv.open()))
    assert len(rows) >= 4
    h = [float(r["h_max_m"]) for r in rows]
    elems = [int(r["elements"]) for r in rows]
    errs = [float(r["relative_energy_error"]) for r in rows]
    assert all(elems[i + 1] > elems[i] for i in range(len(elems) - 1))
    assert all(h[i + 1] < h[i] for i in range(len(h) - 1))
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    assert all(float(r["CE_F_per_m"]) > 0.0 and float(r["CQ_F_per_m"]) > 0.0 for r in rows)
    assert all(float(r["delta_EQ"]) < 1e-10 for r in rows)
    orders = [float(r["observed_order_to_next"]) for r in rows[:-1] if r["observed_order_to_next"]]
    assert all(math.isfinite(p) for p in orders)
    slope, fit_dev = fit_slope(rows, [0, 1, 2])
    assert math.isfinite(slope)
    assert math.isfinite(fit_dev)
    assert all(r["reference_id"] == "R2=M6" for r in rows)
    assert max(float(r["reference_sensitivity"]) for r in rows[:3]) <= 0.03
    stab_rows = list(csv.DictReader(stab.open()))
    assert any(r["analysis_mesh_id"] == "R1_summary" for r in stab_rows)
    assert any(r["analysis_mesh_id"] == "R2_summary" for r in stab_rows)
    assert "error_using_R2" in stab_rows[0]
    assert "sensitivity_R1_to_R2" in stab_rows[0]
