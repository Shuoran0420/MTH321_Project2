import numpy as np

from baseline import gate_b_metrics
from capacitance import extract_capacitance
from fem_solver import solve_microstrip
from geometry import build_baseline_geometry
from materials import relative_permittivity


def test_material_tags_are_piecewise_not_averaged():
    geom = build_baseline_geometry()
    eps_r = relative_permittivity(geom.element_regions)
    assert set(np.unique(eps_r)) == {1.0, 4.4}
    centroids = geom.points[geom.elements].mean(axis=1)
    assert np.all(eps_r[centroids[:, 1] < geom.config.H - 1e-13] == 4.4)
    assert np.all(eps_r[centroids[:, 1] > geom.config.H + 1e-13] == 1.0)
    assert np.all(eps_r > 0.0)


def test_baseline_solution_and_capacitance_are_finite():
    sol = solve_microstrip()
    cap = extract_capacitance(sol)
    metrics = gate_b_metrics(sol, cap)
    assert metrics["min_phi"] >= -1e-10
    assert metrics["max_phi"] <= 1.0 + 1e-10
    assert metrics["matrix_symmetry_error"] < 1e-13
    assert metrics["lambda_min"] > 0.0
    assert metrics["lambda_max"] > metrics["lambda_min"]
    assert np.isfinite(metrics["condition_estimate"])
    assert metrics["condition_estimate"] > 1.0
    assert np.isfinite(metrics["scaled_condition_estimate"])
    assert metrics["scaled_condition_estimate"] > 1.0
    assert metrics["residual_inf"] < 1e-18
    assert np.isfinite(cap.CE) and cap.CE > 0.0
    assert np.isfinite(cap.CQ) and cap.CQ > 0.0
    assert cap.delta_EQ < 1e-10
