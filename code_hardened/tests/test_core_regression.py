"""Run with `pytest -q tests/test_core_regression.py` from repository root.

Add this file to the existing tests directory or run with PYTHONPATH=code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
import pytest
from geometry import GeometryConfig, build_geometry
from materials import MaterialConfig, relative_permittivity
from fem_solver import assemble_stiffness, solve_microstrip
from capacitance import extract_capacitance


def test_unknown_material_fails():
    with pytest.raises(ValueError, match='unknown material'):
        relative_permittivity(np.array(['substrate', 'vacum']))


def test_nonphysical_permittivity_fails():
    with pytest.raises(ValueError, match='finite and positive'):
        relative_permittivity(np.array(['air']), MaterialConfig(epsilon_r_air=0.0))


def test_small_mesh_potential_and_charge():
    cfg = GeometryConfig(nx_outer=12, ny_substrate=5, ny_air=8)
    geom = build_geometry(cfg)
    K, *_ = assemble_stiffness(geom)
    # An exactly symmetric matrix leaves (K-K.T) with zero stored entries, so
    # np.max(...) on .data would raise; use a guarded max instead.
    asym = abs(K - K.T)
    assert (asym.nnz == 0) or (np.max(np.abs(asym.data)) < 1e-20)
    sol = solve_microstrip(geom, compute_conditioning=False)
    assert np.allclose(sol.phi[sol.trace_dofs], cfg.V0)
    assert np.allclose(sol.phi[sol.ground_dofs], 0.0)
    assert np.min(sol.phi) >= -1e-8
    assert np.max(sol.phi) <= cfg.V0 + 1e-8
    rhs_norm = max(np.linalg.norm(sol.rhs_free, np.inf), np.finfo(float).tiny)
    assert np.linalg.norm(sol.matrix_free @ sol.phi[sol.free_dofs] - sol.rhs_free, np.inf)/rhs_norm < 1e-8
    cap = extract_capacitance(sol)
    assert cap.CE > 0 and cap.CQ > 0
    assert cap.delta_EQ < 1e-8
    assert cap.charge_balance_defect < 1e-8
