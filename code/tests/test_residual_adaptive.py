"""Independent checks before residual-adaptive experiment interpretation."""

from types import SimpleNamespace
from pathlib import Path
import json
import hashlib
import csv

import numpy as np
from skfem import MeshTri

from adaptive_mesh import geometry_from_mesh, refine, symmetry_complete, symmetry_error
from capacitance import extract_capacitance
from fem_solver import solve_microstrip
from materials import permittivity
from overlay_energy import overlay_energy_error
from residual_indicator import doerfler_mark, flux_indicator
from uniform_convergence import build_uniform_mesh


def gradients(points, elements, values):
    t = points[elements]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    u, v = values[elements[:, 1]] - values[elements[:, 0]], values[elements[:, 2]] - values[elements[:, 0]]
    return np.column_stack(((u * b[:, 1] - v * a[:, 1]) / det,
                            (v * a[:, 0] - u * b[:, 0]) / det))


def test_smooth_flux_jump_decreases():
    errors = []
    for n in (4, 8, 16, 32):
        mesh = MeshTri.init_tensor(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        p, t = mesh.p.T, mesh.t.T
        g = SimpleNamespace(mesh=mesh, points=p, elements=t, config=SimpleNamespace(W=10, Hair=10))
        u = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
        z = flux_indicator(g, gradients(p, t, u), np.ones(len(t)))
        errors.append(z.eta)
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] / errors[-2] < 0.6


def test_interface_uses_material_weighted_flux():
    g = build_uniform_mesh(1).geom
    grad = np.column_stack((np.zeros(len(g.elements)), np.where(g.element_regions == "substrate", 1 / 4.4, 1)))
    z = flux_indicator(g, grad, permittivity(g.element_regions))
    mid = g.points[g.mesh.facets.T].mean(axis=1)
    interface = z.interior & np.isclose(mid[:, 1], g.config.H, atol=1e-14, rtol=0)
    assert interface.any()
    assert np.max(np.abs(z.edge_jump[interface])) < 1e-25
    raw = flux_indicator(g, grad, np.ones(len(g.elements)))
    assert np.max(np.abs(raw.edge_jump[interface])) > 0.5


def test_orientation_far_residual_and_dirichlet_exclusion():
    g = build_uniform_mesh(0).geom
    grad = np.tile([1., 2.], (len(g.elements), 1))
    z = flux_indicator(g, grad, np.ones(len(g.elements)))
    assert np.max(np.abs(z.edge_jump[z.interior])) < 1e-14
    assert z.far.any() and z.dirichlet.any()
    edges = g.mesh.facets.T
    d = g.points[edges[:, 1]] - g.points[edges[:, 0]]
    expected = np.sum((2 * d[z.far, 0] - d[z.far, 1])**2)
    assert np.isclose(z.element_squared.sum(), expected)
    arbitrary = grad * np.arange(1, len(grad) + 1)[:, None]
    before = flux_indicator(g, arbitrary, np.ones(len(grad)))
    f2t = g.mesh.f2t.copy()
    interior = f2t[1] >= 0
    f2t[:, interior] = f2t[::-1, interior]
    g.mesh._f2t = f2t
    swapped = flux_indicator(g, arbitrary, np.ones(len(grad)))
    assert np.allclose(abs(before.edge_jump[before.interior]), abs(swapped.edge_jump[swapped.interior]))
    assert np.allclose(before.element_squared, swapped.element_squared)
    # Independently reverse both side order and normal for arbitrary flux pairs.
    a, b = np.array([2., 3.]), np.array([-4., 1.])
    n = np.array([0.6, 0.8])
    assert abs((a - b) @ n) == abs((b - a) @ n)
    assert np.isfinite(swapped.eta)


def test_doerfler_is_minimal_and_deterministic():
    a = np.array([1., 4., 2., 3.])
    marked = doerfler_mark(a)
    assert np.array_equal(marked, [1, 3])
    assert a[marked].sum() >= 0.5 * a.sum()
    assert a[marked].sum() - a[marked].min() < 0.5 * a.sum()
    assert not len(doerfler_mark(np.zeros(5)))


def test_exact_overlay_crossing_diagonals():
    p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    def field(t):
        g = SimpleNamespace(points=p, elements=t, config=SimpleNamespace(H=1), element_regions=np.array(["air", "air"]))
        return SimpleNamespace(geometry=g, element_gradients=gradients(p, t, p[:, 0] * p[:, 1]),
                               element_areas=np.array([.5, .5]), epsilon=np.ones(2))
    a = field(np.array([[0, 1, 2], [0, 2, 3]]))
    b = field(np.array([[0, 1, 3], [1, 2, 3]]))
    e = overlay_energy_error(a, b)
    assert e.crossed_triangles == 2
    assert np.isclose(e.absolute, 1, rtol=1e-13)
    assert e.overlay_area_relative_defect < 1e-13
    assert np.isclose(overlay_energy_error(b, a).absolute, e.absolute, rtol=1e-13)


def test_symmetry_indicator_and_conforming_refinement():
    g = symmetry_complete(build_uniform_mesh(1).geom)
    for k in range(3):
        sol = solve_microstrip(g, compute_conditioning=False)
        z = flux_indicator(g, sol.element_gradients, sol.epsilon)
        centers = g.points[g.elements].mean(axis=1)
        lookup = {tuple(np.round(q, 14)): v for q, v in zip(centers, z.element_squared)}
        reflected = np.array([lookup[(round(-q[0], 14), round(q[1], 14))] for q in centers])
        assert np.linalg.norm(z.element_squared - reflected) / np.linalg.norm(z.element_squared) < 1e-10
        assert symmetry_error(sol) < 1e-10
        cap = extract_capacitance(sol)
        assert cap.delta_EQ < 1e-10 and cap.charge_balance_defect < 1e-10
        assert sol.phi.min() > -1e-12 and sol.phi.max() < 1 + 1e-12
        assert np.isclose(sol.element_areas.sum(), 2 * g.config.W * g.config.Hair - g.config.w * g.config.t, rtol=1e-13)
        # Every one-sided edge is on a physical or artificial domain boundary.
        mid = g.points[g.mesh.facets.T].mean(axis=1)
        boundary = g.mesh.f2t[1] < 0
        c = g.config
        on_outer = np.isclose(mid[:, 1], 0) | np.isclose(mid[:, 1], c.Hair) | np.isclose(abs(mid[:, 0]), c.W)
        on_trace = ((abs(mid[:, 0]) <= c.w / 2 + 1e-14) & (np.isclose(mid[:, 1], c.H, rtol=0, atol=1e-14) | np.isclose(mid[:, 1], c.H + c.t, rtol=0, atol=1e-14))) | ((mid[:, 1] >= c.H - 1e-14) & (mid[:, 1] <= c.H + c.t + 1e-14) & np.isclose(abs(mid[:, 0]), c.w / 2))
        assert np.all((on_outer | on_trace)[boundary])
        tri_y = g.points[g.elements][:, :, 1]
        assert not np.any((tri_y.min(axis=1) < c.H - 1e-13) & (tri_y.max(axis=1) > c.H + 1e-13))
        g, actual, closed = refine(g, doerfler_mark(z.element_squared))
        assert actual >= closed > 0


def test_frozen_old_numerical_files_unchanged():
    root = Path(__file__).resolve().parents[2]
    snapshot = json.loads((root / "logs/adaptive_frozen_inputs.json").read_text())
    for name, digest in snapshot.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_corrected_initial_mesh_is_nested_m2_resolution():
    parent = build_uniform_mesh(1).geom
    completed = symmetry_complete(parent)
    original_m2 = build_uniform_mesh(2).geom
    assert len(completed.points) == len(original_m2.points) == 404
    def ordered(p):
        return p[np.lexsort((p[:, 1], p[:, 0]))]
    assert np.allclose(ordered(completed.points), ordered(original_m2.points), rtol=0, atol=1e-17)
    assert len(completed.elements) == 4 * len(parent.elements)
    tri = completed.points[completed.elements]
    for j in range(3):
        dot = np.sum((tri[:, (j + 1) % 3] - tri[:, j]) * (tri[:, (j + 2) % 3] - tri[:, j]), axis=1)
        assert dot.min() > -1e-20


def test_primary_parameter_and_failed_attempt_are_preserved():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "config/adaptive_extension.json").read_text())
    assert cfg["theta"] == .5 and cfg["cycles"] >= 6
    assert cfg["primary_reference"] == "M6" and cfg["sensitivity_reference"] == "M7"
    assert cfg["formal_reference_sensitivity_limit"] == .03
    old = root / "results/adaptive_attempt_01_invalid_companion/adaptive_refinement_results.csv"
    rows = list(csv.DictReader(old.open()))
    assert len(rows) == 16
    assert any(float(r["potential_min"]) < -.01 for r in rows if r["level"].startswith("B"))


def test_overlay_reproduces_uniform_nested_energy_identity():
    coarse = solve_microstrip(build_uniform_mesh(0).geom, compute_conditioning=False)
    fine = solve_microstrip(build_uniform_mesh(2).geom, compute_conditioning=False)
    e = overlay_energy_error(coarse, fine)
    exact = extract_capacitance(coarse).CE - extract_capacitance(fine).CE
    assert np.isclose(e.absolute**2, exact, rtol=1e-12)


def extension_rows(name):
    root = Path(__file__).resolve().parents[2]
    return list(csv.DictReader((root / "results/adaptive_extension" / name).open()))


def test_all_corrected_cycles_pass_integrity_and_timing_counts():
    rows = extension_rows("adaptive_refinement_results.csv")
    assert len(rows) == 16
    samples = extension_rows("adaptive_timing_samples.csv")
    for r in rows:
        assert float(r["potential_min"]) >= -1e-10 and float(r["potential_max"]) <= 1 + 1e-10
        assert float(r["symmetry_error"]) < 1e-10
        assert float(r["normalized_residual"]) < 1e-10
        assert float(r["matrix_symmetry_error"]) < 1e-12
        assert float(r["charge_balance_defect"]) < 1e-9
        assert float(r["delta_EQ"]) < 1e-9
        assert r["geometry_material_conformity_PASS"] == "True"
        assert float(r["overlay_area_defect"]) < 1e-7
        assert sum(s["path"] == r["level"][0] and s["cycle"] == r["cycle"] for s in samples) == 5
    assert all(r["status"] == "PASS" for r in extension_rows("adaptive_validation.csv"))


def test_target_interpolation_reference_rules_and_no_extrapolation():
    from adaptive_analysis import interpolate_target
    rows = extension_rows("adaptive_matched_targets.csv")
    assert [float(r["target"]) for r in rows] == [.4, .3, .2]
    for r in rows:
        for name in ("uniform", "graded", "adaptive"):
            assert 0 <= float(r[f"{name}_weight"]) <= 1
        allowed = all(float(r[f"{name}_sensitivity"]) < .03 for name in ("uniform", "graded", "adaptive"))
        assert (r["formal_claim_allowed"] == "True") == allowed
    assert rows[-1]["formal_claim_allowed"] == "False"
    assert interpolate_target([{"E_rel_M6": .2}, {"E_rel_M6": .1}], .4) is None


def test_overlay_uniform_reproduction_and_new_graded_raw_CE():
    root = Path(__file__).resolve().parents[2]
    previous = list(csv.DictReader((root / "tables/uniform_convergence.csv").open()))
    current = extension_rows("uniform_graded_overlay_comparison.csv")
    for level in range(4):
        a = next(r for r in previous if int(r["level"]) == level)
        b = next(r for r in current if r["level"] == f"M{level}")
        assert np.isclose(float(a["relative_energy_error"]), float(b["E_rel_M6"]), rtol=1e-11)
    old_graded = list(csv.DictReader((root / "tables/graded_convergence.csv").open()))
    for r in old_graded:
        name = r["graded_level"]
        b = next(t for t in current if t["level"] == name)
        assert np.isclose(float(r["CE_F_per_m"]), float(b["CE"]), rtol=1e-12)


def test_hypothesis_frozen_and_final_reference_replay():
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads((root / "results/adaptive_extension/experiment_metadata.json").read_text())
    assert metadata["hypothesis_sha256"] == hashlib.sha256((root / "logs/PRE_ADAPTIVE_HYPOTHESIS.md").read_bytes()).hexdigest()
    replay = extension_rows("reference_replay.csv")[-1]
    assert float(replay["relative_to_frozen_C"]) <= 1e-10
    assert replay["fingerprint"] == "06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75"


def test_fixed_family_timings_and_no_algebraic_vector_shortcut():
    rows = extension_rows("comparison_timing_samples.csv")
    for level in [f"M{i}" for i in range(6)] + [f"G{i}" for i in range(5)]:
        assert sum(r["level"] == level for r in rows) == 5
    root = Path(__file__).resolve().parents[2]
    source = (root / "code/overlay_energy.py").read_text()
    assert ".phi" not in source
    assert "shapely.intersection" in source
