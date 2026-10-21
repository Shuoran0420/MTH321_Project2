"""Standalone dielectric residual-adaptive extension; never writes old outputs.

Run: python3 code/residual_adaptive_study.py
Offline overlay validation is excluded from every algorithm's work timings.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from adaptive_mesh import refine, symmetry_complete, symmetry_error
from capacitance import extract_capacitance
from fem_solver import solve_microstrip
from gate_g_air import load_frozen_geometry, final_mesh_fingerprint
from graded_study import GRADED_SPECS, build_graded_mesh
from overlay_energy import overlay_energy_error
from residual_indicator import flux_indicator, doerfler_mark
from uniform_convergence import build_uniform_mesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/adaptive_extension"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def assert_frozen():
    old = json.loads((ROOT / "logs/adaptive_frozen_inputs.json").read_text())
    bad = [name for name, digest in old.items() if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if bad:
        raise RuntimeError(f"frozen numerical inputs changed: {bad}")


def geometry_checks(sol):
    g, c = sol.geometry, sol.geometry.config
    p = g.points[g.elements]
    area = sol.element_areas.sum()
    total = 2 * c.W * c.Hair - c.w * c.t
    mid = g.points[g.mesh.facets.T].mean(axis=1)
    outer = np.isclose(abs(mid[:, 0]), c.W, rtol=0, atol=1e-13) | np.isclose(mid[:, 1], 0, rtol=0, atol=1e-13) | np.isclose(mid[:, 1], c.Hair, rtol=0, atol=1e-13)
    trace = (((abs(mid[:, 0]) <= c.w / 2 + 1e-13) & (np.isclose(mid[:, 1], c.H, rtol=0, atol=1e-13) | np.isclose(mid[:, 1], c.H + c.t, rtol=0, atol=1e-13)))
             | ((mid[:, 1] >= c.H - 1e-13) & (mid[:, 1] <= c.H + c.t + 1e-13) & np.isclose(abs(mid[:, 0]), c.w / 2, rtol=0, atol=1e-13)))
    boundary = g.mesh.f2t[1] < 0
    return bool(c.W_over_H == c.Hair_over_H == 8 and np.isclose(area, total, rtol=1e-12)
                and np.all((outer | trace)[boundary])
                and not np.any((p[:, :, 1].min(axis=1) < c.H - 1e-13) & (p[:, :, 1].max(axis=1) > c.H + 1e-13))
                and np.all(sol.epsilon_r[g.element_regions == "substrate"] == 4.4)
                and np.all(sol.epsilon_r[g.element_regions == "air"] == 1))


def diagnostics(sol):
    cap = extract_capacitance(sol)
    residual = sol.matrix_free @ sol.phi[sol.free_dofs] - sol.rhs_free
    scale = max(np.max(abs(sol.rhs_free)), 1e-300)
    a = sol.matrix_free
    asym = a - a.T
    matrix_sym = float(np.sqrt(asym.multiply(asym).sum() / a.multiply(a).sum()))
    sym = symmetry_error(sol)
    return {
        "elements": len(sol.geometry.elements), "DOFs": len(sol.phi),
        "NNZ": a.nnz, "free_DOFs": len(sol.free_dofs),
        "CE": cap.CE, "CQ": cap.CQ, "delta_EQ": cap.delta_EQ,
        "potential_min": float(sol.phi.min()), "potential_max": float(sol.phi.max()),
        "symmetry_error": sym, "matrix_symmetry_error": matrix_sym,
        "free_residual": float(np.max(abs(residual))),
        "normalized_residual": float(np.max(abs(residual)) / scale),
        "charge_balance_defect": cap.charge_balance_defect,
        "geometry_material_conformity_PASS": geometry_checks(sol),
        "solver": "scipy.sparse.linalg.spsolve", "far_BC": "homogeneous_Neumann",
        "P1": True, "W_over_H": 8, "Hair_over_H": 8,
    }


def save_solution(name, sol, eta=None):
    path = OUT / "meshes" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=sol.geometry.points, elements=sol.geometry.elements,
                        phi=sol.phi, gradients=sol.element_gradients,
                        areas=sol.element_areas, epsilon=sol.epsilon,
                        eta_squared=np.array([]) if eta is None else eta.element_squared)


def errors(sol, refs, final_C):
    start = time.perf_counter()
    e6 = overlay_energy_error(sol, refs[0])
    e7 = overlay_energy_error(sol, refs[1])
    sensitivity = abs(e7.absolute - e6.absolute) / e7.absolute
    return {
        "E_rel_M6": e6.relative, "E_rel_M7": e7.relative,
        "absolute_error_M6": e6.absolute, "absolute_error_M7": e7.absolute,
        "reference_sensitivity": sensitivity,
        "formal_reference_stable": sensitivity < .03,
        "capacitance_relative_to_M7": abs(extract_capacitance(sol).CE - final_C) / final_C,
        "overlay_crossed_M6": e6.crossed_triangles,
        "overlay_crossed_M7": e7.crossed_triangles,
        "overlay_area_defect": max(e6.overlay_area_relative_defect, e7.overlay_area_relative_defect),
        "offline_error_integration_time_s": time.perf_counter() - start,
    }


def timed_construction(make, repetitions):
    times = []
    for i in range(repetitions + 1):
        start = time.perf_counter(); geom = make()
        if i:
            times.append(time.perf_counter() - start)
    return geom, float(np.median(times))


def adaptive_path(initial, cycles, prefix, refs, final_C, cfg):
    geom, generation_time = timed_construction(lambda: symmetry_complete(build_uniform_mesh(initial).geom), cfg["timing_repetitions"])
    rows, timings = [], []
    spent = np.full(cfg["timing_repetitions"], generation_time)
    for cycle in range(cycles):
        samples = []
        for repetition in range(cfg["timing_repetitions"] + 1):
            start = time.perf_counter()
            sol = solve_microstrip(geom, compute_conditioning=False)
            cap = extract_capacitance(sol)
            solve_wall = time.perf_counter() - start
            start = time.perf_counter()
            indicator = flux_indicator(geom, sol.element_gradients, sol.epsilon)
            estimator_time = time.perf_counter() - start
            start = time.perf_counter(); marked = doerfler_mark(indicator.element_squared, cfg["theta"])
            marking_time = time.perf_counter() - start
            start = time.perf_counter(); child, actual, closed = refine(geom, marked)
            refinement_time = time.perf_counter() - start
            if repetition:
                sample = {"path": prefix, "cycle": cycle, "repetition": repetition,
                          "assembly_time": sol.assembly_time, "solve_time": sol.solve_time,
                          "solve_wall_time": solve_wall, "estimator_time": estimator_time,
                          "marking_time": marking_time, "refinement_time": refinement_time,
                          "total_time": spent[repetition - 1] + solve_wall}
                samples.append(sample); timings.append(sample)
        med = {key: float(np.median([s[key] for s in samples])) for key in samples[0] if key not in ("path", "cycle", "repetition")}
        achieved = med["total_time"]
        err = errors(sol, refs, final_C)
        row = {"method": "Residual adaptive", "level": f"{prefix}{cycle}",
               "path": f"M{initial}-S", "cycle": cycle, "initial_uniform_parent": initial,
               **diagnostics(sol), "marked_elements": len(marked),
               "symmetry_closed_marked_elements": closed, "actual_refined_elements": actual,
               "eta": indicator.eta, **err,
               "estimator_to_reference_error_ratio": indicator.eta / err["absolute_error_M6"],
               **med, "total_time": achieved,
               "cycle_time_including_outgoing_refinement": sum(med[k] for k in ("solve_wall_time", "estimator_time", "marking_time", "refinement_time")),
               "initial_mesh_time": generation_time}
        save_solution(row["level"], sol, indicator)
        rows.append(row)
        print(f"{row['level']}: DOFs={row['DOFs']} E6={row['E_rel_M6']:.6f} Sref={row['reference_sensitivity']:.4%} symmetry={row['symmetry_error']:.3e}", flush=True)
        write_csv(OUT / f"{prefix}_progress.csv", rows)
        spent = np.array([s["total_time"] + s["estimator_time"] + s["marking_time"] + s["refinement_time"] for s in samples])
        geom = child
    return rows, timings


def fixed_comparisons(refs, final_C, cfg):
    rows, timings = [], []
    specs = [("Uniform", f"M{i}", lambda i=i: build_uniform_mesh(i).geom) for i in range(6)]
    specs += [("Deterministic graded", f"G{s.level}", lambda s=s: build_graded_mesh(s).geom) for s in GRADED_SPECS]
    for method, level, make in specs:
        samples = []
        for rep in range(cfg["timing_repetitions"] + 1):
            start = time.perf_counter(); geom = make(); mesh_time = time.perf_counter() - start
            start = time.perf_counter(); sol = solve_microstrip(geom, compute_conditioning=False); extract_capacitance(sol)
            solve_wall = time.perf_counter() - start
            if rep:
                sample = {"method": method, "level": level, "repetition": rep,
                          "mesh_time": mesh_time, "assembly_time": sol.assembly_time,
                          "solve_time": sol.solve_time, "total_time": mesh_time + solve_wall}
                samples.append(sample); timings.append(sample)
        row = {"method": method, "level": level, **diagnostics(sol), **errors(sol, refs, final_C),
               **{key: float(np.median([s[key] for s in samples])) for key in ("mesh_time", "assembly_time", "solve_time", "total_time")}}
        rows.append(row); save_solution(level, sol)
        print(f"{level}: DOFs={row['DOFs']} overlay E6={row['E_rel_M6']:.6f} Sref={row['reference_sensitivity']:.4%}", flush=True)
        write_csv(OUT / "uniform_graded_overlay_comparison.csv", rows)
    return rows, timings


def run():
    assert_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/adaptive_extension.json").read_text())
    frozen, data, _ = load_frozen_geometry(ROOT)
    print("Solving unchanged M6/M7 dielectric references once; no all-air solve.", flush=True)
    r6 = solve_microstrip(build_uniform_mesh(6).geom, compute_conditioning=False)
    r7 = solve_microstrip(frozen, compute_conditioning=False)
    final_row = next(csv.DictReader((ROOT / "tables/final_dielectric_capacitance.csv").open()))
    final_C = float(final_row["CE"])
    replay = abs(extract_capacitance(r7).CE - final_C) / final_C
    if replay > 1e-10:
        raise RuntimeError("M7 reference replay changed final capacitance")
    write_csv(OUT / "reference_replay.csv", [{"mesh": "M6", "CE": extract_capacitance(r6).CE, "fingerprint": "uniform_M6", "relative_to_frozen_C": ""},
                                            {"mesh": "M7", "CE": extract_capacitance(r7).CE, "fingerprint": final_mesh_fingerprint(data), "relative_to_frozen_C": replay}])
    refs = (r6, r7)
    main, main_times = adaptive_path(cfg["initial_uniform_level"], cfg["cycles"], "A", refs, final_C, cfg)
    companion, companion_times = adaptive_path(cfg["companion_initial_uniform_level"], cfg["companion_cycles"], "B", refs, final_C, cfg)
    write_csv(OUT / "adaptive_refinement_results.csv", main + companion)
    write_csv(OUT / "adaptive_timing_samples.csv", main_times + companion_times)
    fixed, fixed_times = fixed_comparisons(refs, final_C, cfg)
    write_csv(OUT / "comparison_timing_samples.csv", fixed_times)
    # Uniform schema for the three-way work plots; old raw tables are untouched.
    keys = ["method", "level", "elements", "DOFs", "NNZ", "CE", "E_rel_M6", "E_rel_M7", "reference_sensitivity", "formal_reference_stable", "capacitance_relative_to_M7", "assembly_time", "solve_time", "total_time"]
    write_csv(OUT / "three_way_comparison.csv", [{k: row[k] for k in keys} for row in fixed + main + companion])
    assert_frozen()
    print("ADAPTIVE DATA GENERATION COMPLETE; interpretation still requires validation.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
