"""Read-only numerical analysis and report assets for the adaptive extension."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from skfem import MeshTri

from adaptive_mesh import geometry_from_mesh
from materials import EPSILON_0
from residual_indicator import doerfler_mark, flux_indicator
from residual_adaptive_study import ROOT, OUT, write_csv, assert_frozen
from uniform_convergence import build_uniform_mesh

FIG = ROOT / "figures/adaptive_extension"
LIMIT = 0.03


def read_rows(name):
    with (OUT / name).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def interpolate_target(rows, target):
    """Consecutive measured points only; no extrapolation or fitted power law."""
    for a, b in zip(rows[:-1], rows[1:]):
        ea, eb = a["E_rel_M6"], b["E_rel_M6"]
        if min(ea, eb) <= target <= max(ea, eb) and ea != eb:
            weight = np.log(target / ea) / np.log(eb / ea)
            assert -1e-12 <= weight <= 1 + 1e-12
            return {
                **{k: float(np.exp((1 - weight) * np.log(a[k]) + weight * np.log(b[k]))) for k in ("DOFs", "NNZ", "total_time")},
                "bracket": f"{a['level']}/{b['level']}",
                "sensitivity": max(a["reference_sensitivity"], b["reference_sensitivity"]),
                "weight": float(weight),
            }
    return None


def targets(all_rows):
    uniform = [r for r in all_rows if r["method"] == "Uniform"]
    graded = [r for r in all_rows if r["method"] == "Deterministic graded"]
    main = [r for r in all_rows if str(r["level"]).startswith("A")]
    companion = [r for r in all_rows if str(r["level"]).startswith("B")]
    rows = []
    for target in (.4, .3, .2):
        adaptive = companion if target > main[0]["E_rel_M6"] else main
        groups = {"uniform": uniform, "graded": graded, "adaptive": adaptive}
        bracket = {name: interpolate_target(seq, target) for name, seq in groups.items()}
        row = {"target": target, "adaptive_path": "M0-S companion" if adaptive is companion else "M1-S primary (M2-resolution)"}
        valid = all(b is not None for b in bracket.values())
        stable = valid and all(b["sensitivity"] < LIMIT for b in bracket.values())
        row["support_status"] = "FORMAL" if stable else ("EXPLORATORY_REFERENCE_SENSITIVE" if valid else "NOT_BRACKETED")
        row["formal_claim_allowed"] = stable
        for name, b in bracket.items():
            for key in ("DOFs", "NNZ", "total_time", "bracket", "sensitivity", "weight"):
                row[f"{name}_{key}"] = b[key] if b else ""
        for metric in ("DOFs", "NNZ", "total_time"):
            for name in ("uniform", "graded"):
                row[f"{name}_over_adaptive_{metric}"] = bracket[name][metric] / bracket["adaptive"][metric] if valid else ""
        rows.append(row)
    write_csv(OUT / "adaptive_matched_targets.csv", rows)
    return rows


def localization(adaptive, fixed):
    rows = []
    for row in adaptive + fixed:
        d = np.load(OUT / "meshes" / f"{row['level']}.npz")
        centers = d["points"][d["elements"]].mean(axis=1)
        c = build_uniform_mesh(0).geom.config
        corners = np.array([[-c.w / 2, c.H], [c.w / 2, c.H], [-c.w / 2, c.H + c.t], [c.w / 2, c.H + c.t]])
        distance = np.linalg.norm(centers[:, None] - corners[None], axis=2)
        near = distance.min(axis=1) < .25 * c.H
        entry = {"method": row["method"], "level": row["level"],
                 "near_corner_radius_H": .25, "element_fraction_near_corners": float(near.mean()),
                 "elements_near_corners": int(near.sum()), "elements_elsewhere": int((~near).sum())}
        if len(d["eta_squared"]):
            eta2 = d["eta_squared"]
            marked = doerfler_mark(eta2)
            nearest = distance.argmin(axis=1)
            geom = geometry_from_mesh(MeshTri(d["points"].T, d["elements"].T), build_uniform_mesh(0).geom)
            indicator = flux_indicator(geom, d["gradients"], d["epsilon"])
            facets = geom.mesh.facets.T
            length_squared = np.sum((geom.points[facets[:, 1]] - geom.points[facets[:, 0]])**2, axis=1)
            entry.update(marked_fraction_near_corners=float(near[marked].mean()),
                         indicator_fraction_near_corners=float(eta2[near].sum() / eta2.sum()),
                         indicator_fraction_nearest_lower=float(eta2[nearest < 2].sum() / eta2.sum()),
                         marked_elsewhere=int((~near[marked]).sum()),
                         far_boundary_indicator_fraction=float(np.sum(length_squared[indicator.far] * indicator.edge_jump[indicator.far]**2) / eta2.sum()),
                         marked_max_abs_x_m=float(np.max(abs(centers[marked, 0]))),
                         marked_max_y_m=float(np.max(centers[marked, 1])))
        else:
            entry.update(marked_fraction_near_corners="", indicator_fraction_near_corners="",
                         indicator_fraction_nearest_lower="", marked_elsewhere="",
                         far_boundary_indicator_fraction="", marked_max_abs_x_m="", marked_max_y_m="")
        rows.append(entry)
    write_csv(OUT / "adaptive_localization.csv", rows)
    return rows


def matched_work(all_rows):
    u = [r for r in all_rows if r["method"] == "Uniform"]
    rows = []
    for a in [r for r in all_rows if str(r["level"]).startswith("A")]:
        m = min(u, key=lambda r: abs(np.log(r["DOFs"] / a["DOFs"])))
        ratio = max(a["DOFs"], m["DOFs"]) / min(a["DOFs"], m["DOFs"])
        if ratio <= 1.5:
            rows.append({"uniform": m["level"], "adaptive": a["level"], "DOF_ratio_max_over_min": ratio,
                         "uniform_DOFs": m["DOFs"], "adaptive_DOFs": a["DOFs"],
                         "uniform_E_rel": m["E_rel_M6"], "adaptive_E_rel": a["E_rel_M6"],
                         "uniform_CE": m["CE"], "adaptive_CE": a["CE"],
                         "uniform_total_time": m["total_time"], "adaptive_total_time": a["total_time"],
                         "formal": max(m["reference_sensitivity"], a["reference_sensitivity"]) < LIMIT})
    write_csv(OUT / "adaptive_matched_work.csv", rows)
    return rows


def save_figure(fig, stem):
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def curves(ax, all_rows, key, ykey="E_rel_M6"):
    groups = [("Uniform", "M", "#28639a", "o", "-"),
              ("Deterministic graded", "G", "#bd682b", "s", "-"),
              ("Residual adaptive (M1-S; M2 resolution)", "A", "#207e65", "^", "-"),
              ("Residual adaptive (M0-S companion)", "B", "#207e65", "v", "--")]
    for label, prefix, color, marker, style in groups:
        rows = [r for r in all_rows if str(r["level"]).startswith(prefix)]
        ax.loglog([r[key] for r in rows], [r[ykey] for r in rows], style, color=color, linewidth=1.4)
        for stable in (True, False):
            subset = [r for r in rows if r["formal_reference_stable"] == stable]
            if subset:
                ax.loglog([r[key] for r in subset], [r[ykey] for r in subset], linestyle="none", marker=marker,
                          color=color, markerfacecolor=color if stable else "white", markersize=5,
                          label=label if stable else f"{prefix}: reference-sensitive")
    ax.grid(True, which="both", alpha=.2)


def figures(all_rows, adaptive):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6), layout="constrained")
    for ax, key, label, letter in zip(axes, ("DOFs", "NNZ", "total_time"), ("Degrees of freedom", "Reduced-system NNZ", "Median cumulative / one-shot time (s)"), "abc"):
        curves(ax, all_rows, key)
        ax.set(xlabel=label, ylabel=r"Relative energy error $E_{\mathrm{rel}}$ (M6)", title=f"({letter}) Energy accuracy versus work")
    axes[0].legend(loc="lower left")
    fig.suptitle("Same 8H model, P1 and sparse direct solver; exact triangle-overlay errors", fontsize=12)
    save_figure(fig, "F_ADAPTIVE_WORK_PRECISION")

    fig, ax = plt.subplots(figsize=(6.4, 4.3), layout="constrained")
    curves(ax, all_rows, "DOFs")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (0, 2, 4, 5, 1)],
              ["Uniform", "Prescribed grading", "Adaptive primary", "Adaptive companion", "Open: reference-sensitive"], fontsize=9)
    ax.set(xlabel="Degrees of freedom", ylabel=r"Relative energy error $E_{\mathrm{rel}}$ (M6)", title="Uniform, prescribed grading and residual refinement")
    save_figure(fig, "adaptive_dofs_slide")

    main = [r for r in adaptive if str(r["level"]).startswith("A")]
    last = [r for r in main if r["formal_reference_stable"]][-1]
    chosen = [main[0], main[3], last]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), layout="constrained")
    arrays = [np.load(OUT / "meshes" / f"{r['level']}.npz") for r in chosen]
    vmax = max(d["eta_squared"].max() for d in arrays)
    norm = LogNorm(vmax * 1e-6, vmax)
    for ax, row, d, letter in zip(axes, chosen, arrays, "abc"):
        p, t = d["points"] * 1e3, d["elements"]
        visible = (p[t, 0].min(axis=1) < 2.5) & (p[t, 0].max(axis=1) > -2.5) & (p[t, 1].min(axis=1) < 2.5) & (p[t, 1].max(axis=1) > .8)
        im = ax.tripcolor(p[:, 0], p[:, 1], t[visible], facecolors=np.maximum(d["eta_squared"][visible], norm.vmin), norm=norm, cmap="viridis", edgecolors="black", linewidth=.15, rasterized=True)
        ax.plot([-1.5, 1.5, 1.5, -1.5, -1.5], [1.6, 1.6, 1.635, 1.635, 1.6], color="#b93131", linewidth=1)
        ax.set(xlim=(-2.5, 2.5), ylim=(.8, 2.5), xlabel="x (mm)", ylabel="y (mm)", title=f"({letter}) {row['level']}: {int(row['DOFs'])} DOFs", aspect="equal")
    fig.colorbar(im, ax=axes, label=r"Element indicator $\eta_K^2$ ($\mathrm{C}^2/\mathrm{m}^2$)", shrink=.75)
    save_figure(fig, "F_ADAPTIVE_MESH_EVOLUTION")

    fig, ax = plt.subplots(figsize=(6.1, 3.2), layout="constrained")
    d = arrays[-1]; p, t = d["points"] * 1e3, d["elements"]
    ax.triplot(p[:, 0], p[:, 1], t, color="#24675c", linewidth=.35)
    ax.plot([-1.5, 1.5, 1.5, -1.5, -1.5], [1.6, 1.6, 1.635, 1.635, 1.6], color="#b93131")
    ax.set(xlim=(-2.5, 2.5), ylim=(.8, 2.5), xlabel="x (mm)", ylabel="y (mm)", aspect="equal", title=f"A7: {int(last['DOFs'])} DOFs; residual-driven marking")
    save_figure(fig, "adaptive_final_mesh_slide")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    for prefix, style in (("A", "o-"), ("B", "s--")):
        seq = [r for r in adaptive if str(r["level"]).startswith(prefix)]
        dof = [r["DOFs"] for r in seq]
        error = [r["E_rel_M6"] for r in seq]
        refnorm = seq[0]["absolute_error_M6"] / seq[0]["E_rel_M6"]
        eta = [r["eta"] / np.sqrt(EPSILON_0) / refnorm for r in seq]
        axes[0].loglog(dof, error, style, label=f"{prefix}: reference error")
        axes[0].loglog(dof, eta, style, alpha=.65, label=f"{prefix}: scaled indicator")
        axes[1].semilogx(dof, [r["estimator_to_reference_error_ratio"] / np.sqrt(EPSILON_0) for r in seq], style, label=f"{prefix} path")
    axes[0].set(xlabel="Degrees of freedom", ylabel="Dimensionless diagnostic", title=r"$E_{\mathrm{rel}}$ and $\eta/(\sqrt{\epsilon_0}\|\phi_{M6}\|_E)$")
    axes[1].set(xlabel="Degrees of freedom", ylabel=r"$\eta/(\sqrt{\epsilon_0}\|e\|_E)$", title="Diagnostic ratio, not a guaranteed bound")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=.2, which="both")
    save_figure(fig, "F_ADAPTIVE_ESTIMATOR")

    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    curves(ax, all_rows, "DOFs", "capacitance_relative_to_M7")
    ax.legend(fontsize=7)
    ax.set(xlabel="Degrees of freedom", ylabel=r"$|C_h-C_{M7}|/|C_{M7}|$", title="Capacitance: selected numerical reference, not exact")
    save_figure(fig, "F_ADAPTIVE_CAPACITANCE")


def validate(adaptive, fixed):
    checks = {
        "six_or_more_main_cycles": len([r for r in adaptive if r["level"].startswith("A")]) >= 6,
        "same_geometry_material_domain_BC_P1": all(r["geometry_material_conformity_PASS"] and r["far_BC"] == "homogeneous_Neumann" and r["P1"] for r in adaptive),
        "potential_bounds": all(r["potential_min"] >= -1e-10 and r["potential_max"] <= 1 + 1e-10 for r in adaptive),
        "continuous_potential_symmetry": max(r["symmetry_error"] for r in adaptive) < 1e-10,
        "matrix_symmetry": max(r["matrix_symmetry_error"] for r in adaptive) < 1e-12,
        "normalized_residual": max(r["normalized_residual"] for r in adaptive) < 1e-10,
        "charge_balance": max(r["charge_balance_defect"] for r in adaptive) < 1e-9,
        "CE_CQ_consistency": max(r["delta_EQ"] for r in adaptive) < 1e-9,
        "positive_CE": all(r["CE"] > 0 for r in adaptive),
        "reference_sensitivity_recorded": all(np.isfinite(r["reference_sensitivity"]) for r in adaptive),
        "complete_overlay": max(r["overlay_area_defect"] for r in adaptive + fixed) < 1e-7,
        "five_timing_repetitions": all(sum(t["path"] == r["level"][0] and t["cycle"] == r["cycle"] for t in read_rows("adaptive_timing_samples.csv")) == 5 for r in adaptive),
    }
    assert_frozen(); checks["all_frozen_numerical_files_unchanged"] = True
    write_csv(OUT / "adaptive_validation.csv", [{"check": k, "status": "PASS" if v else "FAIL"} for k, v in checks.items()])
    return checks


def report_tables(adaptive, target_rows):
    directory = ROOT / "report/tables"
    main = [r for r in adaptive if str(r["level"]).startswith("A")]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Cycle & Elements & DOFs & NNZ & $E_{\rm rel}$ & $S_{\rm ref}$ (\%)\\", r"\midrule"]
    for r in main:
        lines.append(f"{r['level']} & {int(r['elements'])} & {int(r['DOFs'])} & {int(r['NNZ'])} & {r['E_rel_M6']:.4f} & {100*r['reference_sensitivity']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (directory / "adaptive_sequence.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule", r"Target & $N_U$ & $N_G$ & $N_A$ & $N_U/N_A$ & $\mathrm{NNZ}_U/\mathrm{NNZ}_A$ & $T_U/T_A$\\", r"\midrule"]
    for r in target_rows:
        label = f"{r['target']:.2f}" + (r"$^*$" if not r["formal_claim_allowed"] else "")
        lines.append(f"{label} & {r['uniform_DOFs']:.1f} & {r['graded_DOFs']:.1f} & {r['adaptive_DOFs']:.1f} & {r['uniform_over_adaptive_DOFs']:.3f} & {r['uniform_over_adaptive_NNZ']:.3f} & {r['uniform_over_adaptive_total_time']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (directory / "adaptive_targets.tex").write_text("\n".join(lines) + "\n")


def metadata():
    import hashlib
    import platform
    import scipy
    import shapely
    import skfem
    state = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
             "matplotlib": matplotlib.__version__, "scikit_fem": skfem.__version__, "shapely": shapely.__version__,
             "hypothesis_sha256": hashlib.sha256((ROOT / "logs/PRE_ADAPTIVE_HYPOTHESIS.md").read_bytes()).hexdigest(),
             "configuration": json.loads((ROOT / "config/adaptive_extension.json").read_text()),
             "primary_initial_mesh": "M1-S; M2 coordinates and 404 DOFs, mirror-completed triangulation", 
             "failed_attempt_retained": "results/adaptive_attempt_01_invalid_companion",
             "energy_integration": "Exact constant-gradient triangle-overlay areas; no coefficient-vector subtraction",
             "timing": "one warmup plus five repetitions; cumulative adaptive work versus one-shot fixed-family work; offline reference-error integration excluded for all"}
    (OUT / "experiment_metadata.json").write_text(json.dumps(state, indent=2) + "\n")


def main():
    assert_frozen()
    adaptive = read_rows("adaptive_refinement_results.csv")
    fixed = read_rows("uniform_graded_overlay_comparison.csv")
    all_rows = read_rows("three_way_comparison.csv")
    target_rows = targets(all_rows)
    local = localization(adaptive, fixed)
    pairs = matched_work(all_rows)
    checks = validate(adaptive, fixed)
    # Never update report conclusions if any critical numerical check fails.
    if not all(checks.values()):
        raise RuntimeError(f"critical validation failed: {checks}")
    figures(all_rows, adaptive)
    report_tables(adaptive, target_rows)
    metadata()
    print(json.dumps({"checks": checks, "targets": target_rows, "matched_work": pairs}, indent=2))


if __name__ == "__main__":
    main()
