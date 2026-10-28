"""Regenerate report tables from frozen evidence, without running a FEM solve."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from impedance import add_hj_discrepancies


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fitted_order(hs: list[float], errors: list[float]) -> float:
    x = [math.log(h) for h in hs]
    y = [math.log(e) for e in errors]
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def tabular(path: Path, columns: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    lines += [" & ".join(row) + r"\\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def generate(root: Path) -> dict:
    output = root / "report/tables"
    output.mkdir(exist_ok=True)
    mms = read_rows(root / "tables/mms_convergence.csv")
    assert len(mms) >= 4
    assert all(int(b["level"]) == int(a["level"]) + 1 for a, b in zip(mms, mms[1:]))
    assert all(math.isclose(float(a["h_max"]), 2 * float(b["h_max"])) for a, b in zip(mms, mms[1:]))
    mms_rows = []
    for i, row in enumerate(mms):
        h_label = "$h$" if i == 0 else f"$h/{2**i}$"
        mms_rows.append([
            row["level"] + " (" + h_label + ")", row["elements"], row["dofs"], f"{float(row['h_max']):.5f}",
            sci(float(row["l2_error"])), sci(float(row["energy_error"])), sci(float(row["residual_inf"])),
            "--" if not row["l2_order"] else f"{float(row['l2_order']):.3f}",
            "--" if not row["energy_order"] else f"{float(row['energy_order']):.3f}",
        ])
    tabular(output / "mms_refinement.tex", "lrrrrrrrr",
            ["Level", "$N_T$", "DOFs", "$h_{\\max}$", "$\\|e\\|_{L^2}$", "$\\|e\\|_E$", "$\\|r_F\\|_\\infty$", "$p_{L^2}$", "$p_E$"], mms_rows)
    fit = mms[-3:]
    fits = [
        {"norm": norm, "fit_levels": ",".join(r["level"] for r in fit),
         "fitted_order": fitted_order([float(r["h_max"]) for r in fit], [float(r[field]) for r in fit])}
        for norm, field in [("L2", "l2_error"), ("energy", "energy_error")]
    ]
    write_rows(output / "mms_fit_summary.csv", fits)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    hs = [float(r["h_max"]) for r in mms]
    l2 = [float(r["l2_error"]) for r in mms]
    energy = [float(r["energy_error"]) for r in mms]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.loglog(hs, l2, "o-", label=r"$L^2$ error")
    ax.loglog(hs, energy, "s-", label="Energy error")
    ax.loglog(hs, [l2[0] * (h / hs[0]) ** 2 for h in hs], "--", color="C0", alpha=0.7, label="Slope 2 reference")
    ax.loglog(hs, [energy[0] * h / hs[0] for h in hs], "--", color="C1", alpha=0.7, label="Slope 1 reference")
    ax.set_xticks(hs, [f"{h:.3f}" for h in hs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Maximum element diameter $h_{\max}$ (unit square)")
    ax.set_ylabel("Error norm")
    ax.set_title("Manufactured-solution convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "report/figures/mms_convergence_report.png", dpi=200)
    plt.close(fig)

    refs = [r for r in read_rows(root / "tables/reference_stability.csv") if r["analysis_mesh_id"] in ("M0", "M1", "M2")]
    tabular(output / "reference_stability.tex", "lrrrr",
            ["Mesh", "$E_{\\rm rel}^{R0}$", "$E_{\\rm rel}^{R1}$", "$E_{\\rm rel}^{R2}$", "$R1\\to R2$ change"],
            [[r["analysis_mesh_id"], *[f"{float(r[k]):.6f}" for k in ("error_using_R0", "error_using_R1", "error_using_R2")],
              f"${100*float(r['sensitivity_R1_to_R2']):.3f}\\%$"] for r in refs])
    ref_fits = {ref: fitted_order([float(r["h_max_m"]) for r in refs], [float(r["error_using_" + ref]) for r in refs]) for ref in ("R0", "R1", "R2")}
    write_rows(output / "reference_fit_summary.csv", [{"reference": ref, "fit_meshes": "M0,M1,M2", "fitted_order": value} for ref, value in ref_fits.items()])

    graded = read_rows(root / "tables/graded_convergence.csv")
    tabular(output / "grading_rule.tex", "lrrrrr",
            ["Mesh", "$b$", "$u$", "$\\ell$", "DOFs", "Min. corner size (m)"],
            [[r["graded_level"], r["background_refinement_level"], r["upper_corner_refinement_level"], r["lower_corner_refinement_level"], r["DOFs"], sci(float(r["minimum_corner_element_size"]))] for r in graded])

    benchmarks = add_hj_discrepancies(read_rows(root / "tables/engineering_benchmarks.csv"))
    write_rows(root / "tables/engineering_benchmarks.csv", benchmarks)
    if (root / "results/tables").exists():
        write_rows(root / "results/tables/engineering_benchmarks.csv", benchmarks)
    names = {"FEM": "FEM matched pair", "Hammerstad_Jensen": "Hammerstad--Jensen", "IPC_2141": "IPC-2141"}
    tabular(output / "benchmark_specification.tex", "lrrrr",
            ["Method", "$\\varepsilon_{\\rm eff}$", "$Z_0$ ($\\Omega$)", "$100d_{\\rm HJ}$", "Diff. vs FEM"],
            [[names[r["method"]], "--" if not math.isfinite(float(r["epsilon_eff"])) else f"{float(r['epsilon_eff']):.4f}",
              f"{float(r['Z0_ohm']):.3f}", f"${float(r['percent_discrepancy_Z0_vs_HJ']):.3f}\\%$",
              f"${float(r['percent_difference_Z0_vs_FEM']):.3f}\\%$"] for r in benchmarks])
    macros = {
        "MMSLTwoFit": f"{fits[0]['fitted_order']:.4f}", "MMSEnergyFit": f"{fits[1]['fitted_order']:.4f}",
        "RefSlopeROne": f"{ref_fits['R1']:.4f}", "RefSlopeRTwo": f"{ref_fits['R2']:.4f}",
        "HJDiscrepancyFEM": f"{benchmarks[0]['percent_discrepancy_Z0_vs_HJ']:.2f}",
        "HJDiscrepancyIPC": f"{benchmarks[2]['percent_discrepancy_Z0_vs_HJ']:.2f}",
    }
    (output / "evidence_numbers.tex").write_text("".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items()))

    from graded_study import GRADED_SPECS, build_graded_mesh, common_refinement_diagnostics
    from uniform_convergence import build_uniform_mesh

    reference = build_uniform_mesh(6)
    geometry = [{"graded_level": f"G{s.level}", "reference": "R2=M6", "reference_elements": reference.geom.elements.shape[0],
                 **common_refinement_diagnostics(build_graded_mesh(s), reference)} for s in GRADED_SPECS]
    write_rows(root / "tables/graded_reference_geometry_audit.csv", geometry)
    targets = [
        {**r, "integration_validation_status": "PENDING_COMMON_OVERLAY", "formal_claim_allowed_current": False}
        for r in read_rows(root / "tables/efficiency_targets.csv")
    ]
    write_rows(root / "tables/efficiency_targets.csv", targets)
    if (root / "results/tables").exists():
        write_rows(root / "results/tables/efficiency_targets.csv", targets)
    return {"mms_fits": fits, "reference_fits": ref_fits, "benchmarks": benchmarks, "graded_geometry": geometry}


if __name__ == "__main__":
    result = generate(Path(__file__).resolve().parents[1])
    for row in result["mms_fits"]:
        print(row)
    print("Reference fits:", result["reference_fits"])
    print("Report tables regenerated; no FEM solve was run.")
