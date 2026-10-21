# Rubric Coverage Review

Historical review notice: the separate residual-adaptive extension subsequently supplies independently tested exact-overlay errors and a current three-way comparison. See `CHANGELOG_ADAPTIVE.md`, `audit/residual_adaptive_method.md` and the extension CSVs. The original defect and raw records below are retained; the legacy full-run entry point has not been recertified.

Review date: 2026-10-19 (Asia/Shanghai).

## Scope and Preserved Results

The review uses `references/Project_2_Brief.pdf`, `references/Capacitor2d.pdf`, the existing numerical CSVs, and the actual generating code. No production FEM study, final-resolution solve, M8 run, residual-based adaptive study, or speaker script was performed. Regression tests exercise the existing small verification fixtures. The authoritative computed capacitances, effective permittivity, impedance, benchmark impedances, mesh, and stored convergence sequences are unchanged.

Group C has six members. The user confirmed instructor approval on 2026-10-19 and will complete all six individual contribution statements independently. No contribution text was invented.

## Requested Corrections

| Item | Correction | Evidence |
| --- | --- | --- |
| MMS protocol | All five consecutive levels 2-6 shown, including h, h/2, h/4, h/8; both named norms, per-level residuals and pairwise orders shown; direct-solver tolerance explicitly not applicable | `tables/mms_convergence.csv`, `report/tables/mms_refinement.tex` |
| MMS forcing and fit | Explicit forcing 2*pi^2*sin(pi*x)*sin(pi*y); log-log fits on levels 4-6 | `report/tables/mms_fit_summary.csv`, `code/report_evidence.py` |
| Relative energy norm | Numerator and reference-energy denominator explicitly defined, including normalisation when the reference changes | Report equation `eq:relative-energy` |
| Reference stability | M0-M2 errors against R0/R1/R2, actual R1-to-R2 changes and fitted-slope sensitivity tabulated | `tables/reference_stability.csv`, `report/tables/reference_stability.tex`, `report/tables/reference_fit_summary.csv` |
| Grading reproducibility | Base axes, midpoint ratio, axis-band radii, coordinate snapping, diagonal rule and every G-level depth specified | `code/graded_study.py`, `report/tables/grading_rule.tex` |
| Benchmark denominator | Appendix-E d_HJ fields added; old FEM-denominator fields retained and separately labelled | `tables/engineering_benchmarks.csv`, `report/tables/benchmark_specification.tex` |
| M7 precision | Computed M7 result distinguished from a limiting value; 0.156% spatial and 0.182% domain sensitivities disclosed separately; slide 9 shows preferred 0.1% target | Report Sections 8-9; presentation slide 9 |
| Microstrip checks in PPT | Potential bounds, symmetry, free residual and CE/CQ consistency added beside MMS | Presentation slide 5; `tables/baseline_checks.csv` |
| AI case audit | Three explicit claim-risk-falsifier-evidence-decision-limitation cases added; an actual geometry-validation defect is disclosed | Report Appendix A |

The report-only MMS figure is replotted from the same five CSV error pairs with uncluttered x-axis labels. Its source is `tables/mms_convergence.csv`, its generator is `code/report_evidence.py`, and its output is `report/figures/mms_convergence_report.png`. The original required source figure remains unchanged.

The primary specification discrepancies are:

- d_HJ(FEM) = 0.008790393846587151, or 0.8790393846587151%.
- d_HJ(IPC) = 0.016081463779974263, or 1.6081463779974263%.
- d_HJ(HJ) = 0 by definition of the named reference.

The former 0.8713796% and 2.4655129% values remain valid only as separately labelled differences with FEM in the denominator. They must not be labelled d_HJ.

## Actual Defect Found: Graded Reference Geometry

The original `common_refinement_check` tested coordinate-subset identity only. A reference triangle may nevertheless cross a graded coarse-cell diagonal. On those triangles, evaluating the coarse P1 gradient at one reference centroid is not exact integration of the squared gradient difference. The coordinate test alone therefore did not justify the common-refinement claim.

The geometry-only diagnostic is reproducible using `python3 code/report_evidence.py` and saved in `tables/graded_reference_geometry_audit.csv`. It finds 4736, 9216, 21440, 42440, and 36392 crossed R2 triangles for G0-G4 respectively. Corresponding crossed area fractions are approximately 1.013%, 2.050%, 4.554%, 8.934%, and 7.613%. These fractions measure geometry, not an energy-error bound.

An independent tolerance check repeated the vertex-side classification at dimensionless cutoffs 1e-10, 1e-8, and 1e-6. Every crossing count was identical, ruling out tiny coordinate-rounding perturbations as the explanation for these counts.

The guard now checks diagonal crossings and rejects the unsupported integration shortcut before any solve. Tests cover both genuine uniform common refinement and the graded counterexample. No historical error or timing value was silently replaced. The recorded negative graded trend is retained as an observation, while precise efficiency claims require a validated common overlay or quadrature-convergence study. The 0.40/0.30 targets pass the old reference-sensitivity criterion, but that criterion alone does not resolve the new integration issue; 0.20 remains additionally reference-sensitive.

The original `formal_claim_allowed` field is preserved as the historical reference-sensitivity decision. New `integration_validation_status` and `formal_claim_allowed_current` columns in `tables/efficiency_targets.csv` explicitly record the current overlay requirement and disallow a precise formal claim for every target until that requirement is resolved. Interpolated DOFs, NNZ, times, ratios, and original reference-sensitivity values are unchanged.

The uniform hierarchy is genuinely nested and the final engineering mesh was selected from uniform M3-M7, so this finding does not change the stored C, C_air, epsilon_eff, or Z0. It does reopen the mandatory Gate-E validation requirement. The old Gate-E/I/J logs are historical evidence of the previous checks, not proof that this newly identified issue is closed.

## Current Verdict

Reporting corrections are implemented. Gate E requires integration revalidation; full-pipeline reproduction cannot currently be certified after the corrected guard. The package must remain a reviewed report candidate rather than be represented as technically complete. Six individual contribution statements also remain human work. Residual-based adaptivity is a possible later extension, not an experiment performed in this revision.

## Regression and Build Checks

- Final documented test command `pytest -q`: 33 passed in 6.77 seconds; no skips. An earlier run also passed all 33 tests in 7.20 seconds.
- An initial `python3 -m pytest -q` invocation failed during pytest startup because the project's `code` package shadows the standard-library module of that name. The documented `pytest -q` entry point succeeds. No test result was inferred from the failed invocation.
- The report compiles to 19 pages; the presentation retains exactly 12 slides.
- The final sources retain Group C, all six supplied student identities, and the MTH321 right-hand report header.
- Full production reproduction was intentionally not rerun: the corrected Gate-E preflight rejects the invalid common-refinement assumption rather than generating another unsupported comparison.
