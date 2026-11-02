# Contribution map — Shuoran Fu

Student ID: 2362626 · GitHub: [Shuoran0420](https://github.com/Shuoran0420)

This map expands the responsibilities supplied by Shuoran Fu into inspectable project files. It identifies implementation and evidence locations without assigning ownership of unrelated team work. Paths are relative to this folder and resolve after it is placed at `contributions/Shuoran_Fu/` in the repository.

| Area | Work covered | Key implementation | Check or saved evidence | Report location |
|---|---|---|---|---|
| Geometry and boundary conditions | Developing/organising the physical geometry, trace hole and boundary tags | [geometry.py](../../code/geometry.py): `GeometryConfig`, `build_geometry`, `_inside_conductor`, `_on_trace_boundary` | [Geometry tests](../../code/tests/test_geometry.py); [baseline configuration](../../config/baseline.yaml) | Physical and Mathematical Model |
| Sparse FEM solution | Developing/organising assembly, Dirichlet elimination and the potential solve | [fem_solver.py](../../code/fem_solver.py): `assemble_stiffness`, `solve_microstrip`; [capacitance.py](../../code/capacitance.py): `extract_capacitance` | [Material/baseline tests](../../code/tests/test_materials_and_baseline.py); [baseline field diagnostics](../../report/figures/baseline_field_diagnostics.png) | Finite Element Formulation; Solver Verification and Baseline Checks |
| Known-solution verification | Working on numerical verification and checking observed rates | [verification.py](../../code/verification.py): `solve_mms_level`, `run_mms`; [layered_verification.py](../../code/layered_verification.py) | [MMS tests](../../code/tests/test_verification.py); [MMS fit table](../../report/tables/mms_fit_summary.csv) | Solver Verification and Baseline Checks |
| Domain experiments | Working on domain-size comparison and interpreting finite-domain sensitivity | [domain_study.py](../../code/domain_study.py): `build_domain_geometry`, `near_field_equivalence`, `run_domain_study` | [Domain tests](../../code/tests/test_domain_study.py); [Gate C log](../../logs/gate_C_domain.txt) | Domain Truncation |
| Uniform convergence | Working on mesh refinement, reference stability and corner-effect analysis | [uniform_convergence.py](../../code/uniform_convergence.py): `build_uniform_mesh`, `energy_error`, `fit_slope` | [Uniform tests](../../code/tests/test_uniform_convergence.py); [reference stability](../../tables/reference_stability.csv) | Mesh Convergence and Corner Singularity |
| Integration issue and adaptive studies | Checking numerical issues and working on residual-based refinement and cost/error analysis | [overlay_energy.py](../../code/overlay_energy.py): `overlay_energy_error`; [residual_indicator.py](../../code/residual_indicator.py): `flux_indicator`, `doerfler_mark`; [residual_adaptive_study.py](../../code/residual_adaptive_study.py): `adaptive_path`; [adaptive_analysis.py](../../code/adaptive_analysis.py): `interpolate_target`, `validate` | [Adaptive tests](../../code/tests/test_residual_adaptive.py); [interpolated targets](../../results/adaptive_extension/adaptive_matched_targets.csv); [empirical DOF slopes](../../results/ae_revision/adaptive_empirical_rates.csv) | Historical Graded Refinement and Integration Limitation; Residual-Based Adaptive Refinement |
| Unified workflow | Organising study execution, output checks and reproducibility | [run_all.py](../../code/run_all.py): `run_quick_smoke`, `main`; [reproduction.py](../../code/reproduction.py): `run_full` | [Recent test output](logs/pytest.txt); [recent quick summary](logs/quick_summary.txt); [historical full summary](../../reproduction/minimal_20261028_152722/reproduction_summary.json) | Reproduction commands and current scope |
| Report structure and revision | Designing the overall report structure and writing/revising technical explanations, figures/tables and limitations | [main.tex](../../report/main.tex); [residual_adaptive.tex](../../report/sections/residual_adaptive.tex) | Code/result links in this map; [report figure and table sources](../../report/) | Model, FEM method, verification, experiments and conclusions |

## Upload evidence

The [core upload](https://github.com/Shuoran0420/MTH321_Project2/commit/2b9dfed0db366ce5117936fa04dd0a81fa10c9b3) and [report/figure upload](https://github.com/Shuoran0420/MTH321_Project2/commit/b5eee20c50f78abfc16908ccc8efb6fc0aa0d34b) are verified commits associated with `Shuoran0420`. They support repository delivery attribution; they do not record each earlier development or discussion step. Current files include subsequent revisions, so the present links and the original upload snapshots serve different purposes.

## Interpretation retained

- MMS rate checks and small algebraic residuals address implementation; they do not establish spatial convergence of the microstrip solution.
- The 0.182% domain comparison is a production-resolution finite-domain sensitivity, not a strict final-M7 error bound.
- Coordinate nesting did not justify the old graded cross-mesh integration. The separately tested triangle-overlay comparison is used for quantitative interpretation.
- Adaptive DOF/NNZ savings at the 0.30 target do not establish a cumulative-runtime saving or ideal asymptotic efficiency.
- Frozen engineering results, raw experiments and unsuccessful attempts remain in their existing project locations.
