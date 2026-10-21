# MTH321 Project 2
## 2D Finite Element Analysis of a Microstrip Transmission Line

This project uses conforming triangular finite elements to solve the two-dimensional electrostatic microstrip problem, extract capacitance per unit length, and calculate effective relative permittivity and quasi-TEM characteristic impedance.

The main objective is a reproducible numerical evidence chain, not simply a result close to 50 ohms:

**Geometry -> PDE and weak form -> Solver verification -> Physical and matrix checks -> Domain truncation -> Uniform convergence -> Graded and residual-adaptive comparisons -> Final capacitance -> Matched all-air solve -> Impedance and independent benchmarks.**

## 1. Team

Group C is an instructor-approved six-member team.

| Student | Student ID |
| --- | --- |
| Chuqiao Tang | 2360692 |
| Jinyu Ma | 2363433 |
| Muqing Xu | 2362900 |
| Shuoran Fu | 2362626 |
| Taiqi Zhang | 2363684 |
| Yujia Ding | 2362094 |

Individual contribution statements will be completed by the team before submission.

## 2. Physical Model

| Parameter | Value |
| --- | ---: |
| Substrate height, $H$ | 1.6 mm |
| Trace width, $w$ | 3.0 mm |
| Trace thickness, $t$ | 0.035 mm |
| Substrate relative permittivity | 4.4 |
| Air relative permittivity | 1 |
| Trace potential | 1 V |
| Ground potential | 0 V |

The governing equation is

$$
-\nabla\cdot(\varepsilon\nabla\phi)=0.
$$

The perfect-conductor trace interior is removed from the field domain and its surface is an internal Dirichlet boundary. The substrate-air interface is represented explicitly, with discontinuous permittivity integrated elementwise in the conforming weak formulation.

The final domain uses $W/H=H_{\mathrm{air}}/H=8$, with $x\in[-W,W]$ and $y\in[0,H_{\mathrm{air}}]$. Here $W$ is the **half-width**; the full horizontal span is $2W$. The artificial top and side boundaries use homogeneous Neumann conditions throughout the relevant studies. All calculations use SI units internally.

## 3. Numerical Method and Verification

- Conforming triangular $P_1$ finite elements with sparse stiffness assembly.
- Strong Dirichlet elimination and a sparse direct solver.
- Exact elementwise gradients and electrostatic energy integration for the production problem.
- Energy-based capacitance and reaction-based conductor charge extraction.
- Potential bounds, symmetry, reduced-matrix symmetry, conditioning, residual and charge-balance diagnostics.
- Manufactured-solution verification with approximately second-order $L^2$ error and first-order energy error.

The relative energy error is defined by

$$
E_{\mathrm{rel}}=
\frac{\|\phi_{\mathrm{ref}}-\phi_h\|_E}{\|\phi_{\mathrm{ref}}\|_E},
\qquad
\|v\|_E^2=\int_\Omega\varepsilon|\nabla v|^2\,d\Omega.
$$

Energy/charge capacitance agreement is an algebraic consistency diagnostic, not evidence of spatial convergence by itself.

## 4. Numerical Studies

### Domain truncation

The tested domain ratios are 3, 5, 8 and 12, with the near-field mesh preserved while the outer domain grows. The tested 8H result differs from the production 12H reference by approximately **0.182%**. A finer 12H repeat changes capacitance by approximately **0.0360%**. Neither 12H result is described as an exact infinite-domain answer.

### Uniform and deterministic graded refinement

The uniform energy-norm fit gives $p_E\approx0.623$, consistent with corner- and interface-limited regularity. The deterministic grading experiment compares error against DOFs, reduced-system nonzeros and measured time.

An audit found that nested coordinate lines did not guarantee nested triangles in the original graded error integration. Historical data and figures are retained with that limitation disclosed. The separate extension below recomputes comparable errors using exact triangle-overlay integration. Its formal targets still favour uniform refinement over the tested prescribed grading strategy; this is not a claim about all graded or adaptive methods.

### Residual-driven adaptive extension

The extension uses an epsilon-weighted interior-edge flux-jump indicator, homogeneous-Neumann boundary residuals and deterministic Dorfler marking with **theta = 0.5**. Dirichlet edges are excluded from the natural-boundary residual. The indicator is not claimed to be a guaranteed error bound.

The primary A0-A7 path starts from a symmetry-completed refinement of M1 with the exact M2 coordinates and **404 DOFs**, not the unchanged historical M2 triangulation. A separately identified M0-S companion path brackets the moderate error targets. M6 is the primary energy reference; the frozen M7 dielectric solution checks reference sensitivity. Different-mesh coefficient vectors are never directly subtracted.

At relative energy error **0.30**, adaptivity requires approximately **224 DOFs**, compared with **273** for uniform refinement. It also reduces matrix nonzeros, but its **cumulative algorithm time is approximately 1.83 times higher**. Adaptive timing includes the preceding solve-estimate-mark-refine cycles; DOF and NNZ comparisons are independent of this timing convention.

Formal three-way targets are **0.40 and 0.30**. Target **0.20 remains exploratory** because its graded interpolation uses a reference-sensitive point. A failed initial mesh attempt is preserved separately and excluded from the conclusions. The indicator-to-reference-error ratio has not stabilised.

See [research questions and interpretation](ADAPTIVE_RESEARCH_QUESTIONS.md), [method and timing protocol](audit/residual_adaptive_method.md), and [extension change log](CHANGELOG_ADAPTIVE.md).

## 5. Final Engineering Results

The authoritative computed dielectric mesh is `uniform_M7_8H_neumann_P1_v1`. The all-air solve reuses its exact frozen geometry, connectivity and boundary sets; only permittivity changes.

| Quantity | Reported value |
| --- | ---: |
| Dielectric capacitance, $C$ | 120.620 pF/m |
| All-air capacitance, $C_{\mathrm{air}}$ | 35.9682 pF/m |
| Effective relative permittivity, $\varepsilon_{\mathrm{eff}}$ | 3.3535 |
| Characteristic impedance, $Z_0$ | 50.642 ohms |

$$
\varepsilon_{\mathrm{eff}}=\frac{C}{C_{\mathrm{air}}},
\qquad
Z_0=\frac{1}{c_0\sqrt{C C_{\mathrm{air}}}},
\qquad c_0=299792458\ \mathrm{m/s}.
$$

M7 was accepted through an explicitly documented asymptotic/resource fallback. Its final successive capacitance change is approximately **0.156%**, above the preferred 0.1% QC target. The finite-domain sensitivity is approximately **0.182%**. These are separate sensitivity indicators, not rigorous error bounds or confidence intervals. Richardson extrapolation is diagnostic only; it does not replace computed M7 capacitance. M8 is intentionally not part of the workflow.

## 6. Independent Engineering Benchmarks

| Method | $Z_0$ (ohms) | Difference relative to FEM |
| --- | ---: | ---: |
| FEM | 50.642 | 0% |
| Hammerstad-Jensen, thickness corrected | 50.201 | 0.87% |
| IPC-2141, compact external microstrip formula | 49.393 | 2.47% |

The project specification additionally uses the Hammerstad-Jensen denominator:

$$
d_{\mathrm{HJ}}(Z)=\frac{|Z-Z_{0,\mathrm{HJ}}|}{|Z_{0,\mathrm{HJ}}|}.
$$

On that definition, the FEM and IPC discrepancies are approximately **0.879%** and **1.608%**, respectively. These are distinct from the FEM-denominator percentages in the table. Full-precision values and formula variants are recorded in [engineering_benchmarks.csv](tables/engineering_benchmarks.csv).

Engineering formulas are independent approximations, not exact reference solutions or substitutes for verification and convergence studies.

## 7. Repository Structure

```text
.
|-- code/                       # FEM solver, studies and code/tests/
|-- config/                     # Baseline and adaptive configuration
|-- data/processed/             # Exact frozen final engineering mesh
|-- tables/                     # Original CSV evidence and final engineering results
|-- results/                    # New extension data and retained failed attempt
|-- figures/                    # Original and extension figures
|-- report/                     # LaTeX source, tables, figures, bibliography and PDF
|-- presentation/               # Beamer source and presentation PDF
|-- logs/                       # Verification, study and replay records
|-- audit/                      # Method, AI audit, environment and traceability
|-- references/                 # Coursework source documents, where permitted
|-- PROJECT_STATUS.md
|-- ADAPTIVE_RESEARCH_QUESTIONS.md
|-- CHANGELOG_ADAPTIVE.md
|-- pytest.ini
|-- requirements.txt
|-- requirements-adaptive.txt
`-- README.md
```

## 8. Installation and Commands

Run from the repository root after all required folders and artifacts are present. Python 3.13 was used for the recorded validation; exact package versions are pinned in the two requirements files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-adaptive.txt
```

`requirements-adaptive.txt` includes the base requirements and the added Shapely dependency. The following tests include checks against saved evidence and therefore require the uploaded data, tables, logs and extension outputs:

```bash
pytest -q
```

Lightweight, non-destructive infrastructure check:

```bash
python3 code/run_all.py --quick
```

Standalone residual-adaptive reproduction, including M6/M7 dielectric reference solves and five-repeat timing measurements:

```bash
python3 code/residual_adaptive_study.py
python3 code/adaptive_analysis.py
```

The standalone experiment writes separate extension outputs without replacing the historical studies or final engineering values. Timings are machine-dependent. The last recorded regression passed **48 tests with no skips**; see [validation log](logs/adaptive_final_validation.txt).

### Current full-pipeline limitation

The legacy command `python3 code/run_all.py --full` retains a fail-closed guard rejecting the unsupported historical graded integration shortcut. The standalone exact-overlay extension is validated, but it has not yet been integrated into a newly certified full pipeline. **Do not treat the legacy full command as a currently certified one-command reproduction of the revised project.** Historical full-run PASS records remain explicitly historical. See [current project status](PROJECT_STATUS.md).

The frozen `data/processed/final_engineering_mesh.npz` is required for exact M7 reuse and must not be omitted. It contains 720896 elements and 362112 DOFs. The reference solves are the main resource cost; M8 is not executed.

## 9. Report and Presentation

- [Report PDF](report/final_report.pdf): 24 pages.
- [Report LaTeX source](report/main.tex).
- [Presentation PDF](presentation/final_presentation.pdf): 13 slides.
- [Beamer LaTeX source](presentation/main.tex).

The report preserves the historical integration defect and failed adaptive mesh attempt as audit evidence. The six individual contribution statements remain human-only fields to be completed before submission.

## 10. Limitations and Status

The model is electrostatic/quasistatic and quasi-TEM. It omits dielectric and conductor loss, frequency-dependent dielectric dispersion, surface roughness, solder-mask loading, anisotropy, manufacturing tolerance and full-wave radiation/dispersion.

The adaptive study tests one indicator, one primary marking parameter and one symmetry/conformity strategy. Its observed size benefit does not establish universal adaptive efficiency or runtime superiority.

The scientific results and standalone adaptive extension are supported by saved numerical evidence. Renewed integrated full-pipeline certification and team-written contribution statements remain outstanding; see [PROJECT_STATUS.md](PROJECT_STATUS.md). This README does not claim that the entire revised project has completed a new Gate-I/J certification.
