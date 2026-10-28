# Project Status

Updated: 2026-10-19 (Asia/Shanghai).

## Current Status

| Gate | Current status |
| --- | --- |
| A | PASS; five consecutive MMS levels and fitted named-norm orders now shown in the report |
| B / Phase 2.5 | PASS; microstrip checks also shown on slide 5 |
| C | PASS; tested finite-domain and mesh-adequacy evidence unchanged |
| D | PASS; relative energy norm and reference-stability table made explicit |
| E | Exact-overlay comparison now validated separately; historical raw curves retained |
| F | Accepted computed uniform-M7 result unchanged, with documented resource fallback |
| G | Accepted matched-air result unchanged |
| H | Computed impedances unchanged; specification HJ-denominator discrepancy added |
| Residual adaptive | PASS; corrected eight-cycle primary and eight-cycle companion paths |
| I | Original full-run PASS is historical; standalone extension reproduced, legacy full pipeline not recertified |
| J | New extension evidence map updated; integrated certification remains separate |

Overall technical certification: NOT YET COMPLETE for a newly integrated end-to-end pipeline. The requested standalone adaptive experiment is complete and validated, including exact common-overlay three-way comparisons. This does not silently renew the legacy full-run certificate. `run_all --full` retains its fail-closed old graded shortcut; integrating the extension is separate work.

## Adaptive Extension Results

- Primary start: M1-S, nonobtuse conforming completion with exactly the M2 coordinates and 404 DOFs, not historical M2 triangles.
- Theta=0.5 throughout. A0-A7: 404 to 2088 DOFs, M6 relative energy error 0.263719 to 0.135396. All primary M6/M7 sensitivities below 1.61%.
- M0-S companion brackets 0.40/0.30 without extrapolation.
- At formal 0.30: uniform/adaptive DOF ratio 1.218, NNZ ratio 1.235, cumulative-time ratio 0.546. Smaller system, slower trajectory.
- At 0.40 the size gain is small and timings are tiny. Three-way 0.20 remains exploratory due to G3.
- Eta/error ratio is not stable. No guaranteed error-bound claim.
- First diagonal-only M0 companion failed bounds; the complete attempt is retained separately.
- Corrected paths pass 13 integrity checks. All 71 frozen numerical files remain unchanged. No M8, air-only or impedance experiment was run.
- Data: `results/adaptive_extension/`; figures: `figures/adaptive_extension/`; research answers: `ADAPTIVE_RESEARCH_QUESTIONS.md`.

## Frozen Engineering Values

- Final mesh: `uniform_M7_8H_neumann_P1_v1`.
- Dielectric capacitance: `1.206197736939107e-10 F/m`.
- Air capacitance: `3.596821087666098e-11 F/m`.
- Effective permittivity: `3.353510523710212`.
- FEM impedance: `50.64197383662741 ohm`.
- Primary HJ-denominator discrepancies: FEM `0.8790393846587151%`; IPC `1.6081463779974263%`.


## Regression and Deliverables

- `pytest -q`: 48 tests PASS; no skips (see extension validation log).
- Report: 24 pages, new Section 8 and adaptive limitation/conclusion/audit; old source Figures 6/8 unchanged.
- Presentation: 13 slides, one new frame after historical slide 8; Madrid identity retained.
- The original engineering values and original CSV fields remain unchanged; newly appended discrepancy and scope fields are separately labelled.
