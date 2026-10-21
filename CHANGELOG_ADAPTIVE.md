# Residual-Adaptive Extension Changelog

## Scope and findings

An independent dielectric-only residual-adaptive experiment was added with theta=0.5, eight primary and eight companion cycles. The corrected initial primary mesh is M1-S, a conforming mirror completion with the exact M2 coordinates and 404 DOFs. Historical raw data, source Figures 6/8, final M7 mesh and all final engineering quantities remain unchanged; 71 numerical inputs are protected by SHA-256 regression.

The first diagonal-only completion failed M0-companion potential bounds. All its points and raw timings remain under `results/adaptive_attempt_01_invalid_companion/`, with an explicit disqualification notice. No theta tuning or point deletion was used to obtain a win.

At formal target 0.30, uniform/adaptive DOF and NNZ ratios are 1.218 and 1.235, but total-time ratio is 0.546 (adaptive slower). Target 0.40 has only a small DOF advantage; tiny timing differences are not strong speedup evidence. Target 0.20 remains exploratory due to graded reference sensitivity. Eta is not monotone throughout and its ratio to true reference error has not stabilised.

## New source / configuration

- `code/residual_indicator.py`: weighted flux jumps, exact edge integrals, minimal deterministic Dorfler marking.
- `code/adaptive_mesh.py`: symmetry completion, reflection/conformity closure, continuous symmetry metric.
- `code/overlay_energy.py`: exact triangle-overlay gradient-error integration with material/area checks.
- `code/residual_adaptive_study.py`: matched references, five-repeat timings, primary/companion and fixed-family comparisons.
- `code/adaptive_analysis.py`: targets, matched work, localisation, validation, tables and figures.
- `code/tests/test_residual_adaptive.py`: 15 new tests; full suite now 48 tests.
- `config/adaptive_extension.json`, `requirements-adaptive.txt` (Shapely 2.1.2).

## New evidence

- `results/adaptive_extension/`: full cycle table, raw timing samples, exact-overlay three-way comparison, matched targets/work, localisation, validation, reference replay, metadata and saved mesh fields.
- `logs/PRE_ADAPTIVE_HYPOTHESIS.md`, symmetry preflight, frozen-input manifest, execution/analysis/validation logs.
- `ADAPTIVE_RESEARCH_QUESTIONS.md`, `audit/residual_adaptive_method.md`.
- Three required figure families plus capacitance diagnostics and two presentation-only assets under `figures/adaptive_extension/`, each PDF/PNG.

## Report / PPT

- New `report/sections/residual_adaptive.tex` after Section 7, two generated adaptive tables, unchanged historical Figures 6/8.
- `report/main.tex`: transition, new section, adaptive limitation/conclusion, actual AI-audit cases; later sections renumber.
- `report/report_evidence_map.md`, compiled `report/final_report.pdf`.
- `presentation/main.tex`: one new frame after historical slide 8; PDF now 13 slides. Existing twelve scientific frames retained.

## Project records

README, status, requirements, experiment, numerical-method, AI-audit and report-evidence records identify the new standalone extension and its scope. Current report/PPT hashes are refreshed. No M8, all-air or benchmark experiment was run.

The separate extension is reproducible with its documented commands. The legacy `run_all --full` retains the old fail-closed graded shortcut and has not been recertified with the new comparison. This experiment does not silently assert a new Gate-I/J completion. Six approved Group C identities remain; ICS content is left to the team.
