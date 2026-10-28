# MTH321 Project 2 — Microstrip FEM

Group C studies a two-dimensional microstrip cross-section using conforming triangular P1 finite elements. The PDE is $-\nabla\cdot(\varepsilon\nabla\phi)=0$: trace 1 V, ground 0 V, homogeneous-Neumann top/side boundaries. The substrate has relative permittivity 4.4; air has 1. Geometry: H=1.6 mm, w=3.0 mm, t=0.035 mm; final W/H=Hair/H=8 (W is half-width).

## Results

| Quantity | Reported value |
|---|---:|
| Dielectric capacitance | 120.620 pF/m |
| All-air capacitance | 35.9682 pF/m |
| Effective permittivity | 3.3535 |
| Characteristic impedance | 50.642 Ω |

The final mesh is `uniform_M7_8H_neumann_P1_v1`. See [report](report/final_report.pdf) and [slides](presentation/final_presentation.pdf). Their editable sources are beside the PDFs. Three individual contribution statements remain for the team to complete.

## Install and run

Verified local environment: Python 3.13.13, macOS on Apple M2 with 16 GB RAM. Exact package versions are pinned in the two requirements files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-adaptive.txt
pytest -q
python code/run_all.py --quick
python code/run_all.py --full
```

Use `pytest -q`; `python -m pytest` at the project root can encounter a conflict between the project package `code` and the Python standard-library module with that name.

**Quick** validates saved evidence, runs a small infrastructure smoke test and regenerates supported figures/tables in `reproduction_runs/quick_*/saved_analysis/`. It does not recompute final M7 results or overwrite authoritative root outputs. Running `python code/run_all.py` defaults to quick mode. Report-specific cosmetic figure regeneration is not exhaustively certified by that default command.

**Full** freshly runs smooth/layered verification, baseline/domain/uniform studies, capacitance convergence, matched all-air pairs, impedance benchmarks and exact-overlay/adaptive comparisons. It backs up protected inputs, checks resources, and writes isolated results in `reproduction_runs/`. It compares 208 scientific values and validates the frozen mesh fingerprint. The old graded integration safety guard remains enabled. Full execution requires at least 4 GiB currently available memory and 2 GiB free disk; it took about three minutes on the recorded machine. Sparse-direct fill-in can require much more memory than matrix storage. M8 is intentionally excluded. Another OS or machine has not been verified.

## Files

- `code/`: solver, study scripts, plotting and automated tests.
- `config/`: physical and adaptive settings.
- `data/`: frozen M7 mesh and baseline field snapshot.
- `tables/`: authoritative numerical outputs.
- `results/`: adaptive/overlay data and necessary field snapshots; failed-attempt CSVs are retained.
- `figures/`: saved scientific figures required for evidence checks and LaTeX.
- `report/`, `presentation/`: editable LaTeX, required assets and checked PDFs.
- `logs/`, `reproduction/`, `audit/`: selected genuine verification records and method explanations.

The frozen M7 NPZ is approximately 47 MiB and must be retained for exact matched-mesh reuse. No external large-data download is needed. Full-run backups, environments, caches, duplicate archives, course handouts and speaker scripts are omitted. Numerical source data and core code were not changed when preparing this minimal upload.

## Scientific limitations

M7 was accepted by the documented resource/asymptotic fallback: its final capacitance change is about 0.156%, above the preferred 0.1% QC target. The production-domain sensitivity is about 0.182%, not a rigorous final-M7 truncation bound. Richardson is diagnostic. Adaptive refinement reduced interpolated DOFs/NNZ at the 0.30 target but increased cumulative time; ideal DOF convergence was not established. Historical graded centroid-based comparisons remain exploratory; the extension uses exact triangle overlays. The quasi-static/quasi-TEM model omits losses, radiation, roughness and fabrication tolerances.

To compile the report or slides, run `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` from the corresponding directory, with a TeX distribution installed. No Git history is fabricated or included here; upload these files into the existing repository to preserve its history.

## Minimal-package validation — 28 October 2026

This trimmed copy was tested after file selection: **58 tests passed**, quick mode passed, and full numerical reproduction passed **208 comparisons** in 185.75 seconds. The frozen mesh fingerprint matched and protected files remained unchanged. Fresh-run evidence is in `reproduction/minimal_20261028_152722/`; pytest and quick logs are in `logs/github_minimal_*.txt`.
