CURRENT A–E REVISION (28 October 2026): the integrated physical + exact-overlay/adaptive workflow has now passed locally, with 208 comparisons and 58 regression tests. See `logs/ae_reproduction_report.md`, `reproduction/20261028_134044/`, and `audit/high_score_revision/ACCEPTANCE_A_E.md`. Older status paragraphs below are historical and do not override this update. Cross-platform/clean-install validation is not claimed. ICS is excluded.

CURRENT EXTENSION NOTICE (2026-10-19): residual adaptivity adds Shapely 2.1.2 (`requirements-adaptive.txt`). Actual versions and hypothesis SHA are in `results/adaptive_extension/experiment_metadata.json`. Standalone exact-overlay comparisons pass. The legacy full entry point retains its fail-closed Gate-E guard and is not recertified here; original full-run statuses below are historical. No authoritative engineering value changed.

# Environment and Reproducibility Notes

## Validated Local Environment

- Python: `3.13.13 (v3.13.13:01104ce1beb, Apr  7 2026, 14:43:30) [Clang 16.0.0 (clang-1600.0.26.6)]`
- Platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- numpy: `2.5.2`
- scipy: `1.18.1`
- matplotlib: `3.11.1`
- pandas: `3.0.5`
- PyYAML: `6.0.3`
- pytest: `9.1.1`
- scikit-fem: `12.0.2`

The locked package list is recorded in `requirements.txt`. The versions above are the versions used for the final Gate I run.

## Main Commands

```bash
python3 code/run_all.py --quick
python3 code/run_all.py --full
pytest -q
```

`--quick` is a non-destructive smoke/regression check. It reads the authoritative final artifacts, recomputes light consistency checks, and writes only `logs/run_all_quick_smoke.txt` plus `results_summary_quick.txt`.

`--full` regenerates the validated numerical evidence chain for Gates A-H. It does not generate the final report/PPT and does not run M8.

## Heavy-Run Resource Notes

The final full run on this machine took about `128.91 s`. The frozen final engineering mesh contains `720896` triangular elements and `362112` nodal DOFs. M8 is intentionally not part of the final workflow because the Gate-F preflight identified sparse-direct factorization fill-in and resource risk. Gate F therefore accepts M7 through the documented `M7_ASYMPTOTIC_RESOURCE_FALLBACK` path.

## Final Authoritative Mesh

- `FINAL_MESH_ID = uniform_M7_8H_neumann_P1_v1`
- `FINAL_MESH_SHA256 = 06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75`
- Mesh artifact: `data/processed/final_engineering_mesh.npz`
