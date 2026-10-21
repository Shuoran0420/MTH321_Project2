# Final Audit

## 1. Scientific Status

| Gate | Status | Evidence | Remaining issue |
|---|---|---|---|
| Gate A | PASS | `tables/mms_convergence.csv`, `logs/gate_A_verification.txt` | None |
| Gate B | PASS | `tables/baseline_checks.csv`, `logs/gate_B_checks.txt` | None |
| Phase 2.5 | PASS | `logs/phase_2_5_audit.txt`, corrected figures | None |
| Gate C | PASS | `tables/domain_truncation.csv`, `logs/gate_C_domain.txt` | None |
| Gate D | PASS | `tables/uniform_convergence.csv`, `logs/gate_D_uniform.txt` | None |
| Gate E | PASS via separate exact-overlay revalidation | `results/adaptive_extension/uniform_graded_overlay_comparison.csv`, exact-overlay tests | Historical raw curves are not replaced; legacy full-run integration remains separate |
| Gate F | PASS | `tables/final_dielectric_capacitance.csv`, `data/processed/final_engineering_mesh.npz`, `logs/gate_F_capacitance.txt` | M7 fallback caveat remains documented |
| Gate G | PASS | `tables/final_air_capacitance.csv`, `tables/final_effective_permittivity.csv`, `logs/gate_G_air.txt` | Inherits Gate-F spatial caveat |
| Gate H | PASS | `tables/final_characteristic_impedance.csv`, `tables/engineering_benchmarks.csv`, `logs/gate_H_impedance.txt` | Benchmarks are approximate, not exact |

## 2. Reproducibility

Gate I status: historical PASS before the 2026-10-19 geometry audit. The standalone residual-adaptive extension is reproduced and validated, but the legacy full-run entry point has not been integrated and recertified with it. Evidence of the original run remains historical in `logs/gate_I_reproducibility.txt`.

- Historical `python3 code/run_all.py --quick`: PASS
- Historical `python3 code/run_all.py --full`: PASS before the corrected Gate-E guard
- Current `pytest -q`: PASS, 48 tests; standalone overlay/adaptive acceptance passes, but tests alone do not recertify the legacy full pipeline
- Frozen mesh SHA-256: `06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75`

## 3. Traceability

Gate J status: extension evidence is mapped in `report/report_evidence_map.md` and `audit/report_evidence.md`. The original master traceability files record the pre-extension review; renewed integrated final certification remains separate.

## 4. Final Authoritative Results

- `FINAL_MESH_ID = uniform_M7_8H_neumann_P1_v1`
- `C = 1.2061977369391071e-10 F/m = 120.620 pF/m`
- `C_air = 3.5968210876660980e-11 F/m = 35.9682 pF/m`
- `epsilon_eff = 3.3535105237102121e+00`
- `Z0_FEM = 5.0641973836627407e+01 ohm`
- Hammerstad-Jensen: `50.200690 ohm`
- IPC-2141: `49.393389 ohm`

## 5. Report Status

Report build status: PASS, 24 pages. New Section 8 discloses the exact-overlay extension and mixed efficiency result; computed engineering quantities and historical numerical figures remain unchanged. The six contribution-statement placeholders remain.

## 6. PPT Status

PPT status: PASS, 13 slides. One requested residual-adaptive slide follows historical slide 8; the twelve previous scientific frames and Madrid identity are retained. Group C and all six student names/IDs are filled.

## 7. Automated Test Status

`pytest -q`: PASS, 48 tests, no skips. See the extension validation log. Use the documented `pytest -q` command; the unrelated `python3 -m pytest` package-shadowing startup limitation is recorded in the historical rubric review.

## 8. Remaining Human-Only Placeholders

See `docs/HUMAN_COMPLETION_REQUIRED.md`.

Group C, the six-member team size, and all student names/IDs are confirmed. The only remaining human-only fields are the six individual contribution statements. These require each member's input and were not invented.

## 9. Known Limitations

- Quasistatic/quasi-TEM model only.
- No dielectric loss, conductor loss, surface roughness, solder mask, manufacturing tolerance, anisotropy, full-wave dispersion or radiation model.
- Finite-domain indicator remains separate from spatial-discretisation indicator.
- Gate F uses an M7 asymptotic/resource fallback because M8 was not recommended by preflight.
- Engineering formulas are approximate independent benchmarks, not exact reference solutions.

## 10. Final Verdict

Requested standalone adaptive extension: PASS. Technical project-wide recertification: NOT YET COMPLETE until the separately validated comparison is integrated into the legacy full reproduction workflow.

Submission status: REPORT CANDIDATE; full-pipeline recertification and six individual contribution statements remain. No new final engineering result is substituted.
