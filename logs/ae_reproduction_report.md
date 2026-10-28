# Integrated reproduction report — 28 October 2026

## Executed commands and outcomes

| Item | Actual outcome | Evidence |
|---|---|---|
| Original regression before integration | 48 passed, 7.75 s | first task terminal record; backup retained |
| Final-code regression | 58 passed, 8.19 s, no skips | logs/ae_final_pytest.txt |
| First completed full | PASS, 190.36 s | reproduction/20261028_132749/reproduction_summary.json |
| Final numerical full | PASS, 195.88 s | reproduction/20261028_134044/reproduction_summary.json |
| Final quick | PASS, 7.75 s; no M7 solve | logs/ae_quick_final.txt |
| Numerical comparisons | 208 PASS | reproduction/20261028_134044/numerical_comparison.csv |
| Protected original result files | unchanged | same full summary |
| Mesh fingerprint | unchanged | same full summary and Gate-G identity log |

Full command: `python3 code/run_all.py --full`. Quick command: `python3 code/run_all.py --quick`. Test command: `pytest -q`.

## Stage coverage

Fresh numerical solves: smooth MMS; layered 1/10 coefficient verification; baseline; matched-core domain 3/5/8/12H plus finer 12H; uniform M0–M6 energy study; final M3–M7 capacitance and asymptotic diagnostics; matched dielectric/air M5/M6/M7; FEM impedance and independent engineering approximations; exact-overlay uniform/graded comparison; eight-cycle primary and companion adaptivity; 24 theta-sensitivity cycles.

Old graded centroid tables are historical inputs, not falsely regenerated or certified. Their rejecting safety check remains active. Fresh comparison outputs are in the separate overlay directory. Previously generated targets were removed before re-solving. Existing seed-file presence was not counted as regeneration.

Original recorded time ratios remain historical. New timing samples are retained separately; numerical equality is required for physical values within tolerance, never for timings. The 0.30 DOF/NNZ gains reproduce; cumulative adaptive time remains greater than uniform.

## Resources and unchanged final values

Apple M2, 16 GB, macOS 26.6.2; Python 3.13.13. Preflight required >=4 GiB currently available and >=2 GiB free disk, including an envelope for Python arrays and sparse-direct fill-in. Workers ran sequentially; M8 was not run.

- C = 1.206197736939107e-10 F/m
- C_air = 3.596821087666098e-11 F/m
- epsilon_eff = 3.353510523710212
- Z0 = 50.64197383662741 ohm
- FINAL_MESH_ID = uniform_M7_8H_neumann_P1_v1
- FINAL_MESH_SHA256 = 06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75

## Real failures retained

1. An integration test incorrectly searched for `SystemExit` in a log containing the command itself. Result: 1 failed / 51 passed. The assertion was corrected to inspect failure propagation and traceback; no FEM change was needed.
2. A full attempt was stopped after its growing console file was accidentally included among protected input files. It was not certified. Final runs capture the console outside protected inputs.
3. A quick reconstruction first regenerated a historical figure before validating its input fingerprint. The guard correctly stopped it. Reordered input validation before derived plotting; no guard or fingerprint was weakened.

Detailed failed workspaces are retained in the local recovery archive; concise failure reasons are public in this report. They are not erased or counted as successful numerical runs.

## Unverified or limited

- A fresh venv/package-install on this machine was not performed; the recorded existing pinned environment was checked.
- Another OS or machine has not been tested.
- No matched-M7-resolution 12H run, M8 solve or rigorous infinite-domain/spatial error bound.
- No recovery of ideal adaptive N^(-1/2) efficiency established.
- Git development history remains upload-heavy; no commits or remote push were fabricated.
- Named numbered PDFs were unavailable; inspected unnumbered course documents/current report instead.
