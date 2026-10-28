# Genuine intermediate failures and fixes

- Initial core regression: 48 passed. An added infrastructure test then failed (51 others passed) because it mistook the command text containing SystemExit for a traceback. Fixed assertion; no scientific result changed.
- Full attempt 20261028_132708 stopped early: active console file was mistakenly in frozen input manifest. Marked ABORTED_INFRASTRUCTURE_FIX, never PASS. Captured subsequent full consoles externally; both complete runs passed.
- First quick-replot attempt stopped when replotting preceded frozen-input validation. The guard was correct. Reordered validation before replotting derived assets; final quick passed and source files stayed unchanged.
- Duplicate LaTeX launches from the repository root reported main.tex missing; rerun from report/. Final compile succeeded with no unresolved citations/references/overfull warnings.

Temporary raw failure logs/workspaces are retained in the local recovery archive. The repository retains this concise record and full successful numerical logs, and separately retains the scientific failed adaptive mesh experiment under results/adaptive_attempt_01_invalid_companion/.
