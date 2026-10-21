# Pre-adaptive Hypothesis

Recorded before the residual-adaptive experiment, 2026-10-19.

Because the measured uniform energy-norm rate is below the smooth P1 rate and the field singularity is spatially concentrated near the conductor corners and conductor-material junctions, we expect residual-driven refinement to place a larger fraction of degrees of freedom in those regions. This may reduce the energy error per degree of freedom relative to uniform refinement. The deterministic graded experiment did not show such an advantage, so the adaptive experiment is intended to test whether solution-driven marking performs better than prescribed geometric grading. No efficiency improvement is assumed in advance.

## Predeclared Protocol

- Main path: exact historical M2 mesh, A0-A7, theta = 0.5.
- An M0 companion path with the same marking parameter may be used to bracket the 0.40/0.30 targets; it is not a replacement for the main path. M2 already has error below 0.30, so extrapolating those targets from M2 is prohibited.
- M6 is the principal error reference; M7 is the sensitivity reference. Formal points require absolute-energy-error sensitivity strictly below 0.03.
- All old tables, numerical figures, and authoritative final values are frozen. New uniform/graded comparison errors use exact triangle-overlay integration in separate extension tables.
- Deterministic Dörfler marking uses physical flux jumps including the Neumann residual, not manually selected corner elements. Any automatic reflection closure is separately counted and justified by the symmetric physical problem.
- One warm-up and five measured repetitions; reference-error integration is offline validation and excluded equally from algorithm timings. Cumulative adaptive work is included in the primary time comparison.
- All points, including reference-sensitive or unsuccessful points, are retained. No theta tuning is planned.
