# Residual-Adaptive Extension Method and Reproduction

## Commands

From the repository root:

```bash
python3 -m pip install -r requirements-adaptive.txt
pytest -q
python3 code/residual_adaptive_study.py
python3 code/adaptive_analysis.py
pytest -q
```

The standalone extension intentionally does not invoke `run_all --full`, which retains its fail-closed historical graded-integration guard. It does not overwrite old studies. This extension separately validates exact-overlay three-way comparisons; integrating it into a new full-pipeline certification is outside this experiment. No M8 or all-air solve is run.

Runtime versions and the immutable hypothesis SHA are in `results/adaptive_extension/experiment_metadata.json`. Existing dependencies are unchanged; the added geometry engine is Shapely 2.1.2 / GEOS. Source configuration: `config/adaptive_extension.json`.

## Initial mesh and refinement

Primary: M1-S, an exact refinement of historical M1 with M2 node coordinates and 404 DOFs. The historical M2 diagonals are not reflected into themselves. Nonobtuse symmetry completion uses all facet midpoints and four children per parent right triangle. The M0-S companion has 114 DOFs and is used for targets above the primary initial error, not merged with the primary path into an artificially favourable envelope.

The first diagonal-only completion failed on the M0 companion. All results are retained. See `logs/adaptive_symmetry_preflight.md` for the before/after validation rationale. The hypothesis was not rewritten after results.

Dorfler selection is the minimal descending prefix with theta=0.5; element index breaks exact ties. Reflection closure and red/green conformity closure are deterministic. Symmetry-added marks and actual refined parents are separately counted. Green cells have one bisected edge; cells with two selected edges close their third and split red. Shared edge midpoints guarantee conformity. Area, boundary classification, parent nesting and no-interface-crossing are checked independently. No spatial-distance condition drives marking.

## Indicator

Physical flux is epsilon*gradient(phi), never unweighted gradient. For edge length l, constant jump J gives l*integral_edge J^2 = l^2*J^2. Each interior contribution is divided equally between its two elements. Exterior top/side contributions use the homogeneous-Neumann residual with full weight. Ground and trace edges are excluded. Volume residual is zero for this f=0, constant-coefficient P1 problem.

The global eta has units C/m. It is not the same dimensional quantity as the epsilon-weighted energy error, whose units are sqrt(J/m). Both the requested dimensional eta/error ratio and the dimensionless ratio divided additionally by sqrt(epsilon0) are explicitly distinguished. No reliability/efficiency theorem is claimed, especially for inherited anisotropic cells.

## Exact energy integration

M6 is the primary reference; exact frozen M7 is the sensitivity reference and selected numerical capacitance reference. Only the dielectric M7 problem is replayed. The semantic fingerprint and CE replay are checked in `reference_replay.csv`.

The error is sqrt(integral epsilon*|grad(ref)-grad(coarse)|^2), divided by sqrt(integral epsilon*|grad(ref)|^2). Reference and coarse coefficients are never subtracted. P1 gradients are constant on the intersection of each coarse/reference triangle pair. A compiled point-location finder provides a containment fast path; all three reference vertices must be in the candidate coarse triangle. Crossed triangles use GEOS exact polygon intersections. Coordinates are scaled by H for geometry operations, then areas are restored to m^2. Area coverage and material agreement are checked. The maximum observed relative coverage defect is below 2e-12.

Independent tests integrate two piecewise-affine functions on opposite square diagonals (known squared error one), verify symmetry under exchanging the functions, and reproduce the nested Galerkin energy identity. New uniform M0-M3 errors also reproduce the frozen uniform values. The old graded single-centroid shortcut is not used.

Reference sensitivity = abs(absolute_error_M7 - absolute_error_M6)/absolute_error_M7. Formal comparisons require strictly less than 0.03 at both ends of every interpolation bracket. Primary plot error is always relative to M6. Open markers retain sensitive uniform and graded points.

## Timing and work

One warm-up per level plus five measured repetitions. Fixed-family totals include mesh generation and the entire assembly/solve/extraction path. Adaptive `total_time` is the median of repetition-aligned cumulative trajectories: initial generation, all previous solve/indicator/mark/refine work, and the current solve/extraction. The current outgoing refinement is separately reported and is included in the next cycle's cumulative time. Each cycle also reports a full current-cycle total including outgoing refinement. Expensive reference-error evaluation and QA checks are offline, excluded equally for all methods, and error integration time is reported separately.

Raw timing samples are retained. Separate component medians do not necessarily sum to the median total. Repeated previous solves are a genuine adaptive algorithm cost. Millisecond-scale timing differences are not strong performance evidence. Timings are machine-dependent, not frozen scientific constants.

DOFs, reduced CSR NNZ, and measured cumulative/one-shot total seconds are compared. Targets use piecewise log-linear interpolation between adjacent measured error points, with no extrapolation. The companion is selected only when a target lies above the primary initial error. The 0.20 target remains exploratory due to graded G3 reference sensitivity.

## Output and acceptance

Current full-precision data: `results/adaptive_extension/`; figures: `figures/adaptive_extension/`; new report source: `report/sections/residual_adaptive.tex`. All historical numerical inputs are protected by `logs/adaptive_frozen_inputs.json` and an automated hash test.

Acceptance is numerical integrity, not an adaptive win. Corrected A0-A7/B0-B7 satisfy bounds, continuous potential symmetry, matrix symmetry, residual, charge balance, material assignment, hole integrity and conformity. The uncalibrated/nonmonotone eta and the runtime loss at target 0.30 are retained findings, not suppressed failures.
