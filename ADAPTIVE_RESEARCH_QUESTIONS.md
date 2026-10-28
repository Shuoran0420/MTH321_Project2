# Residual-Adaptive Research Questions

## Scope and decision

The corrected extension passes its numerical-integrity checks. This is a mixed efficiency result, not a requirement that adaptivity win. All historical uniform, graded, domain and authoritative engineering files are unchanged (71 input files checked by SHA-256). The failed first attempt is retained and excluded from formal conclusions.

Primary path: A0-A7, M1-S initial mesh with exactly the M2 coordinates and 404 DOFs, but mirror-completed triangles. Companion: B0-B7 from M0-S, used only when a target is above A0's initial error. Both use theta=0.5. The starting-mesh correction and its failure evidence are in `logs/adaptive_symmetry_preflight.md`.

## Q1. Where does the indicator refine?

A7 has 62.0% of its elements within 0.25H of a trace corner, up from 5.11% at A0. That region contains 79.6% of the squared indicator and 87.3% of the final minimal marks. Both trace ends and their upper/lower corners receive refinement. Early indicator mass is concentrated below the trace: 88.6% is assigned to elements nearest a lower corner at A0, but this becomes 50.2% at A7. The latter is not evidence of persistent lower-corner dominance.

Some marks occur outside the corner neighbourhood: final marked centroids reach |x|=5.0 mm and y=4.57 mm. The far-Neumann residual contributes only about 0.00265% of final eta-squared; it is implemented but does not dominate this sequence. Nearest-corner labels are geometric diagnostics, not a decomposition of exact singular functions.

Evidence: `results/adaptive_extension/adaptive_localization.csv`, saved element indicators and the mesh-evolution figure. No corner distances are used by marking.

## Q2. Does residual adaptivity improve energy-error-per-DOF?

At the formal target 0.40, interpolated DOFs are uniform 116.58, graded 146.19, adaptive 115.14. Uniform/adaptive ratio=1.013, a very small size advantage.

At the formal target 0.30, they are uniform 272.84, graded 348.71, adaptive 223.97. Uniform/adaptive ratio=1.218 and graded/adaptive ratio=1.557. This is about 17.9% fewer DOFs than uniform. These targets use companion B0/B1 and B2/B3, respectively, not an extrapolation of the finer primary path.

At a directly measured similar work scale, A5 uses 1206 DOFs and error 0.1553 versus M3's 1512 DOFs and error 0.1694. The primary path therefore also provides a direct, reference-stable size/accuracy comparison.

## Q3. Do NNZ and time preserve that advantage?

At target 0.30, uniform/adaptive NNZ ratio=1.235, so the sparse-system-size advantage remains. Uniform/adaptive cumulative time ratio=0.546: adaptive takes 13.1 ms versus 7.15 ms, about 1.83 times as long. Graded/adaptive time ratio=0.710, also not a runtime win for adaptive at this target.

At 0.40 the measured time ratio is 1.115, but times are only about 3 ms. This is not a defensible general speedup claim. Raw five-repeat samples are retained.

Evidence: `adaptive_matched_targets.csv` and raw timing CSVs. Ratios above one favour adaptive; no extrapolation is used.

## Q4. Why did prescribed grading not win?

The implemented grading adds full coordinate lines, not compact local patches. At a comparable scale, G2 has 2047 elements outside the 0.25H neighbourhoods and only 21.8% inside; A5 has 1322 outside and 41.8% inside. The data support identifying broader spatial spending as one limitation of this coordinate-line rule.

There is no controlled sweep of grading radii, lower-corner weighting or transition design. Therefore these data cannot isolate which parameter caused the negative result or establish that the original weighting was intrinsically wrong. The separately exact-overlay-integrated graded errors still require more DOFs than uniform at 0.40/0.30; original raw values were not replaced.

## Q5. Did residual adaptivity solve that problem?

Partly. Solution-driven marking produced more local resolution and a useful DOF/NNZ gain at 0.30 and in directly measured primary pairs. It did not turn that into a consistent total-time gain. The result is about this indicator, closure, Python implementation and measured range, not all adaptive FEM.

## Q6. Did energy improvements also help capacitance?

For A5 versus M3, CE is 123.541 versus 124.273 pF/m. Their differences from selected computed M7 are 2.42% versus 3.03%. Thus the energy improvement also improves capacitance in this matched-work example. A7 still gives 122.721 pF/m, 1.74% above M7, so it is not a replacement final engineering mesh. CE/CQ consistency remains an algebraic check, not a spatial-convergence argument.

## Q7. Is eta stably related to reference error?

No stable calibration has been demonstrated. On the primary path eta falls from 4.842e-11 to 3.622e-11 C/m by A4, then rises to 4.434e-11 C/m by A7, although true relative error decreases from 0.2637 to 0.1354. The scaled diagnostic eta/(sqrt(epsilon0)*absolute energy error) increases from 5.61 to 10.0. The requested unscaled ratio is stored as well and has units sqrt(F/m).

The indicator is not a guaranteed bound or a calibrated stopping criterion. Explaining the late increase would require a separate estimator/mesh-quality study; it is not explained away by inventing a reliability theorem.

## Q8. Does overhead offset the smaller system?

At target 0.30, yes for measured total time. The primary A7 trajectory contains approximately 0.150 s in the sum of per-cycle median solve/extraction wall times and approximately 0.107 s in prior indicator/marking/refinement work, plus initial generation. The actual repetition-aligned cumulative median is 0.270 s; sums of separate medians need not equal the median sum. Repeated previous solves, not just mesh refinement, are part of the cost. A5 takes about 2.91 times M3's one-shot total time despite its smaller system.

The comparison excludes offline reference solves and exact-error integration equally for every method, includes adaptive history, and does not compare one new solve with an old noisy single-run timing.

## Exploratory target and limitations

The 0.20 target gives an attractive adaptive size ratio but is not a formal three-way claim: graded interpolation uses G3 with 4.29% M6/M7 sensitivity, above the predeclared 3% limit. All such points remain visible with open markers.

The first attempt's invalid M0 companion and its preliminary target classifications remain in `results/adaptive_attempt_01_invalid_companion/`; they are disqualified in full by its failed bounds check. Current conclusions use only `results/adaptive_extension/` after all checks pass. No theta tuning or air-only study was performed.
