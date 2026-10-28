# Symmetry Preflight (before formal experiment)

The unchanged M2 mesh has 404 DOFs and parallel bottom-left/top-right cell diagonals. Its nodal symmetry error is about 1e-15, but its continuous P1 potential has a 2.052% RMS reflection discrepancy on a symmetric sampling grid. Direct library local refinement produced nodal discrepancies up to about 1% in eight trial levels. Those trials are preflight diagnostics, not deleted formal experiment points.

The formal initial mesh is therefore **M2-S**, a conforming refinement of the exact M2 mesh: bisect each rectangular cell diagonal at its midpoint, joining the midpoint to the two opposite cell vertices. Every parent triangle is split into two children. No original coordinate, boundary, hole, interface, or physical parameter changes. M2-S has 756 DOFs (not 404). The M0 companion uses the identical symmetry-completion rule.

This completion is necessary because the inherited diagonal orientation is not mirror invariant. Its cost is included in initial mesh construction. It is not claimed to be the unchanged M2 mesh.

Local refinement: minimal deterministic Doerfler set, then exact reflected-triangle closure. Mark all edges of that set. For conformity and reflection invariance, repeatedly close any triangle with two marked edges by marking its third edge. Split triangles with one marked edge into two children; split triangles with three marked edges into four midpoint children. No triangle is selected by distance to corners. Counts distinguish minimal marking, reflection closure, and all actually split parents.

Eight preflight symmetry-completed cycles had continuous potential reflection discrepancies below 9e-15. Formal runs must independently repeat the checks. This red/green implementation is new extension code and must pass geometry, area conservation, no-hanging-node, and reflection tests before its results are accepted.

## Failed Attempt 01 and Corrective Protocol (before Attempt 02)

Attempt 01 is retained in `results/adaptive_attempt_01_invalid_companion/`, including all A/B points, raw timing samples, exact-overlay comparison, and the failed validation table. Its main M2-S path passed bounds, but the M0-S companion had minimum potentials from -0.0188 to -0.0298 V. It is excluded from all formal claims. No error or efficiency criterion was used to choose the correction.

Diagonal-only completion of long rectangles creates obtuse triangles. The corrected completion adds all edge midpoints and divides each original right triangle into four nonobtuse triangles (eight per rectangle). The four original rectangle corners connect to its center and adjacent edge midpoints. This remains an exact refinement of the inherited family, not independent remeshing.

For Attempt 02, the primary source is M1, with this one completion producing exactly the M2 node coordinates and 404 DOFs. Its triangulation is mirror invariant, unlike historical M2. It is called **M1-S (M2-resolution)**, never the unchanged M2. The companion is corrected M0-S (114 DOFs). Both eight-cycle preflight paths now satisfy 0 <= phi <= 1 and symmetry below 9e-15. The residual formula, theta=0.5, cycle count, solver, references, formal threshold and timing rules are unchanged. This is a validation-driven mesh correction, not tuning to obtain an efficiency win.
