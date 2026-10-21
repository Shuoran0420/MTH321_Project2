CURRENT EXTENSION NOTICE (2026-10-19): new residual-adaptive and exact-overlay methods are specified in `audit/residual_adaptive_method.md` and report Section 8. Physical settings and solver are unchanged. The failed attempt is retained. Legacy full-run certification is not renewed by the standalone extension.

# Numerical Method

## Physical Geometry

The baseline model is a two-dimensional microstrip cross-section. All computations use SI units.

- Substrate height: `H = 1.6e-3 m`
- Trace width: `w = 3.0e-3 m`
- Trace thickness: `t = 0.035e-3 m`
- Baseline domain: `x in [-W, W]`, `y in [0, Hair]`, with `W/H = 8` and `Hair/H = 8`
- Ground plane: `y = 0`
- Trace: centered at `x = 0`, occupying `-w/2 <= x <= w/2`, `H <= y <= H+t`

The trace interior is removed from the finite-element field domain. It is not modelled as a dielectric subdomain. The trace boundary is an internal Dirichlet boundary.

## PDE

The electrostatic potential satisfies `-div(epsilon grad(phi)) = 0` in the air-substrate field domain.

## Piecewise Permittivity

The permittivity is not globally averaged. Each triangular element is tagged from its centroid:

- substrate: `epsilon = epsilon_0 * 4.4`
- air: `epsilon = epsilon_0 * 1.0`

The substrate-air interface is represented by mesh lines at `y = H`, so the material jump is integrated elementwise on conforming subregions.

## Interface Condition

The conforming `H1` finite-element space enforces continuity of `phi`. The weak form with elementwise permittivity enforces normal-flux continuity in the weak sense across the air-substrate interface:

```text
[phi] = 0,
[epsilon grad(phi) dot n] = 0.
```

This does not imply pointwise equality of two separately differentiated low-order flux traces.

## Boundary Conditions

Essential Dirichlet conditions:

- `phi = V0 = 1 V` on the whole trace boundary
- `phi = 0 V` on the ground boundary

Artificial far-boundary convention:

- homogeneous Neumann, `epsilon grad(phi) dot n = 0`, on the top and side boundaries

The far boundary is therefore not included in the Dirichlet set.

## Weak Form

Let `Gamma_D = Gamma_trace union Gamma_ground`. Find `phi` in the affine trial space satisfying the prescribed Dirichlet data such that for every homogeneous test function `v`,

```text
integral_Omega epsilon grad(phi) dot grad(v) dOmega = 0.
```

There is no volume load in the microstrip baseline solve.

## Finite Elements and Mesh Treatment

The baseline implementation uses conforming triangular P1 finite elements. The mesh is generated as a structured, symmetric triangular mesh whose coordinate lines include:

- `y = H`
- `y = H+t`
- `x = -w/2`
- `x = 0`
- `x = w/2`

Cells whose centroids lie strictly inside the conductor rectangle are removed. This makes the conductor appear as an internal rectangular hole with boundary nodes available for the trace Dirichlet condition. The mesh is symmetric about `x = 0`.

## Quadrature and Assembly

For P1 triangles, `grad(phi_h)` is constant on each element. The stiffness matrix is assembled element-by-element using

```text
K_ij^T = epsilon_T * area_T * grad(N_i) dot grad(N_j).
```

This is exact for the P1 stiffness integrand with constant elementwise permittivity.

## Dirichlet Elimination

The full stiffness matrix is assembled before boundary conditions. The trace and ground values are inserted into the full coefficient vector. The reduced free-DOF system is

```text
K_ff u_f = -K_fd u_d.
```

Matrix symmetry and positive definiteness are checked on the correctly reduced free-DOF matrix `K_ff`, not on a row-replaced matrix.

## Linear Solver

The baseline uses SciPy's sparse direct solver `spsolve` on the reduced CSR matrix. The reported residual is the free-DOF residual `r = K_ff u_f - rhs_f`.

## Matrix Conditioning Diagnostics

The reduced free-DOF matrix includes physical permittivity in SI units, including the factor `epsilon_0`, so its absolute eigenvalues are small. For positive definiteness the sign of the smallest eigenvalue is the relevant check, while the ratio `lambda_max/lambda_min` is more informative for conditioning.

The baseline diagnostics therefore report:

- `lambda_min` and `lambda_max` for the physical reduced matrix
- `condition_estimate = lambda_max / lambda_min`
- `scaled_condition_estimate` for the diagonally scaled matrix `D^{-1/2} A D^{-1/2}`

The scaled estimate is a diagonal-scaled diagnostic and should not be confused with the physical matrix condition number.

## Energy Capacitance

The electrostatic energy per unit length is computed from element gradients:

```text
We = 0.5 * integral_Omega epsilon |grad(phi_h)|^2 dOmega.
CE = 2 * We / V0^2.
```

Because all lengths are in metres, `CE` has units `F/m`.

## Reaction Capacitance

The full pre-elimination reaction is computed as `r_full = K_full u_full`. The trace charge magnitude is extracted by summing reactions over trace Dirichlet DOFs:

```text
Q_trace = abs(sum(r_full[trace_dofs]))
CQ = Q_trace / abs(V0).
```

The ground reaction is summed separately over ground Dirichlet DOFs. Trace and ground DOF sets are checked to be disjoint.

The diagnostic `delta_EQ = abs(CE - CQ) / CE` is an implementation consistency check. It is not a proof of spatial convergence.


## Gate C Domain-Truncation Mesh Control

The domain-truncation study uses a separate controlled mesh generator so that changing `W/H` and `Hair/H` extends only the outer mesh. The near-field core is fixed as:

- `x in [-3.2e-3, 3.2e-3] m`
- `y in [0, 2.4e-3] m`

The preserved coordinate lines include:

- `x = -w/2`
- `x = 0`
- `x = w/2`
- `y = H`
- `y = H+t`

The production core resolution is `CORE_NX=64`, `CORE_NY_R0=72`. The refined 12H check uses `CORE_NY_R1=108`. An automated equivalence check compares core node coordinates, core element centroids, trace DOF coordinates, trace geometry, and interface location across the 3H, 5H, 8H, and 12H production domains.

The capacitance differences reported in Gate C are relative to the tested 12H reference. They are not claimed to be exact infinite-domain errors.


## Gate D Uniform Mesh Convergence

Gate D uses the fixed `W/H = 8`, `Hair/H = 8` domain with homogeneous Neumann far boundaries. This is a project decision based on Gate C: the 8H domain differs from the tested 12H reference by about 0.182%, while being cheaper than 12H.

The uniform refinement family is separate from the later adaptive/graded study. A geometry-conforming base grid includes the trace edges and the material interface, and every base interval is bisected globally at each refinement level. No local corner refinement is used in Gate D.

Cross-mesh energy errors are computed on the nested reference mesh. Coarse P1 gradients are evaluated on each reference element from the containing coarse element. Coefficient vectors from different meshes are never directly subtracted. The energy norm uses the same discontinuous physical permittivity as the FEM solve.

The final accepted reference is `R2=M6`. The fitted energy-norm slope is `0.622987` over fit levels `[0, 1, 2]`. On the same fit window, the slope changes from `0.638307` with `R1=M5` to `0.622987` with `R2=M6`, a relative change of about `2.459%`. The fit-window R1-to-R2 error sensitivities are below the 3% target. The direct `R2-R1` reference self-difference remains about `5.881%`, so the report should state this distinction and base Gate D acceptance on the stability of the quantities actually reported.

## Gate E Graded / Local Refinement

Gate E keeps the same `W/H = 8`, `Hair/H = 8` physical problem, material data, boundary conditions, P1 FEM, quadrature convention, sparse direct solver, and capacitance extraction used in Gate D. Only the mesh-coordinate selection is changed.

The graded meshes are deterministic dyadic coordinate subsets of the Gate-D refinement tree. The trace corners are:

- lower junction corners: `(-w/2, H)` and `(w/2, H)`;
- upper conductor corners: `(-w/2, H+t)` and `(w/2, H+t)`.

The grading rule selects finer dyadic coordinate lines within fixed physical neighbourhoods of these corners. The lower junction corners receive one extra early refinement level because Gate D showed strong field concentration at the conductor/material junction. The rule is symmetric about `x=0`, and no element is selected manually by index.

The sequence is:

- `G0`: background `0`, upper corner `1`, lower corner `2`;
- `G1`: background `1`, upper corner `2`, lower corner `3`;
- `G2`: background `1`, upper corner `3`, lower corner `4`;
- `G3`: background `2`, upper corner `4`, lower corner `5`;
- `G4`: background `3`, upper corner `5`, lower corner `5`.

All formal graded meshes have maximum local depth at most `M5`, so every graded coordinate is represented on the accepted `R2=M6` uniform reference. The energy error is integrated on R2 by evaluating the graded P1 gradient on each R2 element from the containing graded cell. Coefficient vectors from different meshes are never directly subtracted.

Timings are measured with three repeated solves for graded meshes, and the median assembly, solve, and total times are reported. Uniform comparison rows use the Gate-D hierarchy with R2-referenced errors. Target-accuracy comparisons use log-linear interpolation only inside the measured error range of both methods.

The tested coordinate-line graded strategy successfully concentrates resolution near corners, but it does not outperform uniform refinement in the target-error interpolation table. This limitation is part of the Gate E evidence and should not be hidden in the report.

## Gate F Final Dielectric Capacitance

Gate F uses the fixed 8H homogeneous-Neumann domain and the nested uniform hierarchy M3-M7. M8 is not run because preflight estimates about 1.45 million DOFs and substantial sparse-direct factorisation fill-in. M7 is accepted through the `M7_ASYMPTOTIC_RESOURCE_FALLBACK` path: CE decreases monotonically, apparent capacitance orders remain stable, and the last Richardson diagnostic changes by about `1.61e-05` relative. The authoritative capacitance is the computed M7 energy capacitance, not the Richardson extrapolate. The final mesh is frozen as `uniform_M7_8H_neumann_P1_v1` for exact reuse by Gate G.

## Gate G Matched Air-Only Solve

Gate G loads the frozen mesh artifact `data/processed/final_engineering_mesh.npz` rather than regenerating a nominally similar M7 mesh. The artifact stores node coordinates, element connectivity, material-region labels, trace/ground/far boundary node sets, the conductor polygon, and geometry metadata. A deterministic SHA-256 fingerprint is computed from those arrays and metadata and is reported in the dielectric replay, all-air solve, and epsilon-effective table.

The dielectric replay uses the same discontinuous material coefficient as Gate F: `epsilon_r=4.4` in the substrate and `epsilon_r=1.0` in air. The replayed capacitance must match the authoritative Gate-F value before any all-air interpretation is made.

The all-air auxiliary problem changes only the material coefficient:

```text
epsilon = epsilon_0 everywhere.
```

The geometry, mesh, element order, quadrature convention, Dirichlet sets, homogeneous-Neumann artificial boundary convention, sparse direct solve, and capacitance extraction are unchanged. The geometric line at `y=H` remains in the mesh because it is part of the frozen geometry, but in the all-air solve it is only a former substrate-interface mesh line and has no material jump.

The effective relative permittivity is then computed from the matched capacitance ratio:

```text
epsilon_eff = C_dielectric / C_air.
```

Gate G does not compute the characteristic impedance `Z0`; that is reserved for Gate H.

## Gate H Characteristic Impedance From the Capacitance Pair

Gate H uses the exact SI speed of light

```text
c0 = 299792458.0 m/s.
```

The authoritative FEM characteristic impedance is computed directly from the matched Gate F/G capacitance pair:

```text
Z0 = 1 / (c0 * sqrt(C * Cair)).
```

The internally equivalent quasi-TEM quantities are:

```text
epsilon_eff = C / Cair,
Z0_air = 1 / (c0 * Cair),
Z0 = Z0_air / sqrt(epsilon_eff),
L = 1 / (c0^2 * Cair),
Z0 = sqrt(L / C),
vp = c0 / sqrt(epsilon_eff) = 1 / sqrt(L * C).
```

These are algebraic consistency checks within the same quasi-TEM model. They are not independent validation.

## Gate H Engineering Benchmark Equations

The engineering benchmark formulas use only the physical geometry and substrate relative permittivity:

```text
epsilon_r = 4.4,
w = 3.0e-3 m,
H = 1.6e-3 m,
t = 0.035e-3 m.
```

They do not use `C`, `Cair`, FEM `epsilon_eff`, or FEM `Z0` as inputs.

### IPC-2141 Compact External Microstrip Formula

The selected IPC-2141 expression is the compact external-microstrip approximation from `Capacitor2d.pdf` Appendix C:

```text
Z0_IPC = 87 / sqrt(epsilon_r + 1.41) * ln(5.98 H / (0.8 w + t)).
```

All geometric dimensions must be in a common unit. The logarithm is the natural logarithm. This is a coarse empirical engineering cross-check.

### Hammerstad-Jensen Finite-Thickness Formula

The selected Hammerstad-Jensen variant is the finite strip-thickness correction from `Capacitor2d.pdf` Appendix D. Define

```text
u = w/H,
tau = t/H.
```

For a positive normalized width `s`:

```text
a(s) = 1 + (1/49) ln((s^4 + (s/52)^2)/(s^4 + 0.432))
       + (1/18.7) ln(1 + (s/18.1)^3),

b(epsilon_r) = 0.564 ((epsilon_r - 0.9)/(epsilon_r + 3))^0.053,

epsilon_HJ(s) = (epsilon_r + 1)/2
              + (epsilon_r - 1)/2 * (1 + 10/s)^(-a(s)b(epsilon_r)),

f(s) = 6 + (2*pi - 6) exp(-(30.666/s)^0.7528),

Z_air(s) = 60 ln(f(s)/s + sqrt(1 + (2/s)^2)).
```

The finite-thickness effective-width corrections are:

```text
delta_u_1 = (tau/pi) ln(1 + (4e/tau) tanh(sqrt(6.517 u))^2),
delta_u_r = delta_u_1 * (1 + sech(sqrt(epsilon_r - 1))) / 2,
u_1 = u + delta_u_1,
u_r = u + delta_u_r.
```

Then:

```text
epsilon_eff_HJ = epsilon_HJ(u_r) * (Z_air(u_1) / Z_air(u_r))^2,
Z0_HJ = Z_air(u_1) / sqrt(epsilon_eff_HJ).
```

For the project geometry this implementation reproduces the Appendix D check values, approximately `epsilon_eff_HJ=3.3008` and `Z0_HJ=50.20 ohm`.

Both IPC-2141 and Hammerstad-Jensen are approximate quasistatic engineering models. They are not exact field solutions and cannot replace the FEM convergence evidence.
