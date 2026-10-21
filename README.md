# MTH321 Project 2
## 2D Finite Element Analysis of a Microstrip Transmission Line

This project uses the Finite Element Method (FEM) to analyse a two-dimensional microstrip transmission line.

The main objective is to solve the electrostatic potential problem, calculate the capacitance of the structure, obtain the effective permittivity, and estimate the characteristic impedance \(Z_0\).

The project also includes numerical verification, mesh-convergence studies, domain-truncation studies, and engineering benchmark comparisons.

---

## 1. Project Overview

The main workflow is

Geometry  
→ PDE model  
→ FEM discretisation  
→ Solver verification  
→ Domain study  
→ Mesh convergence  
→ Capacitance calculation  
→ Effective permittivity  
→ Characteristic impedance

The governing electrostatic equation is

\[
-\nabla \cdot (\varepsilon \nabla \phi) = 0.
\]

The signal conductor is set to \(1\,\mathrm{V}\), while the ground plane is set to \(0\,\mathrm{V}\).

The final characteristic impedance is calculated from

\[
Z_0 =
\frac{1}
{c_0\sqrt{C C_{\mathrm{air}}}}.
\]

---

## 2. Geometry and Material Parameters

The main microstrip parameters are:

- Substrate thickness: \(H = 1.6\,\mathrm{mm}\)
- Trace width: \(w = 3.0\,\mathrm{mm}\)
- Trace thickness: \(t = 0.035\,\mathrm{mm}\)
- Substrate relative permittivity: \(\varepsilon_r = 4.4\)
- Air relative permittivity: \(\varepsilon_r = 1\)

The final numerical model uses an \(8H\) computational domain.

---

## 3. Numerical Method

The problem is solved using conforming triangular \(P_1\) finite elements.

The implementation includes:

- FEM matrix assembly
- Dirichlet boundary-condition elimination
- discontinuous material permittivity
- electrostatic energy calculation
- charge-based capacitance calculation
- sparse direct solution
- mesh refinement
- convergence analysis

The air-substrate interface is handled through the weak FEM formulation.

---

## 4. Verification and Numerical Studies

The project contains the following numerical studies:

### Solver verification

A manufactured solution is used to verify the FEM implementation.

Observed convergence rates are approximately:

- \(L^2\) error: order 2
- energy error: order 1

### Domain-truncation study

The following domain sizes are compared:

- \(3H\)
- \(5H\)
- \(8H\)
- \(12H\)

The \(8H\) domain differs from the tested \(12H\) reference by approximately

\[
0.182\%.
\]

### Mesh-convergence study

Uniform mesh refinement is used to investigate the conductor-corner singularity.

The observed energy-norm convergence rate is approximately

\[
p_E \approx 0.623.
\]

### Graded refinement

A corner-focused graded refinement strategy is compared with uniform refinement using:

- degrees of freedom
- matrix nonzeros
- computational time

For the tested grading strategy, uniform refinement was more efficient at the formally supported error targets.

---

## 5. Final Numerical Results

The final dielectric solution uses the frozen mesh

`uniform_M7_8H_neumann_P1_v1`

The main results are:

| Quantity | Value |
|---|---:|
| Dielectric capacitance \(C\) | 120.620 pF/m |
| All-air capacitance \(C_{\mathrm{air}}\) | 35.9682 pF/m |
| Effective permittivity \(\varepsilon_{\mathrm{eff}}\) | 3.3535 |
| Characteristic impedance \(Z_0\) | 50.642 ohm |

The effective permittivity is calculated using

\[
\varepsilon_{\mathrm{eff}}
=
\frac{C}{C_{\mathrm{air}}}.
\]

---

## 6. Engineering Validation

The FEM impedance is compared with two engineering approximations:

| Method | \(Z_0\) |
|---|---:|
| FEM | 50.642 ohm |
| Hammerstad-Jensen | 50.201 ohm |
| IPC-2141 | 49.393 ohm |

The relative differences from the FEM result are approximately:

- Hammerstad-Jensen: 0.87%
- IPC-2141: 2.47%

These formulas are used as independent engineering benchmarks and do not replace the FEM convergence studies.

---

## 7. Repository Structure

A typical project structure is:

```text
project/
│
├── code/
│   ├── run_all.py
│   ├── FEM solver modules
│   └── numerical study scripts
│
├── tests/
│   └── automated tests
│
├── data/
│   └── processed numerical data
│
├── results/
│   └── final CSV and numerical outputs
│
├── figures/
│   └── generated figures
│
├── report/
│   ├── main.tex
│   └── final_report.pdf
│
├── presentation/
│   ├── main.tex
│   └── final_presentation.pdf
│
├── logs/
│   └── validation and reproducibility logs
│
├── README.md
└── requirements.txt
