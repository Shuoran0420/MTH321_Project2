# Final Results

## Geometry

- Substrate height: `H = 1.6 mm`
- Trace width: `w = 3.0 mm`
- Trace thickness: `t = 0.035 mm`
- Substrate relative permittivity: `epsilon_r = 4.4`

## Numerical Method

- Governing equation: `-div(epsilon grad(phi)) = 0`
- Conforming triangular `P1` finite elements
- Homogeneous Neumann top/side artificial boundary
- Final engineering domain: `W/H = 8`, `Hair/H = 8`
- Final mesh: `uniform_M7_8H_neumann_P1_v1`
- Frozen mesh SHA-256: `06b7c06b25667d77c4292d49764ec42234006b041eab922e5563dfb8669b6f75`

## Final FEM Quantities

- Dielectric capacitance: `C = 1.2061977369391071e-10 F/m = 120.620 pF/m`
- All-air capacitance: `C_air = 3.5968210876660980e-11 F/m = 35.9682 pF/m`
- Effective relative permittivity: `epsilon_eff = 3.3535105237102121e+00 ≈ 3.3535`
- Per-unit-length inductance: `L = 3.0934261920033220e-07 H/m`
- Phase velocity: `v_p = 1.6370836313321367e+08 m/s`
- Characteristic impedance: `Z0 = 5.0641973836627407e+01 ohm ≈ 50.642 ohm`

## Engineering Benchmarks

- Hammerstad-Jensen: `Z0 = 5.0200690000155610e+01 ohm`, difference from FEM `0.871380%`
- IPC-2141: `Z0 = 4.9393389422188392e+01 ohm`, difference from FEM `2.465513%`

## Numerical Sensitivity Notes

- Gate-F spatial indicator: `1.555539e-03` (`0.156%`)
- Gate-C finite-domain indicator: `1.820310e-03` (`0.182%`)
- Conservative numerical sensitivity indicator: `3.375849e-03` (`0.338%`), not a confidence interval
- M7 was accepted via `M7_ASYMPTOTIC_RESOURCE_FALLBACK`; Richardson extrapolation is diagnostic only and is not substituted for the computed FEM capacitance.
