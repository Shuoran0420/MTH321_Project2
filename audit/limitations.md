# Limitations

The present model is a two-dimensional electrostatic/quasi-TEM microstrip calculation. The final capacitance, effective permittivity, and characteristic impedance are nominal quasistatic values for the specified idealized geometry.

The model does not include:

- frequency-dependent dielectric dispersion;
- dielectric loss;
- conductor loss;
- copper surface roughness;
- solder-mask loading;
- radiation;
- anisotropic substrate behaviour;
- fabrication tolerance or etch-width variation;
- finite conductivity inside the trace or ground plane.

The signal trace and ground are treated as perfect conductors. The conductor interior is removed from the field domain and represented by Dirichlet boundaries.

The computational domain is finite. Gate C reports an 8H-to-12H finite-domain sensitivity indicator of about `0.182031%`; this is not an exact infinite-domain error bound.

Gate F selected the final dielectric mesh through the `M7_ASYMPTOTIC_RESOURCE_FALLBACK` path. The preferred direct `0.1%` successive-capacitance target was not reached by M7, and M8 was not run because of sparse-direct factorisation resource risk. Richardson extrapolation is used only as a convergence diagnostic; the authoritative capacitance remains the computed M7 FEM value.

Gate H compares the FEM result with Hammerstad-Jensen and IPC-2141 engineering approximations. These formulas are useful independent cross-checks, not exact solutions and not proof of FEM spatial convergence. The 50-ohm comparison is design context only, not a numerical reference solution.
