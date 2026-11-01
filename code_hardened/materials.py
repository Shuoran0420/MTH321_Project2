"""Piecewise-constant permittivity on substrate and air triangles."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPSILON_0 = 8.8541878128e-12

@dataclass(frozen=True)
class MaterialConfig:
    epsilon_r_substrate: float = 4.4
    epsilon_r_air: float = 1.0


def relative_permittivity(element_regions: np.ndarray, cfg: MaterialConfig = MaterialConfig()) -> np.ndarray:
    regions = np.asarray(element_regions)
    if regions.ndim != 1:
        raise ValueError("element_regions must be one-dimensional")
    known = np.isin(regions, ("substrate", "air"))
    if not np.all(known):
        raise ValueError(f"unknown material region(s): {np.unique(regions[~known]).tolist()}")
    vals = np.array([cfg.epsilon_r_substrate, cfg.epsilon_r_air], dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError("relative permittivities must be finite and positive")
    return np.where(regions == "substrate", vals[0], vals[1]).astype(float)


def permittivity(element_regions: np.ndarray, cfg: MaterialConfig = MaterialConfig()) -> np.ndarray:
    return EPSILON_0 * relative_permittivity(element_regions, cfg)
