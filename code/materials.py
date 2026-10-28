"""Piecewise dielectric material data for the microstrip model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON_0 = 8.8541878128e-12


@dataclass(frozen=True)
class MaterialConfig:
    epsilon_r_substrate: float = 4.4
    epsilon_r_air: float = 1.0


def relative_permittivity(element_regions: np.ndarray, cfg: MaterialConfig = MaterialConfig()) -> np.ndarray:
    eps_r = np.empty(element_regions.shape, dtype=float)
    eps_r[element_regions == "substrate"] = cfg.epsilon_r_substrate
    eps_r[element_regions == "air"] = cfg.epsilon_r_air
    if np.any(~np.isfinite(eps_r)) or np.any(eps_r <= 0.0):
        raise ValueError("invalid element relative permittivity")
    return eps_r


def permittivity(element_regions: np.ndarray, cfg: MaterialConfig = MaterialConfig()) -> np.ndarray:
    return EPSILON_0 * relative_permittivity(element_regions, cfg)
