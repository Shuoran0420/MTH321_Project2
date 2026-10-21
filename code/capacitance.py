"""Capacitance extraction diagnostics for the microstrip baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fem_solver import MicrostripSolution


@dataclass(frozen=True)
class CapacitanceResult:
    energy: float
    CE: float
    CQ: float
    delta_EQ: float
    Q_trace: float
    Q_ground: float
    charge_balance_defect: float


def extract_capacitance(sol: MicrostripSolution) -> CapacitanceResult:
    e2 = np.einsum("ij,ij->i", sol.element_gradients, sol.element_gradients)
    energy = float(0.5 * np.sum(sol.epsilon * e2 * sol.element_areas))
    V0 = sol.geometry.config.V0
    CE = float(2.0 * energy / V0**2)
    raw_trace = float(np.sum(sol.reaction_full[sol.trace_dofs]))
    raw_ground = float(np.sum(sol.reaction_full[sol.ground_dofs]))
    Q_trace = abs(raw_trace)
    Q_ground = raw_ground
    CQ = float(Q_trace / abs(V0))
    delta_EQ = float(abs(CE - CQ) / CE)
    balance_scale = max(abs(raw_trace), abs(raw_ground), CE * abs(V0), 1e-300)
    charge_balance_defect = float(abs(raw_trace + raw_ground) / balance_scale)
    return CapacitanceResult(energy, CE, CQ, delta_EQ, Q_trace, Q_ground, charge_balance_defect)
