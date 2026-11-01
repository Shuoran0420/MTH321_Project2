"""Energy and boundary-reaction capacitance estimates (units F/m)."""
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
    V0 = float(sol.geometry.config.V0)
    if not np.isfinite(V0) or V0 == 0:
        raise ValueError("non-zero finite trace voltage required")
    e2 = np.einsum("ij,ij->i", sol.element_gradients, sol.element_gradients)
    energy = float(0.5 * np.sum(sol.epsilon * e2 * sol.element_areas))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("electrostatic energy must be finite and positive")
    CE = float(2.0 * energy / V0**2)
    q_trace_signed = float(np.sum(sol.reaction_full[sol.trace_dofs]))
    q_ground_signed = float(np.sum(sol.reaction_full[sol.ground_dofs]))
    Q_trace = abs(q_trace_signed)
    # Preserve the project's original signed ground-charge convention.
    Q_ground = q_ground_signed
    CQ = Q_trace / abs(V0)
    delta_EQ = abs(CE - CQ) / CE
    scale = max(abs(q_trace_signed), abs(q_ground_signed), CE * abs(V0), 1e-300)
    balance = abs(q_trace_signed + q_ground_signed) / scale
    return CapacitanceResult(energy, CE, CQ, delta_EQ, Q_trace, Q_ground, balance)
