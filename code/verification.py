"""Gate A verification problems for the reusable FEM core."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, Functional, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import dot, grad


@dataclass(frozen=True)
class VerificationRow:
    level: int
    h_max: float
    elements: int
    dofs: int
    nnz: int
    l2_error: float
    energy_error: float
    residual_inf: float
    l2_order: float | None
    energy_order: float | None


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def exact_gradient(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))


@LinearForm
def load(v, w):
    return 2.0 * np.pi**2 * exact_solution(w.x) * v


@Functional
def squared_l2_error(w):
    return (w.uh - exact_solution(w.x)) ** 2


@Functional
def squared_energy_error(w):
    diff = w.uh.grad - exact_gradient(w.x)
    return dot(diff, diff)


def _observed_order(prev_h: float, prev_error: float, h: float, error: float) -> float:
    return math.log(prev_error / error) / math.log(prev_h / h)


def solve_mms_level(level: int) -> VerificationRow:
    mesh = MeshTri().refined(level)
    basis = Basis(mesh, ElementTriP1())
    matrix = asm(laplace, basis).tocsr()
    rhs = asm(load, basis)
    boundary_dofs = basis.get_dofs().all()
    solution = solve(*condense(matrix, rhs, D=boundary_dofs))
    free_dofs = np.setdiff1d(np.arange(basis.N), boundary_dofs)
    residual = np.linalg.norm((matrix @ solution - rhs)[free_dofs], ord=np.inf)
    return VerificationRow(
        level=level,
        h_max=float(mesh.param()),
        elements=mesh.nelements,
        dofs=basis.N,
        nnz=matrix.nnz,
        l2_error=float(np.sqrt(asm(squared_l2_error, basis, uh=solution))),
        energy_error=float(np.sqrt(asm(squared_energy_error, basis, uh=solution))),
        residual_inf=float(residual),
        l2_order=None,
        energy_order=None,
    )


def run_mms(levels: list[int]) -> list[VerificationRow]:
    rows = []
    for row in (solve_mms_level(level) for level in levels):
        if rows:
            prev = rows[-1]
            row = VerificationRow(
                **{
                    **row.__dict__,
                    "l2_order": _observed_order(prev.h_max, prev.l2_error, row.h_max, row.l2_error),
                    "energy_order": _observed_order(prev.h_max, prev.energy_error, row.h_max, row.energy_error),
                }
            )
        rows.append(row)
    return rows


def write_csv(rows: list[VerificationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(VerificationRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_convergence(rows: list[VerificationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hs = np.array([row.h_max for row in rows])
    l2 = np.array([row.l2_error for row in rows])
    energy = np.array([row.energy_error for row in rows])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.loglog(hs, l2, "o-", label="L2 error")
    ax.loglog(hs, energy, "s-", label="energy error")
    ax.loglog(hs, l2[0] * (hs / hs[0]) ** 2, "--", color="C0", alpha=0.7, label="slope 2 reference")
    ax.loglog(hs, energy[0] * (hs / hs[0]), "--", color="C1", alpha=0.7, label="slope 1 reference")
    ax.set_xlabel("maximum element size h")
    ax.set_ylabel("error norm")
    ax.set_title("Gate A MMS convergence for P1 FEM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_solution(path: Path, level: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = MeshTri().refined(level)
    basis = Basis(mesh, ElementTriP1())
    matrix = asm(laplace, basis).tocsr()
    rhs = asm(load, basis)
    solution = solve(*condense(matrix, rhs, D=basis.get_dofs().all()))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    triang = ax.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, solution, levels=20)
    ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, color="k", lw=0.2, alpha=0.35)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MMS FEM solution")
    fig.colorbar(triang, ax=ax, label="u_h")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_log(rows: list[VerificationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_three = rows[-3:] if len(rows) >= 3 else rows
    mean_l2 = np.mean([row.l2_order for row in last_three[1:] if row.l2_order is not None])
    mean_energy = np.mean([row.energy_order for row in last_three[1:] if row.energy_order is not None])
    passed = mean_l2 > 1.8 and mean_energy > 0.85 and max(row.residual_inf for row in rows) < 1e-10
    with path.open("w") as stream:
        stream.write("Gate A verification: known-solution P1 FEM on unit square\n")
        stream.write("PDE: -Delta u = 2*pi^2*sin(pi*x)*sin(pi*y), u=0 on boundary\n")
        stream.write("Expected smooth P1 FEM rates: L2 about 2, energy about 1\n\n")
        for row in rows:
            stream.write(
                f"level={row.level} h={row.h_max:.6e} elements={row.elements} "
                f"dofs={row.dofs} nnz={row.nnz} L2={row.l2_error:.6e} "
                f"energy={row.energy_error:.6e} residual_inf={row.residual_inf:.3e} "
                f"L2_order={row.l2_order} energy_order={row.energy_order}\n"
            )
        stream.write(f"\nMean observed L2 order over finest available steps: {mean_l2:.3f}\n")
        stream.write(f"Mean observed energy order over finest available steps: {mean_energy:.3f}\n")
        stream.write(f"Gate A status: {'PASS' if passed else 'FAIL'}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rows = run_mms(args.levels)
    write_csv(rows, args.project_root / "tables" / "mms_convergence.csv")
    plot_solution(args.project_root / "figures" / "mms_solution.png")
    plot_convergence(rows, args.project_root / "figures" / "mms_convergence.png")
    write_log(rows, args.project_root / "logs" / "gate_A_verification.txt")


if __name__ == "__main__":
    main()
