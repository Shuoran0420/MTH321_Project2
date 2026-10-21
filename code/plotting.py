"""Report-quality baseline plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from fem_solver import MicrostripSolution
from geometry import GeometryData


def _add_trace(ax, geom: GeometryData, **kwargs) -> None:
    cfg = geom.config
    ax.add_patch(
        Rectangle(
            (cfg.trace_xmin * 1e3, cfg.trace_ymin * 1e3),
            cfg.w * 1e3,
            cfg.t * 1e3,
            fill=False,
            lw=1.3,
            **kwargs,
        )
    )


def plot_geometry_full_domain(geom: GeometryData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = geom.config
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.add_patch(Rectangle((-cfg.W * 1e3, 0), 2 * cfg.W * 1e3, cfg.H * 1e3, facecolor="#cfe8ff", edgecolor="none", label="FR-4 substrate, eps_r=4.4"))
    ax.add_patch(Rectangle((-cfg.W * 1e3, cfg.H * 1e3), 2 * cfg.W * 1e3, (cfg.Hair - cfg.H) * 1e3, facecolor="#fff4cf", edgecolor="none", label="air, eps_r=1.0"))
    ax.add_patch(Rectangle((cfg.trace_xmin * 1e3, cfg.trace_ymin * 1e3), cfg.w * 1e3, cfg.t * 1e3, facecolor="white", edgecolor="crimson", lw=1.7, label="signal conductor removed, phi=1 V"))
    ax.plot([-cfg.W * 1e3, cfg.W * 1e3], [0, 0], color="black", lw=2.0, label="ground, phi=0")
    ax.plot([-cfg.W * 1e3, cfg.W * 1e3], [cfg.H * 1e3, cfg.H * 1e3], color="navy", lw=1.1, ls="--", label="air-substrate interface")
    ax.plot([-cfg.W * 1e3, -cfg.W * 1e3, cfg.W * 1e3, cfg.W * 1e3], [0, cfg.Hair * 1e3, cfg.Hair * 1e3, 0], color="dimgray", lw=1.0, ls=":", label="artificial far boundary")
    ax.annotate("2W = 25.6 mm full span", xy=(0.0, 0.96 * cfg.Hair * 1e3), ha="center", fontsize=9)
    ax.annotate("W = 12.8 mm half-width", xy=(-0.5 * cfg.W * 1e3, 0.90 * cfg.Hair * 1e3), ha="center", fontsize=8)
    ax.annotate("Hair = 12.8 mm", xy=((-cfg.W + 0.0008) * 1e3, 0.52 * cfg.Hair * 1e3), rotation=90, va="center", fontsize=9)
    ax.annotate("H = 1.6 mm", xy=((-cfg.W + 0.0016) * 1e3, 0.5 * cfg.H * 1e3), rotation=90, va="center", fontsize=9)
    ax.annotate("w = 3.0 mm; t = 0.035 mm", xy=(0, (cfg.trace_ymax + 0.0003) * 1e3), ha="center", fontsize=8)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Full computational domain: microstrip baseline geometry")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_geometry_near_trace(geom: GeometryData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = geom.config
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.add_patch(Rectangle((-3.2, 1.25), 6.4, (cfg.H - 0.00125) * 1e3, facecolor="#cfe8ff", edgecolor="none", label="FR-4 substrate"))
    ax.add_patch(Rectangle((-3.2, cfg.H * 1e3), 6.4, 0.85, facecolor="#fff4cf", edgecolor="none", label="air"))
    ax.add_patch(Rectangle((cfg.trace_xmin * 1e3, cfg.trace_ymin * 1e3), cfg.w * 1e3, cfg.t * 1e3, facecolor="white", edgecolor="crimson", lw=2.0, label="signal conductor removed"))
    ax.axhline(cfg.H * 1e3, color="navy", lw=1.1, ls="--", label="air-substrate interface")
    ax.annotate("w = 3.0 mm", xy=(0, (cfg.trace_ymax + 0.00018) * 1e3), ha="center", fontsize=9)
    bracket_x = (cfg.trace_xmin - 0.00018) * 1e3
    y0 = cfg.trace_ymin * 1e3
    y1 = cfg.trace_ymax * 1e3
    ax.plot([bracket_x, bracket_x], [y0, y1], color="crimson", lw=1.2)
    ax.plot([bracket_x - 0.05, bracket_x + 0.05], [y0, y0], color="crimson", lw=1.2)
    ax.plot([bracket_x - 0.05, bracket_x + 0.05], [y1, y1], color="crimson", lw=1.2)
    ax.annotate("t = 0.035 mm", xy=(bracket_x - 0.09, y1 + 0.02), ha="right", va="bottom", fontsize=9, color="crimson")
    ax.text(0.02, 0.03, "Zoomed view; not the full computational domain.", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(1.25, 2.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Near-trace geometry zoom")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mesh_full(sol: MicrostripSolution, path: Path) -> None:
    geom = sol.geometry
    cfg = geom.config
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.triplot(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, color="0.25", lw=0.08)
    _add_trace(ax, geom, edgecolor="crimson")
    ax.axhline(cfg.H * 1e3, color="navy", lw=1.0, ls="--")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Full-domain baseline FEM mesh")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mesh_near_trace(sol: MicrostripSolution, path: Path) -> None:
    geom = sol.geometry
    cfg = geom.config
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.triplot(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, color="0.2", lw=0.25)
    _add_trace(ax, geom, edgecolor="crimson")
    ax.axhline(cfg.H * 1e3, color="navy", lw=1.0, ls="--")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(1.25, 2.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Near-trace baseline FEM mesh zoom")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_potential(sol: MicrostripSolution, path: Path) -> None:
    geom = sol.geometry
    cfg = geom.config
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    contour = ax.tricontourf(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, sol.phi, levels=30)
    _add_trace(ax, geom, edgecolor="white")
    ax.axhline(cfg.H * 1e3, color="white", lw=1.0, ls="--")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Finite-element potential phi_h (V)")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    fig.colorbar(contour, ax=ax, label="V")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_element_scalar(sol: MicrostripSolution, values: np.ndarray, path: Path, title: str, label: str, closeup: bool = False) -> None:
    geom = sol.geometry
    cfg = geom.config
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    artist = ax.tripcolor(geom.points[:, 0] * 1e3, geom.points[:, 1] * 1e3, geom.elements, facecolors=values, shading="flat")
    _add_trace(ax, geom, edgecolor="crimson")
    ax.axhline(cfg.H * 1e3, color="white", lw=1.0, ls="--")
    if closeup:
        ax.set_xlim(cfg.trace_xmin * 1e3 - 0.35, cfg.trace_xmin * 1e3 + 0.65)
        ax.set_ylim(cfg.H * 1e3 - 0.18, cfg.H * 1e3 + 0.22)
        ax.set_aspect("equal")
    else:
        ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
        ax.set_ylim(0, cfg.Hair * 1e3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.colorbar(artist, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_element_scalar_log(sol: MicrostripSolution, values: np.ndarray, path: Path, title: str, label: str) -> None:
    geom = sol.geometry
    cfg = geom.config
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = values[values > 0.0]
    floor = float(max(np.min(positive), np.max(values) * 1e-12)) if positive.size else 1e-300
    plot_values = np.maximum(values, floor)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    artist = ax.tripcolor(
        geom.points[:, 0] * 1e3,
        geom.points[:, 1] * 1e3,
        geom.elements,
        facecolors=plot_values,
        shading="flat",
        norm=LogNorm(vmin=floor, vmax=float(np.max(plot_values))),
    )
    _add_trace(ax, geom, edgecolor="crimson")
    ax.axhline(cfg.H * 1e3, color="white", lw=1.0, ls="--")
    ax.set_xlim(-cfg.W * 1e3, cfg.W * 1e3)
    ax.set_ylim(0, cfg.Hair * 1e3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title + f" (log colour scale; floor={floor:.2e})")
    fig.colorbar(artist, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_baseline_figures(sol: MicrostripSolution, root: Path) -> None:
    grad_norm = np.linalg.norm(sol.element_gradients, axis=1)
    energy_density = 0.5 * sol.epsilon * grad_norm**2
    plot_geometry_full_domain(sol.geometry, root / "figures" / "geometry_full_domain.png")
    plot_geometry_near_trace(sol.geometry, root / "figures" / "geometry_near_trace.png")
    plot_geometry_near_trace(sol.geometry, root / "figures" / "geometry_labelled.png")
    plot_mesh_full(sol, root / "figures" / "microstrip_mesh_full.png")
    plot_mesh_near_trace(sol, root / "figures" / "microstrip_mesh_near_trace.png")
    plot_mesh_near_trace(sol, root / "figures" / "microstrip_mesh.png")
    plot_potential(sol, root / "figures" / "potential.png")
    plot_element_scalar(sol, grad_norm, root / "figures" / "electric_field.png", "Electric-field magnitude |E_h|", "V/m")
    plot_element_scalar(sol, grad_norm, root / "figures" / "electric_field_corner_zoom.png", "Lower trace-corner electric-field zoom", "V/m", closeup=True)
    plot_element_scalar(sol, energy_density, root / "figures" / "energy_density_full_linear.png", "Electrostatic energy density, full domain", "J/m^3")
    plot_element_scalar(sol, energy_density, root / "figures" / "energy_density.png", "Electrostatic energy density, full domain", "J/m^3")
    plot_element_scalar(sol, energy_density, root / "figures" / "energy_density_near_trace.png", "Electrostatic energy density near trace", "J/m^3", closeup=True)
    plot_element_scalar_log(sol, energy_density, root / "figures" / "energy_density_log.png", "Electrostatic energy density", "J/m^3")
    plot_element_scalar(sol, grad_norm, root / "figures" / "trace_corner_closeup.png", "Trace-corner electric-field close-up", "V/m", closeup=True)
