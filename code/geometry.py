"""Geometry and mesh generation for the microstrip baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from skfem import MeshTri


@dataclass(frozen=True)
class GeometryConfig:
    H: float = 1.6e-3
    w: float = 3.0e-3
    t: float = 0.035e-3
    W_over_H: float = 8.0
    Hair_over_H: float = 8.0
    V0: float = 1.0
    nx_outer: int = 96
    ny_substrate: int = 40
    ny_air: int = 80

    @property
    def W(self) -> float:
        return self.W_over_H * self.H

    @property
    def Hair(self) -> float:
        return self.Hair_over_H * self.H

    @property
    def trace_xmin(self) -> float:
        return -0.5 * self.w

    @property
    def trace_xmax(self) -> float:
        return 0.5 * self.w

    @property
    def trace_ymin(self) -> float:
        return self.H

    @property
    def trace_ymax(self) -> float:
        return self.H + self.t


@dataclass
class GeometryData:
    config: GeometryConfig
    mesh: MeshTri
    points: np.ndarray
    elements: np.ndarray
    element_regions: np.ndarray
    ground_nodes: np.ndarray
    trace_nodes: np.ndarray
    far_nodes: np.ndarray
    conductor_polygon: np.ndarray


def load_config(path: Path) -> GeometryConfig:
    raw = yaml.safe_load(path.read_text())
    geom = raw["geometry"]
    mesh = raw["mesh"]
    bc = raw["boundary_conditions"]
    return GeometryConfig(
        H=float(geom["H"]),
        w=float(geom["w"]),
        t=float(geom["t"]),
        W_over_H=float(geom["W_over_H"]),
        Hair_over_H=float(geom["Hair_over_H"]),
        V0=float(bc["V0"]),
        nx_outer=int(mesh["nx_outer"]),
        ny_substrate=int(mesh["ny_substrate"]),
        ny_air=int(mesh["ny_air"]),
    )


def _axis_with_required(start: float, stop: float, n: int, required: list[float]) -> np.ndarray:
    base = np.linspace(start, stop, n + 1)
    values = np.concatenate([base, np.array(required, dtype=float)])
    values = np.unique(np.round(values, decimals=15))
    values.sort()
    return values


def _inside_conductor(x: np.ndarray, y: np.ndarray, cfg: GeometryConfig, tol: float = 1e-14) -> np.ndarray:
    return (
        (x > cfg.trace_xmin + tol)
        & (x < cfg.trace_xmax - tol)
        & (y > cfg.trace_ymin + tol)
        & (y < cfg.trace_ymax - tol)
    )


def _on_trace_boundary(x: np.ndarray, y: np.ndarray, cfg: GeometryConfig, tol: float = 1e-12) -> np.ndarray:
    in_x = (x >= cfg.trace_xmin - tol) & (x <= cfg.trace_xmax + tol)
    in_y = (y >= cfg.trace_ymin - tol) & (y <= cfg.trace_ymax + tol)
    on_vertical = (np.isclose(x, cfg.trace_xmin, atol=tol) | np.isclose(x, cfg.trace_xmax, atol=tol)) & in_y
    on_horizontal = (np.isclose(y, cfg.trace_ymin, atol=tol) | np.isclose(y, cfg.trace_ymax, atol=tol)) & in_x
    return on_vertical | on_horizontal


def build_geometry(cfg: GeometryConfig) -> GeometryData:
    xs = _axis_with_required(-cfg.W, cfg.W, cfg.nx_outer, [0.0, cfg.trace_xmin, cfg.trace_xmax])
    ys_sub = _axis_with_required(0.0, cfg.H, cfg.ny_substrate, [cfg.H])
    ys_air = _axis_with_required(cfg.H, cfg.Hair, cfg.ny_air, [cfg.trace_ymax])
    ys = np.unique(np.concatenate([ys_sub, ys_air]))
    ys.sort()

    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points_all = np.column_stack([xx.ravel(), yy.ravel()])
    node_id = np.arange(points_all.shape[0]).reshape(len(ys), len(xs))

    tris: list[list[int]] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            quad = [
                node_id[j, i],
                node_id[j, i + 1],
                node_id[j + 1, i],
                node_id[j + 1, i + 1],
            ]
            cell_x = np.array([points_all[k, 0] for k in quad])
            cell_y = np.array([points_all[k, 1] for k in quad])
            cx = float(cell_x.mean())
            cy = float(cell_y.mean())
            if _inside_conductor(np.array([cx]), np.array([cy]), cfg)[0]:
                continue
            tris.append([quad[0], quad[1], quad[3]])
            tris.append([quad[0], quad[3], quad[2]])

    used = np.unique(np.array(tris, dtype=int).ravel())
    remap = -np.ones(points_all.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    points = points_all[used]
    elements = remap[np.array(tris, dtype=int)]

    mesh = MeshTri(points.T, elements.T)
    centroids = points[elements].mean(axis=1)
    element_regions = np.where(centroids[:, 1] < cfg.H - 1e-13, "substrate", "air")
    x = points[:, 0]
    y = points[:, 1]
    ground_nodes = np.flatnonzero(np.isclose(y, 0.0, atol=1e-12))
    trace_nodes = np.flatnonzero(_on_trace_boundary(x, y, cfg))
    far_nodes = np.flatnonzero(
        (
            np.isclose(x, -cfg.W, atol=1e-12)
            | np.isclose(x, cfg.W, atol=1e-12)
            | np.isclose(y, cfg.Hair, atol=1e-12)
        )
        & ~np.isin(np.arange(points.shape[0]), ground_nodes)
        & ~np.isin(np.arange(points.shape[0]), trace_nodes)
    )
    conductor_polygon = np.array(
        [
            [cfg.trace_xmin, cfg.trace_ymin],
            [cfg.trace_xmax, cfg.trace_ymin],
            [cfg.trace_xmax, cfg.trace_ymax],
            [cfg.trace_xmin, cfg.trace_ymax],
            [cfg.trace_xmin, cfg.trace_ymin],
        ]
    )
    return GeometryData(cfg, mesh, points, elements, element_regions, ground_nodes, trace_nodes, far_nodes, conductor_polygon)


def build_baseline_geometry(project_root: Path | None = None) -> GeometryData:
    root = project_root or Path(__file__).resolve().parents[1]
    return build_geometry(load_config(root / "config" / "baseline.yaml"))
