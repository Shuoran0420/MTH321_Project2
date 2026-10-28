"""Exact integration of piecewise-constant P1 gradients on a triangle overlay."""

from dataclasses import dataclass
import numpy as np
import shapely
from shapely import STRtree
from adaptive_mesh import finder


@dataclass
class EnergyError:
    absolute: float
    relative: float
    reference_norm: float
    crossed_triangles: int
    overlay_area_relative_defect: float


def overlay_energy_error(coarse, reference):
    """Containment fast path, exact GEOS polygon intersections for crossings.

    Unlike centroid sampling alone, this remains valid when reference triangles
    cross coarse triangle edges or when local coarse cells are smaller than ref.
    """
    cg, rg = coarse.geometry, reference.geometry
    scale = rg.config.H
    rt = rg.points[rg.elements] / scale
    ct = cg.points[cg.elements] / scale
    centers = rt.mean(axis=1)
    locate = finder(cg)
    owner = locate(centers[:, 0] * scale, centers[:, 1] * scale)
    if np.any(owner < 0):
        raise ValueError("reference centroid outside coarse field domain")
    tri = ct[owner]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d = rt - tri[:, :1]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    u = (d[:, :, 0] * b[:, None, 1] - d[:, :, 1] * b[:, None, 0]) / det[:, None]
    v = (a[:, None, 0] * d[:, :, 1] - a[:, None, 1] * d[:, :, 0]) / det[:, None]
    contained = np.all((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9), axis=1)
    diff = reference.element_gradients[contained] - coarse.element_gradients[owner[contained]]
    value = np.sum(reference.epsilon[contained] * np.sum(diff**2, axis=1) * reference.element_areas[contained])
    crossed = np.flatnonzero(~contained)
    max_defect = 0.0
    if len(crossed):
        cp = shapely.polygons(ct)
        tree = STRtree(cp)
        # Bound temporary overlay arrays, including the large M7 mesh.
        for start in range(0, len(crossed), 10000):
            ids = crossed[start:start + 10000]
            rp = shapely.polygons(rt[ids])
            rid, cid = tree.query(rp, predicate="intersects")
            area = shapely.area(shapely.intersection(rp[rid], cp[cid])) * scale**2
            positive = area > 1e-30
            rid, cid, area = rid[positive], cid[positive], area[positive]
            if np.any(cg.element_regions[cid] != rg.element_regions[ids[rid]]):
                # GEOS may return negligible slivers at an aligned interface.
                wrong = cg.element_regions[cid] != rg.element_regions[ids[rid]]
                if np.any(area[wrong] > 1e-10 * reference.element_areas[ids[rid[wrong]]]):
                    raise ValueError("material mismatch on common overlay")
            covered = np.bincount(rid, weights=area, minlength=len(ids))
            defect = np.max(np.abs(covered - reference.element_areas[ids]) / reference.element_areas[ids])
            max_defect = max(max_defect, float(defect))
            diff = reference.element_gradients[ids[rid]] - coarse.element_gradients[cid]
            value += np.sum(reference.epsilon[ids[rid]] * np.sum(diff**2, axis=1) * area)
    if max_defect > 1e-7:
        raise ValueError(f"incomplete triangle overlay: {max_defect}")
    norm_sq = np.sum(reference.epsilon * np.sum(reference.element_gradients**2, axis=1) * reference.element_areas)
    return EnergyError(float(np.sqrt(value)), float(np.sqrt(value / norm_sq)),
                       float(np.sqrt(norm_sq)), len(crossed), max_defect)
