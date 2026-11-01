"""Conforming library refinement of the existing microstrip triangle mesh."""

import numpy as np
import matplotlib.tri as mtri
from geometry import GeometryData
from skfem import MeshTri


def geometry_from_mesh(mesh, parent):
    c = parent.config
    points, elements = mesh.p.T.copy(), mesh.t.T.copy()
    x, y = points.T
    close = lambda a, b: np.isclose(a, b, rtol=0, atol=1e-13)
    in_x = (x >= -c.w / 2 - 1e-13) & (x <= c.w / 2 + 1e-13)
    in_y = (y >= c.H - 1e-13) & (y <= c.H + c.t + 1e-13)
    trace = np.flatnonzero((in_x & (close(y, c.H) | close(y, c.H + c.t)))
                           | (in_y & close(np.abs(x), c.w / 2)))
    ground = np.flatnonzero(close(y, 0))
    far = np.setdiff1d(np.flatnonzero(close(np.abs(x), c.W) | close(y, c.Hair)), ground)
    regions = np.where(points[elements].mean(axis=1)[:, 1] < c.H, "substrate", "air")
    return GeometryData(c, mesh, points, elements, regions, ground, trace, far,
                        parent.conductor_polygon.copy())


def finder(geom):
    return mtri.Triangulation(geom.points[:, 0], geom.points[:, 1], geom.elements).get_trifinder()


def refine(geom, marked):
    # Reflection closure is algorithmic, not a geometric corner prescription.
    p = geom.points
    lookup = {tuple(np.round(q, 14)): i for i, q in enumerate(p)}
    reflected_nodes = np.array([lookup[(round(-q[0], 14), round(q[1], 14))] for q in p])
    triangles = {tuple(sorted(t)): i for i, t in enumerate(geom.elements)}
    mirrored = np.array([triangles[tuple(sorted(reflected_nodes[t]))] for t in geom.elements])
    closed_marked = np.union1d(marked, mirrored[marked])
    facets = geom.mesh.facets.T
    t2f = geom.mesh.t2f.T
    split = np.zeros(len(facets), dtype=bool)
    split[t2f[closed_marked].ravel()] = True
    # Two split edges would require a diagonal choice. Split the third instead;
    # only unambiguous green (one edge) and red (three edges) children remain.
    while True:
        two = np.flatnonzero(split[t2f].sum(axis=1) == 2)
        if not len(two):
            break
        split[t2f[two].ravel()] = True
    selected = np.flatnonzero(split)
    midpoint = np.full(len(facets), -1, dtype=int)
    midpoint[selected] = len(p) + np.arange(len(selected))
    points = np.vstack((p, p[facets[selected]].mean(axis=1)))
    children = []
    for tri, fs in zip(geom.elements, t2f):
        mids = {tuple(sorted(facets[f])): midpoint[f] for f in fs if split[f]}
        if not mids:
            children.append(tri.tolist())
        elif len(mids) == 1:
            (a, b), m = next(iter(mids.items()))
            c = next(int(v) for v in tri if v != a and v != b)
            children.extend([[a, m, c], [m, b, c]])
        else:
            a, b, c = tri
            ab = mids[tuple(sorted((a, b)))]; bc = mids[tuple(sorted((b, c)))]; ca = mids[tuple(sorted((c, a)))]
            children.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
    child = geometry_from_mesh(MeshTri(points.T, np.array(children).T), geom)
    centers = child.points[child.elements].mean(axis=1)
    owners = finder(geom)(centers[:, 0], centers[:, 1])
    if np.any(owners < 0):
        raise ValueError("child outside parent domain")
    actual = int(np.count_nonzero(np.bincount(owners, minlength=len(geom.elements)) > 1))
    return child, actual, len(closed_marked)


def symmetry_complete(geom):
    """Nonobtuse mirror completion: four children per original right triangle.

    All edge midpoints are added. Joining the right vertex to the hypotenuse
    midpoint creates a mirror-invariant eight-triangle rectangular cell. Unlike
    diagonal-only bisection, these initial triangles are all nonobtuse.
    """
    facets = geom.mesh.facets.T
    edge_length = np.linalg.norm(geom.points[facets[:, 1]] - geom.points[facets[:, 0]], axis=1)
    fs = geom.mesh.t2f.T
    diagonals = fs[np.arange(len(fs)), np.argmax(edge_length[fs], axis=1)]
    unique = np.arange(len(facets))
    ids = {f: len(geom.points) + i for i, f in enumerate(unique)}
    points = np.vstack((geom.points, geom.points[facets[unique]].mean(axis=1)))
    children = []
    for tri, f, tri_facets in zip(geom.elements, diagonals, fs):
        b, c = facets[f]; a = next(int(v) for v in tri if v != b and v != c)
        by_nodes = {tuple(sorted(facets[index])): ids[index] for index in tri_facets}
        ab = by_nodes[tuple(sorted((a, b)))]; ac = by_nodes[tuple(sorted((a, c)))]; bc = ids[f]
        children.extend([[a, ab, bc], [a, bc, ac], [b, bc, ab], [c, ac, bc]])
    return geometry_from_mesh(MeshTri(points.T, np.array(children).T), geom)


def symmetry_error(sol):
    """Relative RMS potential reflection error at a symmetric physical grid."""
    g = sol.geometry
    x = np.linspace(-g.config.W, g.config.W, 121)
    y = np.unique(np.r_[np.linspace(0, g.config.Hair, 101), g.config.H, g.config.H + g.config.t])
    xx, yy = np.meshgrid(x, y)
    p = np.column_stack((xx.ravel(), yy.ravel()))
    locate = finder(g)
    ids = locate(p[:, 0], p[:, 1])
    reflected = p.copy(); reflected[:, 0] *= -1
    other = locate(reflected[:, 0], reflected[:, 1])
    valid = (ids >= 0) & (other >= 0)
    def evaluate(q, tri_ids):
        centers = g.points[g.elements[tri_ids]].mean(axis=1)
        mean_values = sol.phi[g.elements[tri_ids]].mean(axis=1)
        return mean_values + np.einsum("ij,ij->i", sol.element_gradients[tri_ids], q - centers)
    v = evaluate(p[valid], ids[valid])
    vr = evaluate(reflected[valid], other[valid])
    return float(np.linalg.norm(v - vr) / np.linalg.norm(v))
