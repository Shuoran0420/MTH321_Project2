import numpy as np

from geometry import build_baseline_geometry


def test_nominal_dimensions_and_trace_center():
    geom = build_baseline_geometry()
    cfg = geom.config
    assert cfg.H == 1.6e-3
    assert cfg.w == 3.0e-3
    assert cfg.t == 0.035e-3
    assert abs(0.5 * (cfg.trace_xmin + cfg.trace_xmax)) < 1e-15
    assert np.isclose(cfg.trace_xmax - cfg.trace_xmin, cfg.w)
    assert np.isclose(cfg.trace_ymax - cfg.trace_ymin, cfg.t)


def test_no_element_centroid_inside_conductor():
    geom = build_baseline_geometry()
    cfg = geom.config
    centroids = geom.points[geom.elements].mean(axis=1)
    inside = (
        (centroids[:, 0] > cfg.trace_xmin)
        & (centroids[:, 0] < cfg.trace_xmax)
        & (centroids[:, 1] > cfg.trace_ymin)
        & (centroids[:, 1] < cfg.trace_ymax)
    )
    assert not np.any(inside)


def test_boundaries_are_identified_and_disjoint():
    geom = build_baseline_geometry()
    cfg = geom.config
    assert geom.ground_nodes.size > 0
    assert geom.trace_nodes.size > 0
    assert geom.far_nodes.size > 0
    assert np.allclose(geom.points[geom.ground_nodes, 1], 0.0)
    trace_pts = geom.points[geom.trace_nodes]
    on_trace = (
        np.isclose(trace_pts[:, 0], cfg.trace_xmin)
        | np.isclose(trace_pts[:, 0], cfg.trace_xmax)
        | np.isclose(trace_pts[:, 1], cfg.trace_ymin)
        | np.isclose(trace_pts[:, 1], cfg.trace_ymax)
    )
    assert np.all(on_trace)
    assert np.intersect1d(geom.ground_nodes, geom.trace_nodes).size == 0
    assert np.intersect1d(geom.far_nodes, geom.trace_nodes).size == 0
    assert np.intersect1d(geom.far_nodes, geom.ground_nodes).size == 0


def test_interface_and_symmetry():
    geom = build_baseline_geometry()
    cfg = geom.config
    yvals = np.unique(np.round(geom.points[:, 1], 15))
    assert np.any(np.isclose(yvals, cfg.H))
    lookup = {(round(float(x), 15), round(float(y), 15)) for x, y in geom.points}
    for x, y in geom.points:
        assert (round(float(-x), 15), round(float(y), 15)) in lookup
