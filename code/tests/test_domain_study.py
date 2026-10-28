import csv
from pathlib import Path

import numpy as np

from domain_study import DOMAIN_RATIOS, build_domain_geometry, near_field_equivalence


def test_required_domain_ratios_and_geometry_derivation():
    assert DOMAIN_RATIOS == [3.0, 5.0, 8.0, 12.0]
    for ratio in DOMAIN_RATIOS:
        geom = build_domain_geometry(ratio, "R0")
        assert np.isclose(geom.config.W, ratio * geom.config.H)
        assert np.isclose(geom.config.Hair, ratio * geom.config.H)
        assert np.isclose(2.0 * geom.config.W, 2.0 * ratio * geom.config.H)
        assert np.isclose(geom.config.trace_xmax - geom.config.trace_xmin, geom.config.w)
        assert np.isclose(geom.config.trace_ymax - geom.config.trace_ymin, geom.config.t)


def test_near_field_mesh_equivalence():
    geoms = [build_domain_geometry(r, "R0") for r in DOMAIN_RATIOS]
    ok, msg = near_field_equivalence(geoms)
    assert ok, msg


def test_domain_outputs_if_present_are_consistent():
    path = Path("tables/domain_truncation.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    main = [r for r in rows if r["resolution_id"] == "R0"]
    assert [float(r["W_over_H"]) for r in main] == DOMAIN_RATIOS
    c12 = next(float(r["CE_F_per_m"]) for r in main if float(r["W_over_H"]) == 12.0)
    for r in main:
        assert float(r["CE_F_per_m"]) > 0.0
        assert float(r["CQ_F_per_m"]) > 0.0
        assert float(r["delta_EQ"]) < 1e-10
        assert 0.0 <= float(r["min_phi"]) <= 1.0
        assert 0.0 <= float(r["max_phi"]) <= 1.0
        expected = abs(float(r["CE_F_per_m"]) - c12) / abs(c12)
        assert np.isclose(float(r["relative_to_C12"]), expected)
    assert next(float(r["relative_to_C12"]) for r in main if float(r["W_over_H"]) == 12.0) == 0.0
    assert any(r["resolution_id"] == "R1" and float(r["W_over_H"]) == 12.0 for r in rows)
