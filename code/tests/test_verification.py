from verification import run_mms


def test_mms_converges_on_four_meshes():
    rows = run_mms([2, 3, 4, 5])
    assert len(rows) == 4
    assert rows[-1].l2_error < rows[0].l2_error
    assert rows[-1].energy_error < rows[0].energy_error
    assert rows[-1].l2_order is not None
    assert rows[-1].energy_order is not None
    assert rows[-1].l2_order > 1.8
    assert rows[-1].energy_order > 0.85
    assert max(row.residual_inf for row in rows) < 1e-10
