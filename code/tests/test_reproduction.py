"""Integration safeguards without expensive final-reference solves."""
import json
from pathlib import Path

import pytest

from reproduction import check_close, snapshot, seed_workspace, run_command


def test_comparison_rejects_material_changes_and_nonfinite():
    assert check_close('same', 1.0, 1.0 + 1e-12)['status'] == 'PASS'
    with pytest.raises(RuntimeError):
        check_close('changed', 1.0, 1.01)
    with pytest.raises(RuntimeError):
        check_close('nan', 1.0, float('nan'))


def test_backup_is_independent_of_source_mutation(tmp_path):
    root = tmp_path/'root'; (root/'tables').mkdir(parents=True)
    p = root/'tables/final.csv'; p.write_text('CE\n1.2e-10\n')
    manifest = snapshot(root, tmp_path/'backup')
    assert 'tables/final.csv' in manifest
    p.write_text('changed')
    assert (tmp_path/'backup/tables/final.csv').read_text() == 'CE\n1.2e-10\n'


def test_command_failure_is_not_reported_as_success(tmp_path):
    import sys
    with pytest.raises(RuntimeError, match='command failed'):
        run_command([sys.executable, '-c', 'raise SystemExit(3)'], tmp_path, tmp_path/'failure.log')
    assert 'Traceback' not in (tmp_path/'failure.log').read_text()


def test_overlay_entry_uses_verified_modules_and_preserves_guard(monkeypatch, tmp_path):
    import graded_study
    import residual_adaptive_study
    import adaptive_analysis
    (tmp_path/'logs').mkdir()
    calls = []
    monkeypatch.setattr(residual_adaptive_study, 'run', lambda **kwargs: calls.append('solve_overlay'))
    monkeypatch.setattr(adaptive_analysis, 'main', lambda: calls.append('validate_and_analyze'))
    result = graded_study.run_graded_overlay_study(tmp_path)
    assert calls == ['solve_overlay', 'validate_and_analyze']
    assert result['integration'] == 'triangle_overlay'
    log = json.loads((tmp_path/'logs/graded_overlay_routing.json').read_text())
    assert all(not r['valid'] and r['crossed_triangles'] > 0 for r in log['common_refinement_diagnostics'])
    with pytest.raises(RuntimeError, match='common-overlay'):
        graded_study.run_graded_study(tmp_path, quick=True)
