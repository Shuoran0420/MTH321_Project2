"""Isolated, fail-closed reproduction of frozen studies and the overlay extension."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_close(label, expected, actual, rtol=1e-9, atol=0.0):
    expected, actual = float(expected), float(actual)
    if not np.isfinite(actual) or not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise RuntimeError(f'{label}: frozen={expected:.17g}, reproduced={actual:.17g}')
    return dict(check=label, frozen=expected, reproduced=actual,
                relative_difference=abs(actual-expected)/max(abs(expected), 1e-300), status='PASS')


def protected_files(root):
    return sorted(p for folder in ('tables', 'data', 'logs', 'results', 'figures', 'report', 'presentation', 'config')
                  for p in (root / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.name not in {'.DS_Store', 'ae_full_run_console.txt'})


def resource_preflight(root):
    disk = shutil.disk_usage(root).free
    available = None
    if sys.platform == 'darwin':
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
        pressure = subprocess.check_output(['memory_pressure'], text=True)
        import re
        match = re.search(r'System-wide memory free percentage: (\d+)%', pressure)
        if match:
            available = total * int(match[1]) / 100
    elif Path('/proc/meminfo').exists():
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(values['MemAvailable'].split()[0]) * 1024
    # Conservative M7 process envelope includes Python arrays and direct-solver fill-in.
    required_memory = 4 * 1024**3
    passed = available is not None and available >= required_memory and disk >= 2 * 1024**3
    return dict(platform=platform.platform(), available_memory_bytes=available,
                required_available_memory_bytes=required_memory, free_disk_bytes=disk,
                execution='serial subprocesses; M7 maximum; no M8',
                decision='FEASIBLE' if passed else 'NOT_RUN_RESOURCE_LIMIT')


def snapshot(root, destination):
    manifest = {}
    for source in protected_files(root):
        name = source.relative_to(root)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(name)] = digest(source)
    return manifest


def seed_workspace(root, target):
    target.mkdir()
    for folder in ('code', 'config', 'data', 'tables', 'logs', 'figures', 'results', 'report'):
        shutil.copytree(root / folder, target / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '.DS_Store', '*.pyc'))


def run_command(command, cwd, log):
    started = time.perf_counter()
    print('RUN', ' '.join(command), flush=True)
    with log.open('w') as stream:
        stream.write('Command: ' + ' '.join(command) + '\n'); stream.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        for line in process.stdout:
            stream.write(line); stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    elapsed = time.perf_counter() - started
    if code:
        raise RuntimeError(f'command failed ({code}); see {log}')
    return elapsed


def compare_outputs(root, physical, extension):
    checks = []
    for name, fields in {
        'final_dielectric_capacitance.csv': ['CE', 'CQ', 'spatial_indicator', 'domain_indicator'],
        'final_air_capacitance.csv': ['CE_air', 'CQ_air'],
        'final_effective_permittivity.csv': ['C_dielectric', 'C_air', 'epsilon_eff'],
        'final_characteristic_impedance.csv': ['Z0_FEM_ohm', 'L_H_per_m', 'phase_velocity_m_per_s'],
    }.items():
        a, b = rows(root/'tables'/name)[0], rows(physical/'tables'/name)[0]
        for field in fields:
            checks.append(check_close(name+':'+field, a[field], b[field]))
    for name, keys, fields in [
        ('mms_convergence.csv', ['level'], ['l2_error', 'energy_error']),
        ('domain_truncation.csv', ['W_over_H', 'resolution_id'], ['CE_F_per_m', 'relative_to_C12']),
        ('uniform_convergence.csv', ['mesh_id'], ['relative_energy_error']),
        ('engineering_benchmarks.csv', ['method'], ['Z0_ohm']),
    ]:
        old, new = rows(root/'tables'/name), rows(physical/'tables'/name)
        for a in old:
            matches = [b for b in new if all(a[k] == b[k] for k in keys)]
            if len(matches) != 1:
                raise RuntimeError(f'{name}: missing or duplicate row {[a[k] for k in keys]}')
            for field in fields:
                if a[field] != '':
                    checks.append(check_close(name+':'+str([a[k] for k in keys])+':'+field, a[field], matches[0][field], atol=1e-14))
    for name, fields in {
        'adaptive_refinement_results.csv': ['DOFs', 'NNZ', 'CE', 'E_rel_M6', 'E_rel_M7', 'reference_sensitivity'],
        'uniform_graded_overlay_comparison.csv': ['DOFs', 'NNZ', 'CE', 'E_rel_M6', 'E_rel_M7'],
        'adaptive_matched_targets.csv': ['uniform_DOFs', 'graded_DOFs', 'adaptive_DOFs', 'uniform_NNZ', 'graded_NNZ', 'adaptive_NNZ'],
    }.items():
        old = rows(root/'results/adaptive_extension'/name)
        new = rows(extension/'results/adaptive_extension'/name)
        key = 'target' if name.startswith('adaptive_matched') else 'level'
        if len(old) != len(new):
            raise RuntimeError(f'{name}: changed row count')
        for a in old:
            b = next(b for b in new if a[key] == b[key])
            for field in fields:
                checks.append(check_close(name+':'+a[key]+':'+field, a[field], b[field], rtol=1e-8, atol=1e-13))
    from gate_g_air import load_frozen_geometry, final_mesh_fingerprint
    mesh_ids = []
    for base in (root, physical, extension):
        _, data, _ = load_frozen_geometry(base)
        mesh_ids.append((str(data['FINAL_MESH_ID']), final_mesh_fingerprint(data)))
    if len(set(mesh_ids)) != 1:
        raise RuntimeError(f'frozen mesh identity changed: {mesh_ids}')
    return checks, mesh_ids[0]


def run_full(root):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = root/'reproduction_runs'/stamp
    directory.mkdir(parents=True)
    report = dict(status='RUNNING', output_directory=str(directory.relative_to(root)), stages={})
    started = time.perf_counter()
    manifest = snapshot(root, directory/'backup')
    (directory/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
    def save():
        report['runtime_s'] = time.perf_counter()-started
        (directory/'reproduction_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        preflight = resource_preflight(root)
        report['resource_preflight'] = preflight
        (directory/'resource_preflight.json').write_text(json.dumps(preflight, indent=2))
        if preflight['decision'] != 'FEASIBLE':
            raise RuntimeError('M7 resource preflight failed; no heavy solves started')
        physical, extension = directory/'physical', directory/'overlay_adaptive'
        seed_workspace(root, physical)
        # Delete generated outputs: file existence after a run cannot masquerade as reproduction.
        for p in (physical/'tables').glob('*.csv'):
            if p.name not in {'graded_convergence.csv', 'uniform_vs_graded.csv', 'efficiency_targets.csv',
                              'matched_work_comparison.csv', 'graded_reference_geometry_audit.csv'}:
                p.unlink()
        for name in ('gate_A_verification.txt', 'gate_B_checks.txt', 'gate_C_domain.txt', 'gate_D_uniform.txt',
                     'gate_F_capacitance.txt', 'gate_G_air.txt', 'gate_H_impedance.txt'):
            (physical/'logs'/name).unlink(missing_ok=True)
        (physical/'data/processed/final_engineering_mesh.npz').unlink()
        report['stages']['physical_A_D_F_H_runtime_s'] = run_command(
            [sys.executable, 'code/run_all.py', '--worker', 'physical'], physical, directory/'physical_full.log')
        save()
        seed_workspace(root, extension)
        shutil.rmtree(extension/'results/adaptive_extension')
        shutil.rmtree(extension/'figures/adaptive_extension')
        report['stages']['overlay_E_adaptive_runtime_s'] = run_command(
            [sys.executable, 'code/run_all.py', '--worker', 'overlay'], extension, directory/'overlay_adaptive_full.log')
        checks, mesh_id = compare_outputs(root, physical, extension)
        with (directory/'numerical_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checks[0])); writer.writeheader();writer.writerows(checks)
        report.update(status='PASS', numerical_checks_passed=len(checks), FINAL_MESH_ID=mesh_id[0], FINAL_MESH_SHA256=mesh_id[1])
    except Exception as error:
        report.update(status='FAIL', error=str(error))
        raise
    finally:
        changed = [name for name, value in manifest.items() if not (root/name).exists() or digest(root/name) != value]
        report['original_protected_files_unchanged'] = not changed
        report['changed_original_files'] = changed
        if changed:
            report['status'] = 'FAIL'
        save()
        print(json.dumps(report, indent=2), flush=True)
        if changed:
            raise RuntimeError(f'frozen source files changed: {changed}')
    return report


def regenerate_saved_figures(root):
    """Quick mode validates saved results and regenerates plots, with no M6/M7 solves."""
    directory = root/'reproduction_runs'/('quick_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    directory.mkdir(parents=True)
    work = directory/'saved_analysis'
    seed_workspace(root, work)
    shutil.rmtree(work/'figures/adaptive_extension')
    elapsed = run_command([sys.executable, 'code/run_all.py', '--worker', 'analysis'],
                          work, directory/'quick_analysis.log')
    summary = dict(status='PASS', mode='saved-data-analysis-only', runtime_s=elapsed,
                   output=str(directory.relative_to(root)), no_reference_solves=True)
    (directory/'quick_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
