"""Retain reproducibility evidence without duplicate workspaces or frozen mesh copies."""
from pathlib import Path
import csv
import hashlib
import json
import shutil


def publish(root, source):
    summary=json.loads((source/'reproduction_summary.json').read_text())
    if summary['status']!='PASS' or not summary['original_protected_files_unchanged']:
        raise RuntimeError('only a completed, frozen-input-preserving run may be published')
    out=root/'reproduction'/source.name
    out.mkdir(parents=True,exist_ok=False)
    for name in ['reproduction_summary.json','resource_preflight.json','numerical_comparison.csv','physical_full.log','overlay_adaptive_full.log']:
        shutil.copy2(source/name,out/name)
    historical={'graded_convergence.csv','uniform_vs_graded.csv','efficiency_targets.csv','matched_work_comparison.csv','graded_reference_geometry_audit.csv'}
    table=out/'physical/tables';table.mkdir(parents=True)
    for p in (source/'physical/tables').glob('*.csv'):
        if p.name not in historical:shutil.copy2(p,table/p.name)
    logdir=out/'physical/logs';logdir.mkdir(parents=True)
    for p in (source/'physical/logs').glob('*.txt'):
        if p.name.startswith(('gate_A','gate_B','gate_C','gate_D','gate_F','gate_G','gate_H','M7_preflight','M8_preflight')):shutil.copy2(p,logdir/p.name)
    shutil.copytree(source/'physical/results/ae_revision',out/'physical/results/ae_revision')
    snapshot=source/'physical/data/processed/baseline_fields.npz'
    if snapshot.exists():
        shutil.copy2(snapshot,root/'data/processed/baseline_fields.npz')
    shutil.copytree(source/'overlay_adaptive/results/adaptive_extension',out/'overlay_adaptive/results/adaptive_extension')
    shutil.copytree(source/'overlay_adaptive/results/ae_revision',out/'overlay_adaptive/results/ae_revision')
    shutil.copy2(source/'overlay_adaptive/logs/graded_overlay_routing.json',out/'graded_overlay_routing.json')
    (out/'README.md').write_text('''# Completed isolated reproduction

The original numerical CSVs, report and figures in the repository remain frozen.
The physical and overlay directories here contain fresh solve outputs and timing samples.
Historical graded centroid estimates were not regenerated or certified; the fresh comparison uses triangle overlays.
Frozen mesh identity was checked after independent regeneration. Its duplicate NPZ is omitted here;
the authoritative artifact remains `data/processed/final_engineering_mesh.npz` at repository root.
Code/config copies, copied report assets and temporary workspace backups are omitted.
`physical_full.log` and `overlay_adaptive_full.log` are unedited command output.
Floating-point values are compared with explicit tolerances; timing equality is not required.
''')
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (out/'file_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(files)
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_directory',type=Path);args=p.parse_args()
    print(publish(Path(__file__).resolve().parents[1],args.run_directory.resolve()))
