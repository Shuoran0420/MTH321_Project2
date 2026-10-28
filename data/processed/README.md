# Frozen mesh

The large NPZ is distributed in a lossless ZIP for GitHub browser upload. From the repository root run:

```bash
python -m zipfile -e data/processed/final_engineering_mesh.npz.zip data/processed
```

Extract before `pytest` or `run_all.py`. The ZIP contains the exact original file, not a regenerated mesh.

Original NPZ SHA-256: `44cf7ac522a04b38945f2d3b4f2992b248bebf40d666b4d7bd9940def5991d33`

The other file, `baseline_fields.npz`, is a small saved field snapshot.
