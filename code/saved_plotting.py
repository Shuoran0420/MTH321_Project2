"""Regenerate curves/field visualizations from saved arrays and CSVs, without solving."""
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
from geometry import GeometryConfig


def numeric_rows(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            try:r[k]=float(v) if v else float('nan')
            except ValueError:pass
    return rows


def save_baseline(sol, root):
    np.savez_compressed(root/'data/processed/baseline_fields.npz',points=sol.geometry.points,
                        elements=sol.geometry.elements,phi=sol.phi,gradients=sol.element_gradients,epsilon=sol.epsilon)


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from verification import plot_convergence
    from domain_study import plot_domain_figures
    from uniform_convergence import plot_outputs
    from final_capacitance import _plot
    from impedance import _plot_outputs
    from plotting import write_baseline_figures
    from ae_analysis import write
    cfg=GeometryConfig(W_over_H=8.0,Hair_over_H=8.0)
    def solution(d):
        geom=SimpleNamespace(points=d['points'],elements=d['elements'],config=cfg)
        return SimpleNamespace(geometry=geom,phi=d['phi'],element_gradients=d['gradients'],epsilon=d.get('epsilon'))
    coverage=[]
    mms=numeric_rows(root/'tables/mms_convergence.csv');plot_convergence([SimpleNamespace(**r) for r in mms],root/'figures/mms_convergence.png');coverage.append(dict(asset='MMS convergence',status='REGENERATED_FROM_CSV'))
    domain=numeric_rows(root/'tables/domain_truncation.csv');plot_domain_figures(root,domain);coverage.append(dict(asset='domain plots',status='REGENERATED_FROM_CSV_AND_GEOMETRY'))
    uniform=numeric_rows(root/'tables/uniform_convergence.csv');stability=numeric_rows(root/'tables/reference_stability.csv')
    solutions=[]
    for i in range(4):
        data=np.load(root/f'results/adaptive_extension/meshes/M{i}.npz');sol=solution(data)
        solutions.append(dict(mesh=SimpleNamespace(geom=sol.geometry),solution=sol,level=i))
    slope=np.polyfit(np.log([r['h_max_m'] for r in uniform[:3]]),np.log([r['relative_energy_error'] for r in uniform[:3]]),1)[0]
    plot_outputs(root,uniform,stability,solutions,float(slope),[0,1,2]);coverage.append(dict(asset='uniform/corner plots',status='REGENERATED_FROM_CSV_AND_SAVED_FIELDS'))
    selected=numeric_rows(root/'tables/final_capacitance_selection.csv');u=[r for r in selected if r['family']=='uniform'];g=[r for r in selected if r['family']=='graded'];ext=numeric_rows(root/'tables/capacitance_asymptotic_diagnostics.csv')
    _plot(root,u,g,'M7',ext[-1]['C_extrapolated'],True,ext);coverage.append(dict(asset='capacitance diagnostics',status='REGENERATED_FROM_CSV'))
    fem=numeric_rows(root/'tables/final_characteristic_impedance.csv')[0];bench=numeric_rows(root/'tables/engineering_benchmarks.csv');z=numeric_rows(root/'tables/Z0_convergence.csv');_plot_outputs(root,fem,bench,z);coverage.append(dict(asset='engineering curves',status='REGENERATED_FROM_CSV'))
    e=numeric_rows(root/'tables/epsilon_eff_convergence.csv');fig,ax=plt.subplots(figsize=(6.1,4),layout='constrained');ax.plot([r['DOFs'] for r in e],[r['epsilon_eff'] for r in e],'o-')
    for r in e:ax.annotate(r['mesh_id'],(r['DOFs'],r['epsilon_eff']),xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='Degrees of freedom',ylabel=r'$\varepsilon_{\mathrm{eff}}$',title='Matched-pair effective permittivity');ax.grid(alpha=.2);fig.savefig(root/'figures/epsilon_eff_convergence.png',dpi=180);plt.close(fig)
    baseline=root/'data/processed/baseline_fields.npz'
    if baseline.exists():
        write_baseline_figures(solution(np.load(baseline)),root);coverage.append(dict(asset='baseline potential/field/energy',status='REGENERATED_FROM_SAVED_P1_FIELDS'))
    else:
        coverage.append(dict(asset='baseline potential/field/energy',status='NOT_REGENERATED_MISSING_FIELD_SNAPSHOT_RUN_FULL_FIRST'))
    write(root/'results/ae_revision/quick_figure_coverage.csv',coverage)
    return coverage
