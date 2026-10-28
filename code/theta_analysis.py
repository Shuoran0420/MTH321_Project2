"""Analyze a separately reproduced marking-parameter study, without retuning theta."""
from pathlib import Path
import csv
import numpy as np
from ae_analysis import write


def run(root, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with source.open() as f:rows=list(csv.DictReader(f))
    summaries=[];fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for theta in [.3,.5,.7]:
        seq=[r for r in rows if float(r['theta'])==theta]
        if len(seq)!=8:raise RuntimeError('incomplete theta sequence')
        for r in seq:
            if not (r['geometry_material_conformity_PASS']=='True' and float(r['potential_min'])>=-1e-10 and float(r['potential_max'])<=1+1e-10 and float(r['normalized_residual'])<1e-10 and float(r['charge_balance_defect'])<1e-9 and float(r['symmetry_error'])<1e-10 and float(r['matrix_symmetry_error'])<1e-12 and float(r['delta_EQ'])<1e-9 and float(r['overlay_area_defect'])<1e-7):
                raise RuntimeError(f"theta validation failed: {theta}, {r['level']}")
        for ax,key,label in zip(axes,['DOFs','total_time'],['Degrees of freedom','Median cumulative algorithm time (s)']):
            ax.loglog([float(r[key]) for r in seq],[float(r['E_rel_M6']) for r in seq],'-',label=f'theta={theta}')
            for r in seq:ax.plot(float(r[key]),float(r['E_rel_M6']),'o',markerfacecolor='white' if float(r['reference_sensitivity'])>=.03 else ax.lines[-1].get_color(),color=ax.lines[-1].get_color(),markersize=4)
            ax.set(xlabel=label,ylabel='Relative energy error (M6 reference)');ax.grid(alpha=.2,which='both')
        last=seq[-1];summaries.append(dict(theta=theta,cycles=8,initial_DOFs=seq[0]['DOFs'],final_DOFs=last['DOFs'],final_error_M6=last['E_rel_M6'],reference_sensitivity=last['reference_sensitivity'],median_cumulative_time_s=last['total_time'],all_sanity_checks='PASS',scope='same M1-S start; not matched-work merely because cycle counts match'))
    axes[0].legend();fig.suptitle('Marking sensitivity; theta=0.5 remains primary; open = reference-sensitive')
    out=root/'results/ae_revision';out.mkdir(parents=True,exist_ok=True);write(out/'theta_sensitivity_summary.csv',summaries)
    dest=root/'figures/ae_revision';dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'theta_sensitivity.pdf');plt.close(fig)
    return summaries

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);args=p.parse_args();print(run(Path(__file__).resolve().parents[1],args.source))
