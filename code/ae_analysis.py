"""Evidence-only robustness analysis; no scientific source CSV is overwritten."""
from pathlib import Path
import csv
import json
import platform
import numpy as np


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fit(x,y):
    x,y=np.log(np.asarray(x,float)),np.log(np.asarray(y,float));p=np.polyfit(x,y,1)
    residual=np.sum((y-np.polyval(p,x))**2);total=np.sum((y-y.mean())**2)
    return float(p[0]),float(1-residual/total)


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'results/ae_revision';figs=root/'figures/ae_revision'
    out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    adaptive=read(root/'results/adaptive_extension/adaptive_refinement_results.csv')
    main=[r for r in adaptive if r['level'].startswith('A')]
    fitrows=[]
    for start,end in [(0,7),(3,7)]:
        data=main[start:end+1]
        for ref in ['M6','M7']:
            slope,r2=fit([r['DOFs'] for r in data],[r['E_rel_'+ref] for r in data])
            fitrows.append(dict(window=f'A{start}-A{end}',reference=ref,signed_DOF_slope=slope,decay_exponent=-slope,R_squared=r2,ideal_decay_exponent=.5))
    write(out/'adaptive_empirical_rates.csv',fitrows)
    fig,ax=plt.subplots(figsize=(6.5,4.2),layout='constrained')
    n=np.array([float(r['DOFs']) for r in main]);e=np.array([float(r['E_rel_M6']) for r in main]);ax.loglog(n,e,'o-',label='A0-A7, M6 reference');ax.loglog(n,e[0]*(n/n[0])**(-.5),'k--',label=r'2D smooth P1 comparison: $N^{-1/2}$')
    for r in main:ax.annotate(r['level'],(float(r['DOFs']),float(r['E_rel_M6'])),xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Degrees of freedom N',ylabel='Relative energy error',title='Residual adaptivity: measured DOF decay');ax.legend();ax.grid(alpha=.2,which='both');fig.savefig(figs/'adaptive_empirical_rates.pdf');plt.close(fig)
    stability=read(root/'tables/reference_stability.csv'); uniform=read(root/'tables/uniform_convergence.csv')
    fits=[]
    for window in [(0,1),(0,2),(1,2),(0,3)]:
        data=[r for r in stability if r['analysis_mesh_id'] in [f'M{i}' for i in range(window[0],window[1]+1)]]
        for ref in ['R0','R1','R2']:
            slope,r2=fit([r['h_max_m'] for r in data],[r['error_using_'+ref] for r in data])
            fits.append(dict(window=f'M{window[0]}-M{window[1]}',reference=ref,points=len(data),slope=slope,R_squared=r2,note='sensitivity analysis; original fit window unchanged'))
    write(out/'uniform_fit_robustness.csv',fits)
    comparisons=read(root/'results/adaptive_extension/three_way_comparison.csv');targetrows=read(root/'results/adaptive_extension/adaptive_matched_targets.csv');endpoints=[]
    for r in targetrows:
        for method in ['uniform','graded','adaptive']:
            for level in r[method+'_bracket'].split('/'):
                x=next(x for x in comparisons if x['level']==level)
                endpoints.append(dict(target=r['target'],method=method,level=level,DOFs=x['DOFs'],NNZ=x['NNZ'],measured_error=x['E_rel_M6'],reference_sensitivity=x['reference_sensitivity'],formal_point=float(x['reference_sensitivity'])<.03,interpolated_DOFs=r[method+'_DOFs'],path=r['adaptive_path'] if method=='adaptive' else method,interpolation_weight=r[method+'_weight']))
    write(out/'measured_target_brackets.csv',endpoints)
    timing=[]
    for filename,key in [('adaptive_timing_samples.csv','adaptive'),('comparison_timing_samples.csv','fixed')]:
        samples=read(root/'results/adaptive_extension'/filename);groups={}
        for r in samples:
            level=r['path']+r['cycle'] if key=='adaptive' else r['level'];groups.setdefault(level,[]).append(float(r['total_time']))
        for level,vals in groups.items():
            q1,q3=np.quantile(vals,[.25,.75]);timing.append(dict(level=level,repetitions=len(vals),median_total_time_s=float(np.median(vals)),Q1_s=float(q1),Q3_s=float(q3),IQR_s=float(q3-q1),accounting='cumulative to current solution; outgoing refinement charged next cycle' if key=='adaptive' else 'one-shot mesh+assembly+solve+extraction',hardware_scope='original saved samples; see original experiment metadata, not retimed by this analysis'))
    write(out/'timing_median_iqr.csv',timing)
    cap=read(root/'tables/final_capacitance_convergence.csv');diag=read(root/'tables/capacitance_asymptotic_diagnostics.csv');domain=read(root/'tables/domain_truncation.csv');d8=next(r for r in domain if float(r['W_over_H'])==8 and r['resolution_id']=='R0');d12=next(r for r in domain if float(r['W_over_H'])==12 and r['resolution_id']=='R0')
    final=read(root/'tables/final_dielectric_capacitance.csv')[0]
    summary=dict(adaptive_fits=fitrows,uniform_fit_windows=fits,final_computed_C=final['CE'],domain_8H_production_C=d8['CE_F_per_m'],domain_12H_production_C=d12['CE_F_per_m'],domain_indicator=d8['relative_to_C12'],domain_scope='matched-core production domain sequence, not a bound for final M7',richardson_diagnostics=diag,spatial_indicator=final['spatial_indicator'],ideal_adaptive_efficiency_demonstrated=False,python=platform.python_version())
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Larger report-only mesh view, using saved arrays; no remeshing.
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    for col,level in enumerate(['M2','G2']):
        data=np.load(root/f'results/adaptive_extension/meshes/{level}.npz');p=data['points']*1e3;t=data['elements']
        for row,window in enumerate([(-2.5,2.5,.8,2.5),(1.35,1.65,1.45,1.8)]):
            ax=axes[row,col];visible=(p[t,0].min(axis=1)<window[1])&(p[t,0].max(axis=1)>window[0])&(p[t,1].min(axis=1)<window[3])&(p[t,1].max(axis=1)>window[2]);ax.triplot(p[:,0],p[:,1],t[visible],linewidth=.4,color='#245b79');ax.plot([-1.5,1.5,1.5,-1.5,-1.5],[1.6,1.6,1.635,1.635,1.6],color='firebrick',linewidth=1);ax.set(xlim=window[:2],ylim=window[2:],xlabel='x (mm)',ylabel='y (mm)',aspect='equal',title=f'{level}: '+('near trace' if row==0 else 'right lower/upper corners'))
    fig.savefig(figs/'uniform_graded_corner_mesh.pdf');plt.close(fig)
    theta_file=root/"results/adaptive_extension/theta_sensitivity.csv"
    if not theta_file.exists():
        theta_file=root/"results/ae_revision/theta_sensitivity_raw.csv"
    if theta_file.exists():
        from theta_analysis import run as analyze_theta
        analyze_theta(root,theta_file)
    return summary

if __name__=='__main__':
    print(json.dumps(run(Path(__file__).resolve().parents[1]),indent=2))
