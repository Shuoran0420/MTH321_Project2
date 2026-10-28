"""Regenerate report tables from saved audited CSVs; never edits report prose or ICS."""
from pathlib import Path
import csv

def run(root):
    r=Path(root);d=r/'report'
    def rows(name):
     with (r/name).open() as f:return list(csv.DictReader(f))
    def table(name,header,data,cols):
     lines=[r'\begin{tabular}{@{}'+cols+r'@{}}',r'\toprule',header+r'\\',r'\midrule']
     lines += [' & '.join(x)+r'\\' for x in data];lines +=[r'\bottomrule',r'\end{tabular}'];(d/'tables'/name).write_text('\n'.join(lines)+'\n')
    L=rows('results/ae_revision/layered_interface_verification.csv');table('ae_layered.tex',r'$N$ & Fitted? & DOFs & $\|u_h-u_*\|_\infty$ (nodes) & $\|q_h-q_*\|_{L^2}$',[[x['n'],'yes' if x['interface_fitted']=='True' else 'no',x['DOFs'],f"{float(x['nodal_error_inf']):.2e}",f"{float(x['flux_L2_error']):.2e}"] for x in L], 'lcrrr')
    F=rows('results/ae_revision/uniform_fit_robustness.csv');sel=[x for x in F if x['window']=='M0-M2' or (x['window']=='M0-M3' and x['reference']=='R2')];table('ae_uniform_fit.tex',r'Window & Reference & Points & Fitted $h$ slope',[[x['window'],x['reference'],x['points'],f"{float(x['slope']):.6f}"] for x in sel],'llrr')
    R=rows('results/ae_revision/adaptive_empirical_rates.csv');table('ae_adaptive_rates.tex',r'Window & Reference & DOF slope & $R^2$',[[x['window'],x['reference'],f"{float(x['signed_DOF_slope']):.6f}",f"{float(x['R_squared']):.4f}"] for x in R],'llrr')
    B=rows('results/ae_revision/measured_target_brackets.csv'); table('ae_target_brackets.tex',r'Target & Method & Measured pair & DOFs (pair) & $E_{\rm rel}$ (pair) & max $S_{\rm ref}$ (\%)',[[a['target'],a['method'].replace('adaptive','adaptive B' if a['level'].startswith('B') else 'adaptive A'),a['level']+' / '+b['level'],a['DOFs']+' / '+b['DOFs'],f"{float(a['measured_error']):.4f} / {float(b['measured_error']):.4f}",f"{100*max(float(a['reference_sensitivity']),float(b['reference_sensitivity'])):.3f}"] for a,b in zip(B[::2],B[1::2])],'lllllr')
    T=rows('results/ae_revision/timing_median_iqr.csv');table('ae_timing.tex',r'Mesh & Repetitions & Median total (ms) & IQR (ms)',[[x['level'],x['repetitions'],f"{1e3*float(x['median_total_time_s']):.3f}",f"{1e3*float(x['IQR_s']):.3f}"] for x in T if x['level'] in ['M1','M2','G0','G1','B1','B2','A5','A7']],'lrrr')
    C=rows('tables/capacitance_asymptotic_diagnostics.csv');table('ae_richardson.tex',r'Triple & $p_C$ & $C_{\rm ext}$ (pF/m) & Finest-to-ext (\%) & Limit change (\%)',[[x['triple'],f"{float(x['p_C']):.5f}",f"{1e12*float(x['C_extrapolated']):.6f}",f"{100*float(x['relative_finest_to_extrapolated']):.4f}",'--' if x['relative_change_from_previous_Cext']=='nan' else f"{100*float(x['relative_change_from_previous_Cext']):.6f}"] for x in C],'lrrrr')
    theta_file=r/'results/ae_revision/theta_sensitivity_summary.csv'
    if theta_file.exists():
        values=list(csv.DictReader(theta_file.open()))
        table('ae_theta.tex',r'$\theta$ & A7 DOFs & A7 $E_{\rm rel}$ & $S_{\rm ref}$ (\%) & Cumulative time (ms)',
              [[x['theta'],x['final_DOFs'],f"{float(x['final_error_M6']):.4f}",f"{100*float(x['reference_sensitivity']):.3f}",f"{1000*float(x['median_cumulative_time_s']):.1f}"] for x in values],'rrrrr')

if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
