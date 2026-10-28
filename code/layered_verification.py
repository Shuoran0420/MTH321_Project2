"""Problem Pack layered analytic benchmark, using production P1 assembly."""
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
from skfem import MeshTri
from scipy.sparse.linalg import spsolve, eigsh
from fem_solver import assemble_stiffness
from materials import MaterialConfig, EPSILON_0


def exact(x):
    return np.where(x <= .5, 20*x/11, (2*x+9)/11)


def solve_layered(n):
    mesh = MeshTri.init_tensor(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    p, t = mesh.p.T, mesh.t.T
    regions = np.where(p[t].mean(axis=1)[:, 0] < .5, 'substrate', 'air')
    geom = SimpleNamespace(points=p, elements=t, element_regions=regions)
    matrix, areas, basis, eps, _, _ = assemble_stiffness(geom, MaterialConfig(epsilon_r_substrate=1.0, epsilon_r_air=10.0))
    matrix = matrix / EPSILON_0
    boundary = mesh.boundary_nodes(); free = np.setdiff1d(np.arange(len(p)), boundary)
    u = exact(p[:, 0]); rhs = -matrix[free][:, boundary]@u[boundary]
    reduced = matrix[free][:, free].tocsr(); u[free] = spsolve(reduced, rhs)
    gradient = np.einsum('eik,ei->ek', basis, u[t]); a = eps/EPSILON_0
    edges = mesh.facets.T; mid = p[edges].mean(axis=1)
    interface = np.isclose(mid[:, 0], .5, atol=1e-14) & (mesh.f2t[1] >= 0)
    left, right = mesh.f2t[:, interface]
    jump = a[left]*gradient[left, 0]-a[right]*gradient[right, 0]
    flux = -a*gradient[:, 0]
    # Three-point degree-two triangle quadrature; exact piecewise affine u* on fitted mesh.
    bary = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    xq = np.einsum('qi,eid->eqd', bary, p[t]); uq=np.einsum('qi,ei->eq',bary,u[t])
    l2 = np.sqrt(np.sum(areas*np.mean((uq-exact(xq[:,:,0]))**2,axis=1)))
    flux_error = np.sqrt(np.sum(areas*(flux+20/11)**2))
    node_error = np.max(abs(u-exact(p[:, 0])))
    asym= reduced-reduced.T
    row=dict(n=n, interface_fitted=n%2==0, elements=len(t), DOFs=len(p),
             nodal_error_inf=float(node_error), L2_error=float(l2),
             flux_L2_error=float(flux_error), interface_flux_jump_max=float(np.max(abs(jump))) if len(jump) else '',
             potential_jump=0.0 if n%2==0 else '', min_u=float(u.min()), max_u=float(u.max()),
             residual_inf=float(np.max(abs(reduced@u[free]-rhs))), matrix_symmetry=float(np.sqrt(asym.multiply(asym).sum())),
             lambda_min=float(eigsh(reduced,k=1,which='SA',return_eigenvectors=False,tol=1e-9)[0]),
             treatment='interface_fitted_elementwise_a' if n%2==0 else 'deliberately_unresolved_centroid_tags_not_reference')
    return row, p, u, flux


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'results/ae_revision';out.mkdir(parents=True,exist_ok=True)
    rows=[]; fitted=None; unresolved=None
    for n in [4,8,16,32,7,15,31]:
        result=solve_layered(n);rows.append(result[0])
        if n==32:fitted=result
        if n==31:unresolved=result
    with (out/'layered_interface_verification.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aligned=[r for r in rows if r['interface_fitted']]
    passed=all(r['nodal_error_inf']<1e-11 and r['flux_L2_error']<1e-10 and r['interface_flux_jump_max']<1e-10 and r['lambda_min']>0 for r in aligned)
    if not passed:raise RuntimeError('layered-interface analytic verification failed')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    x=np.linspace(0,1,300);axes[0].plot(x,exact(x),'k--',label='analytic piecewise affine')
    for label,result in [('fitted N=32',fitted),('unresolved N=31 (control)',unresolved)]:
        row,p,u,flux=result; mesh=MeshTri.init_tensor(np.linspace(0,1,row['n']+1),np.linspace(0,1,row['n']+1));t=mesh.t.T
        mask=np.isclose(p[:,1],0.5,atol=.5/row['n']); ids=np.flatnonzero(mask);ids=ids[np.argsort(p[ids,0])];axes[0].plot(p[ids,0],u[ids],'.',label=label)
        center=p[t].mean(axis=1);band=abs(center[:,1]-.5)<1/row['n'];axes[1].scatter(center[band,0],flux[band],s=10,label=label)
    axes[1].axhline(-20/11,color='k',linestyle='--',label='analytic flux -20/11')
    for ax in axes:ax.axvline(.5,color='gray',linestyle=':');ax.set_xlabel('x (dimensionless)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Potential u (dimensionless)');axes[1].set_ylabel('Normal flux -a du/dx (dimensionless)')
    dest=root/'figures/ae_revision';dest.mkdir(exist_ok=True,parents=True);fig.savefig(dest/'layered_interface_verification.pdf');plt.close(fig)
    return {'status':'PASS','rows':len(rows),'maximum_fitted_nodal_error':max(r['nodal_error_inf'] for r in aligned)}

if __name__=='__main__':
    print(run(Path(__file__).resolve().parents[1]))
