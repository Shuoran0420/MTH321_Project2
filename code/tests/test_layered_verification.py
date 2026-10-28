import numpy as np
from layered_verification import exact, solve_layered


def test_layered_analytic_interface_data():
    assert np.isclose(exact(.5),10/11)
    assert np.isclose(exact(0),0) and np.isclose(exact(1),1)
    assert np.isclose(1*20/11,10*2/11)


def test_production_assembly_reproduces_layered_solution_and_flux():
    for n in [4,8,16]:
        r,_,_,_=solve_layered(n)
        assert r['interface_fitted'] and r['nodal_error_inf']<1e-12
        assert r['L2_error']<1e-12 and r['flux_L2_error']<1e-11
        assert r['interface_flux_jump_max']<1e-11 and r['potential_jump']==0
        assert r['lambda_min']>0 and r['matrix_symmetry']<1e-12
        assert r['residual_inf']<1e-12 and 0<=r['min_u']<=r['max_u']<=1


def test_unresolved_interface_control_is_not_an_exact_solution():
    r,_,_,_=solve_layered(7)
    assert not r['interface_fitted']
    assert r['nodal_error_inf']>1e-3
    assert r['flux_L2_error']>1e-3
