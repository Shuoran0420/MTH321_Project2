import csv
from pathlib import Path
import numpy as np
import pytest
from ae_analysis import fit
from adaptive_analysis import interpolate_target


def test_power_law_fit_and_no_false_ideal_rate():
    n=np.array([100,200,400,800]);e=.7*(n/100)**(-.25)
    slope,r2=fit(n,e)
    assert np.isclose(slope,-.25) and np.isclose(r2,1)
    assert -slope<.5


def test_target_interpolation_uses_real_bracket_and_never_extrapolates():
    rows=[dict(level='A0',E_rel_M6=.4,DOFs=100,NNZ=500,total_time=.01,reference_sensitivity=.01),
          dict(level='A1',E_rel_M6=.2,DOFs=400,NNZ=2000,total_time=.04,reference_sensitivity=.04)]
    r=interpolate_target(rows,.3)
    assert r['bracket']=='A0/A1' and 100<r['DOFs']<400
    assert r['sensitivity']==.04
    assert interpolate_target(rows,.5) is None and interpolate_target(rows,.1) is None


def test_saved_adaptive_rates_and_measured_endpoints():
    root=Path(__file__).resolve().parents[2]
    with (root/'results/ae_revision/adaptive_empirical_rates.csv').open() as f:rates=list(csv.DictReader(f))
    assert len(rates)==4
    assert all(0<float(r['decay_exponent'])<.5 for r in rates)
    with (root/'results/ae_revision/measured_target_brackets.csv').open() as f:brackets=list(csv.DictReader(f))
    assert len(brackets)==18
    for a,b in zip(brackets[::2],brackets[1::2]):
        assert min(float(a['measured_error']),float(b['measured_error']))<=float(a['target'])<=max(float(a['measured_error']),float(b['measured_error']))
