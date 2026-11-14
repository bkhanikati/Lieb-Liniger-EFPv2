"""Independent numerical checks for the parameterized production program.

Run after the EOS cache has been built. Checks thermodynamic consistency,
scaling, the matched high-density tail, analytic TG radial modes, spline
transfer and conserved boundary traces. No large hydro solve is required.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from efp_production_core import TonksEOS
from efp_production_eos import ProductionEOS, validate_table
from efp_production_fem import (initial_axes,knots,constraints,insert_coefficients,
                                evaluate,basis,coefficient_derivative)
from efp_production_diagnostics import radial_basis
from efp_production import refine_interface,load_fields
from efp_production_reference import reference,solve_at


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'production-output')
    args=parser.parse_args();out=args.output
    eos_checks=[]
    for c in (10.,1.,.1,.01):
        eos=ProductionEOS(c,out/'eos_table.npz')
        rho=np.array([.003,.03,.1,.3,.8,1.2,3.,30.,300.])
        h=1e-4*rho
        derivative=(eos.potential(rho+h)-eos.potential(rho-h))/(2*h)
        error=float(np.max(abs(derivative-eos.derivative(rho))/np.maximum(abs(eos.derivative(rho)),1)))
        assert error<2e-7
        second=(eos.mu(rho+h)-eos.mu(rho-h))/(2*h)
        second_error=float(np.max(abs(second/eos.dmu(rho)-1)))
        assert second_error<2e-7
        assert np.all(eos.dmu(np.geomspace(1e-9,1e6,1000))>0)
        join=c*eos.join
        continuity=[float(abs(f(join*(1+1e-9))-f(join*(1-1e-9)))/max(abs(f(join)),1e-30))
                    for f in (eos.energy,eos.mu,eos.dmu)]
        assert max(continuity)<1e-7
        base=ProductionEOS(1.,out/'eos_table.npz')
        for method,power in (('energy',3),('mu',2),('dmu',1)):
            np.testing.assert_allclose(getattr(eos,method)(rho),c**power*getattr(base,method)(rho/c),rtol=2e-15)
        eos_checks.append(dict(gamma=c,potential_gradient_error=error,mu_derivative_error=second_error,
                               join_continuity=continuity,background=eos.background()))
    r=np.linspace(.002,.8,101)
    for n in range(4):
        np.testing.assert_allclose(radial_basis(TonksEOS(),r,n),
            np.pi**(2*n)*r**(2*n)/math.factorial(2*n+1),rtol=3e-13,atol=1e-15)
    # A field satisfying the outer/slit constraints is transferred exactly
    # under arbitrary nested insertion. This tests the discretization,
    # independently of whether a particular minimization converges.
    x,s=initial_axes(16,13.)
    tx,ts=knots(x,True),knots(s)
    fixed,full,cm,cw=constraints(tx,ts,13.,.25)
    old=dict(x_axis=x,s_axis=s,x_knots=tx,s_knots=ts,coefficients=full.reshape(len(ts)-3,len(tx)-3))
    nx=np.sort(np.r_[x,(x[:-1]+x[1:])[::2]/2]);ns=np.sort(np.r_[s,(s[:-1]+s[1:])[::2]/2])
    new=dict(x_axis=nx,s_axis=ns,x_knots=knots(nx,True),s_knots=knots(ns),coefficients=insert_coefficients(old,nx,ns))
    points=np.linspace(.001,12.99,101)
    errors=[float(np.max(abs(a-b))) for a,b in zip(evaluate(old,points,points),evaluate(new,points,points))]
    assert max(errors)<1e-9
    # Independent normalization check of the analytic outer dipole. Since
    # I=rho0^2*f(c/rho0), dI/drho0=2*f-gamma*f' at rho0=1. This must equal
    # 4*pi*alpha*cs0. It checks the generalized coupling factors rather
    # than merely increasing the integral-equation resolution.
    dipole_checks=[]
    for c in (10.,1.,.1,.01):
        eos=ProductionEOS(c,out/'eos_table.npz')
        ref,_=reference(c,eos)
        h=.001*c
        values={}
        for k in (-2,-1,1,2):
            interaction=c+k*h
            other=ProductionEOS(interaction,out/'eos_table.npz')
            q=other.background()['q']
            value,_=solve_at(interaction,q,ref['modes'],np.array([0.]))
            values[k]=value['action']
        derivative=(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*h)
        lhs=2*ref['action']-c*derivative
        rhs=4*np.pi*ref['alpha']*ref['cs0']
        error=float(lhs/rhs-1)
        assert abs(error)<2e-7
        dipole_checks.append(dict(gamma=c,density_derivative=lhs,dipole_expression=rhs,relative_error=error))
    completed=[]
    for folder in sorted(out.glob('gamma_*')):
        if not (folder/'config.json').exists():
            continue
        config=json.loads((folder/'config.json').read_text())
        path=folder/f"refine_{config['refinements']}.npz" if config['refinements'] else folder/'base.npz'
        if config.get('slit_refinements',0):
            path=folder/f"slit_{config['slit_refinements']}.npz"
        if not path.exists():
            continue
        f=load_fields(path);coarse=load_fields(folder/'coarse.npz')
        z=np.linspace(0,config['extent'],1001)
        traces=[]
        for field in (coarse,f):
            bx,_=basis(field['x_knots'],z);bs,_=basis(field['s_knots'],z)
            traces.append((bx@field['coefficients'][-1],bs@field['coefficients'][:,-1]))
        trace_error=max(float(np.max(abs(a-b))) for a,b in zip(*traces))
        assert trace_error<1e-11
        coefficients=coefficient_derivative(f['x_knots'])@f['coefficients'].T
        assert coefficients.min()>=-1e-10
        completed.append(dict(gamma=folder.name,outer_trace_error=trace_error,
                              density_coefficient_min=float(coefficients.min())))
    result=dict(eos_checks=eos_checks,radial_modes_pass=True,transfer_errors=errors,
                dipole_normalization_checks=dipole_checks,completed_fields=completed)
    (out/'production_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
