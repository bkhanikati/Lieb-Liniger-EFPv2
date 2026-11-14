"""Convex hydrodynamic action and sparse Newton optimizer.

The verified optimizer is shared mathematically with the preceding hydro
calculations. This standalone production module contains no historical
study driver or LL EOS: the latter lives in efp_production_eos.py.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve


class TonksEOS:
    """Exact impenetrable-boson EOS: epsilon=pi**2*rho**3/6.

    potential(), derivative(), second() return V, V', V'' in acoustic units.
    The subtraction of the background tangent gives V(1)=V'(1)=0, while
    V(0)>0: an empty region still contributes to the instanton action.
    """
    sound = np.pi
    name = "Tonks-Girardeau"
    metadata = {}

    def potential(self, r):
        return (r - 1) ** 2 * (r + 2) / 6

    def derivative(self, r):
        return (r * r - 1) / 2

    def second(self, r):
        return r


def geometric_tail(length, first_width, count):
    """Geometric cell widths, matching the cell next to the inner interval."""
    powers = np.arange(count)
    ratio = brentq(lambda q: first_width*np.sum(q**powers)-length, .01, 10)
    edges = np.cumsum(first_width*ratio**powers)
    edges[-1] = length
    return edges


def exterior_angular_factor(a,b):
    """Integral max(cos(theta)^2/a^2,sin(theta)^2/b^2) over the full circle."""
    split=np.arctan2(b,a)
    factor=4*((split/2+np.sin(2*split)/4)/a**2
              +((np.pi/2-split)/2+np.sin(2*split)/4)/b**2)
    z,w=np.polynomial.legendre.leggauss(48)
    first=(z+1)*split/2
    second=split+(z+1)*(np.pi/2-split)/2
    direct=4*(split/2*np.sum(w*np.cos(first)**2)/a**2
              +(np.pi/2-split)/2*np.sum(w*np.sin(second)**2)/b**2)
    assert abs(factor-direct)<1e-13
    return factor


def action_terms(rho, k, eos):
    """Acoustic action density; k=u_s, so physical current is j=-cs*k.

    Quadrature nodes have strictly positive rho during barrier solves.
    Vacuum is approached as rho,k tend to zero; its finite V(0) is retained.
    """
    return k*k/(2*rho)+eos.potential(rho)


def minimize(gx, gs, weights, fixed, full, cmat, cweights, eos, upper, maxiter=100,
             barriers=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9),
             permc_spec='MMD_AT_PLUS_A', gap_atol=1e-9):
    """Sparse feasible Newton minimization with decreasing log barriers.

    gx,gs map full displacement coefficients to rho,u_s at Gauss nodes.
    fixed marks Dirichlet coefficients, eliminated before optimization.
    cmat@full >= 0 is the sufficient, global density-positivity condition;
    cweights normalizes its barrier by density-basis support areas.
    upper contains bounds implied by spatial monotonicity and boundary
    values. These bounds certify a convex objective gap; they are not extra
    active constraints in the Newton equations.

    Returns the full field, iteration records, and a gap bound for this
    finite-quadrature problem. The bound does not cover mesh or box error.
    gap_atol is an absolute allowance in the quarter-box objective. It can
    be increased explicitly for a lower-barrier sensitivity check when
    cancellation in the gradient reaches its floating-point floor. The
    actually measured gap is always returned, rather than replaced by this
    stopping tolerance.
    """
    free = ~fixed
    bx, bs = gx[:, fixed]@full[fixed], gs[:, fixed]@full[fixed]
    dx, dt = gx[:, free].tocsr(), gs[:, free].tocsr()
    c0 = cmat[:, fixed]@full[fixed]
    c = cmat[:, free].tocsr()
    variable = np.diff(c.indptr) > 0
    c, c0, cw = c[variable], c0[variable], cweights[variable]
    u = full[free].copy()
    upper = upper[free]
    assert np.min(c@u+c0) > 0
    history = []
    for barrier in barriers:
        print(f'Newton stage: barrier={barrier:.0e}, free coefficients={len(u)}', flush=True)
        converged = False
        for it in range(maxiter):
            r, k, slack = dx@u+bx, dt@u+bs, c@u+c0
            if min(r.min(), slack.min()) <= 0:
                raise RuntimeError('Lost strict feasibility')
            value = weights@action_terms(r, k, eos)-barrier*np.sum(cw*np.log(slack))
            vr = eos.derivative(r)-k*k/(2*r*r)
            vk = k/r
            gradient = dx.T@(weights*vr)+dt.T@(weights*vk)-barrier*c.T@(cw/slack)
            # Convex first-order lower bound using lambda=barrier*cw/slack
            # and the implied box bounds 0<=u_i<=upper_i. This also controls
            # Newton error; barrier*cw.sum alone assumes exact stationarity.
            gap_bound = float(barrier*cw.sum()+np.maximum(gradient,0)@u
                              +np.maximum(-gradient,0)@np.maximum(upper-u,0))
            a = eos.second(r)+k*k/r**3
            cross, d = -k/r**2, 1/r
            hessian = (dx.T@(dx.multiply((weights*a)[:, None])+dt.multiply((weights*cross)[:, None]))
                       +dt.T@(dx.multiply((weights*cross)[:, None])+dt.multiply((weights*d)[:, None]))
                       +c.T@c.multiply((barrier*cw/slack**2)[:, None])).tocsc()
            scale = 1/np.sqrt(hessian.diagonal())
            # Diagonal equilibration is essential near the vacuum, where
            # terms proportional to 1/rho make the raw Hessian ill-scaled.
            scaling = sparse.diags(scale)
            # A symmetric minimum-degree ordering reduces fill in this
            # symmetric local-stencil Hessian. The linear equations are
            # identical to those with SciPy's general default COLAMD order.
            step = scale*spsolve(scaling@hessian@scaling, -scale*gradient,
                                permc_spec=permc_spec)
            decrement = float(-gradient@step)
            # Keep the historical JSON key for compatibility. In the free-
            # boundary solver this action INCLUDES vacuum, despite "liquid".
            history.append(dict(barrier=barrier, iteration=it, quarter_liquid_action=float(weights@action_terms(r,k,eos)),
                                newton_decrement=decrement, gradient_inf=float(np.max(abs(gradient))),
                                quarter_liquid_gap_bound=gap_bound,
                                density_min=float(r.min()), density_max=float(r.max())))
            if it > 0 and it % 20 == 0:
                print(f'  iteration={it} quarter_action={history[-1]["quarter_liquid_action"]:.10f} '
                      f'decrement={decrement:.2e} gap_bound={gap_bound:.2e}', flush=True)
            if decrement < -1e-12 or not np.isfinite(decrement):
                raise RuntimeError('Newton direction failed')
            if decrement/2 < 1e-12 and (barrier>barriers[-1] or gap_bound<2*barrier*cw.sum()+gap_atol):
                converged = True
                break
            delta = c@step
            negative = delta < 0
            length = min(1., .995*np.min(-slack[negative]/delta[negative], initial=np.inf))
            slope = float(gradient@step)
            for _ in range(60):
                # Feasibility first, then Armijo decrease of the barrier
                # objective. Dirichlet data never change during this loop.
                trial = u+length*step
                rt, kt, st = dx@trial+bx, dt@trial+bs, c@trial+c0
                if min(rt.min(), st.min()) > 0:
                    trial_value = weights@action_terms(rt,kt,eos)-barrier*np.sum(cw*np.log(st))
                    rounding = 20*np.finfo(float).eps*max(abs(value),1) if decrement<1e-10 else 0
                    if trial_value <= value+.01*length*slope+rounding:
                        break
                length *= .5
            else:
                raise RuntimeError('Newton line search failed')
            u = trial
        print(f'barrier={barrier:.0e} steps={it+1} quarter_action={weights@action_terms(r,k,eos):.10f} '
              f'decrement={decrement:.2e} rho=[{r.min():.2e},{r.max():.3f}]', flush=True)
        if not converged:
            raise RuntimeError('Newton iteration cap reached')
    full[free] = u
    return full, history, dict(barrier=barrier, nominal_barrier_gap_at_exact_stationarity=float(barrier*cw.sum()),
                              sparse_ordering=permc_spec,
                              gap_absolute_tolerance=gap_atol,
                              quarter_gap_stop_threshold=float(2*barrier*cw.sum()+gap_atol),
                              quarter_liquid_gap_bound=gap_bound,
                              newton_decrement=decrement, gradient_inf=float(np.max(abs(gradient))),
                              free_unknowns=int(free.sum()), density_constraint_min=float((c@u+c0).min()))
