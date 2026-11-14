"""Analytic EFP reference at arbitrary positive coupling, with resolution checks.

Solves the manuscript's Chebyshev form of the boundary-response integral
equation. Only alpha and background sound speed are passed to hydro.
The action, arctic curve and closing time are comparison data, never mesh
or internal-boundary constraints. Units R=rho0=m=hbar=1.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def analytic_basis(z, modes):
    z = np.asarray(z, dtype=complex)[..., None]
    degree = 2*np.arange(modes)+1
    root = np.sqrt(z-1)*np.sqrt(z+1)
    f = (1/(z+root))**degree
    return f, -degree*f/root, degree**2*f/root**2+degree*z*f/root**3


def solve_at(c, q, modes, eta):
    theta = (np.arange(modes)+.5)*np.pi/(2*modes)
    degree = 2*np.arange(modes)+1
    local = np.sin(theta[:, None]*degree)/np.sin(theta[:, None])*degree
    shifted = analytic_basis(np.cos(theta)+1j*c/q, modes)[1].real
    coefficients = np.linalg.solve(local-shifted, np.ones(modes))
    low = analytic_basis(1j*eta, modes)
    high = analytic_basis(1j*(eta+c/q), modes)
    intercept = 1+(high[1].real-low[1].real)@coefficients
    tau = (low[2].imag-high[2].imag)@coefficients/q
    x = intercept-q*eta*tau
    x[0] = 0.
    return dict(q=float(q), alpha=float(q*(degree@coefficients)**2/(4*np.pi)),
                action=float(q*q*coefficients[0]/2), tau_c=float(tau[0])), dict(
                x=np.r_[x, 1.], tau=np.r_[tau, 0.], coefficients=coefficients)


def reference(gamma, eos=None):
    if np.isinf(gamma):
        s = np.linspace(0., 1., 2001)
        return dict(gamma='inf', q=np.pi, cs0=np.pi, alpha=.25,
                    action=np.pi**2/2, tau_c=1/np.pi, reference_check=0.), dict(
                    x=np.maximum(1-s**(2/3), 0)**1.5, tau=s/np.pi)
    background = eos.background()
    q = background['q']
    modes = max(96, int(np.ceil(14*q/gamma/32))*32)
    eta = np.r_[0., np.geomspace(1e-6, 1e4, 2000)]
    coarse, first = solve_at(gamma, q, modes, eta)
    fine, curve = solve_at(gamma, q, 2*modes, eta)
    errors = {key:abs(fine[key]/coarse[key]-1) for key in ('alpha','action','tau_c')}
    errors['curve_x_absolute'] = float(np.max(abs(curve['x']-first['x'])))
    errors['curve_tau_relative'] = float(np.max(abs(curve['tau']-first['tau']))/fine['tau_c'])
    if max(errors.values()) > 2e-8:
        raise ValueError(f'Analytic reference has not converged at gamma={gamma}: {errors}')
    fine.update(gamma=float(gamma), cs0=float(background['cs0']),
                reference_check=errors, modes=2*modes, background=background)
    return fine, curve


def curve_at_time(curve, tau):
    """Interpolate comparison data after removing duplicate endpoint times."""
    t, ids = np.unique(curve['tau'], return_index=True)
    return np.where(np.asarray(tau) <= t[-1],
                    PchipInterpolator(t, curve['x'][ids])(np.minimum(tau, t[-1])), 0.)
