"""Coupling-independent Bethe EOS table and a controlled high-density tail.

Units m=hbar=1. Scaling reduces every finite c to one table at c=1:
rho=c*n, mu=c**2*M(n), epsilon=c**3*E(n). Cubic Hermite interpolation
uses both mu and dmu/drho=cs**2/rho from Bethe dressing. Integrating that
same interpolant keeps the potential, gradient and Hessian consistent.

For n above the table, use the weak-coupling energy through gamma**(5/2):
e(g)=g-4*g**1.5/(3*pi)+(1/6-1/pi**2)*g**2+A*g**2.5,
A=(3*zeta(3)/8-1/2)/pi**3; epsilon=rho**3*e(c/rho)/2.
Source: Prolhac, arXiv:1610.08912; Ristivojevic, arXiv:1905.13705.
This tail is used only at local gamma about .002 or less, is checked
against direct higher-order Bethe solves, and is smoothly matched in mu
and its first derivative. The tiny matching correction decays at infinity.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.special import roots_legendre, zeta
from scipy.optimize import brentq

VERSION = 1
WEAK_A = (3*zeta(3, 1)/8-.5)/np.pi**3
WEAK_B = 1/6-1/np.pi**2


@lru_cache(maxsize=32)
def quadrature(order):
    return roots_legendre(order)


def bethe(q, order=None):
    """Dimensionless c=1 dressing; increase resolution with Q/c.

    The narrow Lorentzian is resolved before matrix inversion. Using one
    fixed order while increasing Q/c can produce unphysical densities.
    """
    if order is None:
        order = max(128, int(np.ceil(32*q/128))*128)
    z, w = quadrature(order)
    k, weights = q*z, q*w
    kernel = 1/(np.pi*(1+(k[:, None]-k[None, :])**2))
    mat = np.eye(order)-kernel*weights[None, :]
    dressed = np.linalg.solve(mat, np.column_stack((np.ones(order), k, k*k/2)))
    edge = weights/(np.pi*(1+(q-k)**2))
    Z = 1+edge@dressed[:, 0]
    mu = (q*q/2+edge@dressed[:, 2])/Z
    sound = (q+edge@dressed[:, 1])/Z
    rho = weights@dressed[:, 0]/(2*np.pi)
    energy = weights@(k*k*dressed[:, 0])/(4*np.pi)
    if min(rho, mu, sound, energy) <= 0:
        raise ValueError('Nonphysical Bethe solution: increase quadrature order')
    return np.array([rho, mu, sound, energy])


def weak(n, derivative=0):
    """Energy, chemical potential or compressibility of the asymptotic EOS."""
    n = np.asarray(n)
    if derivative == 0:
        return .5*n*n-2*n**1.5/(3*np.pi)+WEAK_B*n/2+WEAK_A*np.sqrt(n)/2
    if derivative == 1:
        return n-np.sqrt(n)/np.pi+WEAK_B/2+WEAK_A/(4*np.sqrt(n))
    if derivative == 2:
        return 1-1/(2*np.pi*np.sqrt(n))-WEAK_A/(8*n**1.5)
    raise ValueError(derivative)


def build_table(path, size=1200):
    """Build once; the resulting NPZ is shared by every finite coupling."""
    path = Path(path)
    if path.exists():
        with np.load(path) as f:
            if int(f['version']) != VERSION or int(f['size']) != size:
                raise ValueError('EOS cache settings differ; use a separate output directory')
        return
    qs = np.geomspace(1e-7, 44., size)
    rows = []
    for i, q in enumerate(qs):
        rows.append(bethe(q))
        if (i+1) % 200 == 0:
            print(f'EOS table {i+1}/{size}, Q/c={q:.3g}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, version=VERSION, size=size, q=np.r_[0., qs],
                        data=np.vstack((np.zeros(4), rows)))


class ProductionEOS:
    """Repulsive LL EOS at fixed c, including the vacuum and high density."""
    def __init__(self, c, table):
        if not np.isfinite(c) or c <= 0:
            raise ValueError('Finite repulsive coupling required')
        self.c, self.name = float(c), f'Lieb-Liniger gamma={c:g}'
        with np.load(table) as f:
            self.q = f['q']
            data = f['data']
            size = int(f['size'])
        n, mu, sound, energy = data.T
        if np.any(np.diff(n) <= 0) or np.any(np.diff(mu) <= 0):
            raise ValueError('Nonmonotone Bethe table')
        slope = np.divide(sound**2, n, out=np.zeros_like(n), where=n>0)
        self._mu = CubicHermiteSpline(n, mu, slope, extrapolate=False)
        self._dmu = self._mu.derivative()
        self._energy = self._mu.antiderivative()
        self.join = float(n[-1])
        self.join_energy = float(self._energy(self.join))
        self.delta0 = float(mu[-1]-weak(self.join, 1))
        self.delta1 = float(slope[-1]-weak(self.join, 2))
        self.q_of_n = PchipInterpolator(n, self.q)
        self.sound = float(np.sqrt(self.dmu(1.)))
        self.mu0 = float(self.mu(1.))
        self.e0 = float(self.energy(1.))
        # The asymptotic continuation is physical and validated, rather
        # than the trial-only tangent continuation of the earlier solver.
        self.rmax = np.inf
        probes = (n[:-1]+n[1:])/2
        if np.any(self._dmu(probes) <= 0):
            raise ValueError('Interpolated energy is not convex')
        self.metadata = dict(c=c, version=VERSION, table_size=size,
            table_density_max=c*self.join, high_density_tail='weak-coupling through gamma^(5/2)',
            maximum_local_gamma_in_tail=1/self.join,
            matching_mu_relative=self.delta0/weak(self.join, 1),
            matching_dmu_relative=self.delta1/weak(self.join, 2),
            weak_expansion_source='https://arxiv.org/abs/1610.08912')

    def _tail(self, n, derivative):
        t = (n-self.join)/self.join
        a, b = self.delta0, self.delta0+self.join*self.delta1
        ex = np.exp(-t)
        if derivative == 0:
            correction = self.join*(a*(-np.expm1(-t))+b*(1-(t+1)*ex))
            return self.join_energy+weak(n)-weak(self.join)+correction
        if derivative == 1:
            return weak(n, 1)+(a+b*t)*ex
        return weak(n, 2)+(b-a-b*t)*ex/self.join

    def _evaluate(self, rho, derivative):
        r = np.asarray(rho)
        n = np.maximum(r/self.c, 0)
        inside = np.minimum(n, self.join)
        interp = (self._energy, self._mu, self._dmu)[derivative]
        # Evaluate the tail only outside the table: this avoids unnecessary
        # work in large sparse solves and sqrt/division at zero density.
        value = np.asarray(interp(inside))
        outside = n > self.join
        if np.any(outside):
            value = value.copy()
            value[outside] = self._tail(n[outside], derivative)
        return self.c**(3-derivative)*value

    def energy(self, rho):
        return self._evaluate(rho, 0)

    def mu(self, rho):
        return self._evaluate(rho, 1)

    def dmu(self, rho):
        return self._evaluate(rho, 2)

    def potential(self, r):
        return (self.energy(r)-self.e0-self.mu0*(r-1))/self.sound**2

    def derivative(self, r):
        return (self.mu(r)-self.mu0)/self.sound**2

    def second(self, r):
        return self.dmu(r)/self.sound**2

    def background(self):
        """Direct Bethe background, using the table only to bracket Q/c."""
        n0 = 1/self.c
        guess = float(self.q_of_n(n0)) if n0 <= self.join else 2*np.sqrt(n0)
        q = brentq(lambda q: bethe(q)[0]-1/self.c, .95*guess, 1.05*guess,
                   xtol=2e-13)
        values = bethe(q)
        checked = bethe(q, max(256, int(np.ceil(48*q/128))*128))
        error = float(np.max(abs(checked/values-1)))
        if error > 2e-9:
            raise ValueError(f'Background Bethe quadrature unresolved: {error}')
        return dict(q=self.c*q, cs0=self.c*values[2],
                    quadrature_relative_check=error,
                    interpolated_sound_relative_error=self.sound/(self.c*values[2])-1)


def validate_table(table):
    """Direct higher-order solves at off-grid points and in the tail."""
    eos = ProductionEOS(1., table)
    records = []
    for q in (.0013, .071, .43, 1.37, 4.1, 13.7, 27.3, 41., 49., 60.):
        ref = bethe(q, max(256, int(np.ceil(48*q/128))*128))
        n, mu, cs, en = ref
        records.append(dict(q=q, n=n, mu_relative=float(eos.mu(n)/mu-1),
            sound_squared_relative=float(n*eos.dmu(n)/cs**2-1),
            energy_relative=float(eos.energy(n)/en-1), tail=bool(n>eos.join)))
    maxima = {key:max(abs(r[key]) for r in records) for key in
              ('mu_relative', 'sound_squared_relative', 'energy_relative')}
    if max(maxima.values()) > 2e-6:
        raise ValueError('EOS validation failed: '+json.dumps(maxima))
    return dict(metadata=eos.metadata, off_grid_checks=records, max_errors=maxima)
