"""Vacuum-boundary comparison: analytic hydrodynamic curve vs numerical minimization (read-only).

Usage: python tools/plot_boundary.py runs/slit-4g [--gamma 100 10 1 0.01] [--markers 16] [--no-limits]

Lines are the analytic boundary (reference.npz); open circles are the
liquid-side boundary extracted from the final stage of the numerical
minimization (<stage>_curves.npz), spaced evenly in arc length along the
part of the curve the fits accept. Dashed lines are the two limits, both
exact polytropes in acoustic coordinates, (x, c_s0 tau)/R = (b_d - eta b_d', b_d'/d)
with b_d(eta) = N_d int_0^eta ds/(1+s^2)^((d+2)/2): gamma = infinity is free
fermions (d = 1, the astroid x^(2/3) + (c_s0 tau)^(2/3) = 1) and gamma -> 0 is
the mean-field Bose gas (d = 2, c_s0 tau_c = 2/pi). Writes hydro-boundary.png
and .pdf into the run folder; nothing else changes. Run results_*.py first.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ('#2b5d8f', '#c4522a', '#2f8065', '#8e5bab', '#b08a1e', '#555555')
LIMIT_COLORS = dict(inf='k', zero='#d08c00')


def read_npz(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def order(folder):
    tag = folder.name.removeprefix('gamma_')
    return -np.inf if tag == 'inf' else -float(tag)


def evenly_spaced(x, s, count):
    # Points nearest to equal arc-length steps along the accepted curve.
    ok = np.isfinite(x)
    x, s = x[ok], s[ok]
    arc = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(s)))]
    picks = np.unique(np.searchsorted(arc, np.linspace(0., arc[-1], count)).clip(0, len(arc)-1))
    return x[picks], s[picks]


def free_fermions():
    t = np.linspace(0., 1., 2001)
    return np.maximum(1-t**(2/3), 0)**1.5, t      # x, c_s0 tau on the astroid


def weak_limit():
    # d = 2 polytrope: b_2 = (2/pi)[arctan eta + eta/(1+eta^2)], b_2' = (4/pi)/(1+eta^2)^2.
    eta = np.r_[0., np.geomspace(1e-4, 1e4, 2000)]
    u = 1+eta**2
    return 2/np.pi*(np.arctan(eta)+eta/u-2*eta/u**2), 2/np.pi/u**2


def main(run, gammas=None, markers=16, limits=True):
    run = Path(run)
    folders = sorted((f for f in run.glob('gamma_*') if (f/'observables.json').exists()), key=order)
    if gammas:
        folders = [f for f in folders if f.name.removeprefix('gamma_') in gammas]
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    handles = []
    if limits:
        ax.plot(*free_fermions(), color=LIMIT_COLORS['inf'], lw=1.3, ls='--')
        handles.append(Line2D([], [], color=LIMIT_COLORS['inf'], lw=1.3, ls='--', label=r'$\gamma_0=\infty$'))
    for folder, color in zip(folders, COLORS):
        final = json.loads((folder/'observables.json').read_text())['final']
        analytic = read_npz(folder/'reference.npz')
        curves = read_npz(folder/(final['stage']+'_curves.npz'))
        tag = final['gamma']
        ax.plot(analytic['x'], analytic['tau']*final['sound'], color=color, lw=1.8)
        x, s = evenly_spaced(curves['liquid_x'], curves['liquid_s'], markers)
        ax.plot(x, s, 'o', ms=4, mfc='white', mec=color, mew=1.1, zorder=3)
        handles.append(Line2D([], [], color=color, lw=1.8,
                              label=rf"$\gamma_0={r'\infty' if tag == 'inf' else tag}$"))
    if limits:
        ax.plot(*weak_limit(), color=LIMIT_COLORS['zero'], lw=1.3, ls='--')
        handles.append(Line2D([], [], color=LIMIT_COLORS['zero'], lw=1.3, ls='--', label=r'$\gamma_0\to 0$'))
    ax.set(xlim=(0, 1), ylim=(0, None), xlabel=r'$x/R$', ylabel=r'$c_{s0}\tau/R$')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(handles=handles, title='Lieb–Liniger', frameon=False, loc='upper right')
    fig.legend(handles=[Line2D([], [], color='.2', lw=1.8, label='Hydrodynamic solution'),
                        Line2D([], [], ls='', marker='o', ms=4, mfc='white', mec='.2', label='Numerical minimization')],
               loc='outside upper center', ncols=2, frameon=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(run/('hydro-boundary.'+suffix), dpi=180)
    print('wrote', run/'hydro-boundary.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', nargs='?', default='runs/slit-4g')
    p.add_argument('--gamma', nargs='+', help='couplings to plot, as in the folder names (default: all)')
    p.add_argument('--markers', type=int, default=16, help='open circles per curve')
    p.add_argument('--no-limits', action='store_true', help='omit the dashed gamma = infinity and gamma -> 0 curves')
    args = p.parse_args()
    main(args.run, args.gamma, args.markers, not args.no_limits)
