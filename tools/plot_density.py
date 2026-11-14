"""Density maps on a color scale centered at the background density (read-only).

Usage: python tools/plot_density.py runs/slit-4g [--zoom] [--linear [VMAX]] [--cmap BuYl] [--columns 2] [--gamma 100 1] [--contours]

The production figure (hydro-profiles.png) colors rho on a log scale from
1e-4 to 4, so the whole liquid sits in the top of the colormap. Here the
color coordinate is rho/(1+rho) = 1/(1+exp(-log rho)): logarithmic near
rho0 = 1 (white), symmetric under rho -> 1/rho, and reaching the ends only at
vacuum (darkest blue) and rho -> infinity (darkest red), so neither the
vacuum nor the divergence at the slit endpoint is clipped. Top row: the same window as the production
figure, from the saved profiles.npz. With --zoom, a second row shows the slit
endpoint, with the final-stage field evaluated on a finer grid.
Writes hydro-density.png and .pdf into the run folder; nothing else changes.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from efp_production_fem import evaluate

ZOOM_X, ZOOM_S = (.8, 1.2), (0., .2)
TICKS = (0., .1, .25, .5, 1., 2., 4., 10.)


def color_scale(rho):
    # Logistic in log(rho): symmetric under rho -> 1/rho, finite at 0 and infinity.
    rho = np.asarray(rho, dtype=float)
    return rho/(1+rho)


def read_npz(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def order(folder):
    tag = folder.name.removeprefix('gamma_')
    return -np.inf if tag == 'inf' else -float(tag)


CONTOURS = tuple(np.arange(.25, 2.01, .25))
CLABEL_SIZE = 6
CLABEL_LEVELS = None     # which contour levels get labels (None: all)
CLABEL_MANUAL = {}       # level -> (x, s) near where its label should go; placed after the automatic ones


def panel(ax, x, s, rho, case, xlim, slim, linear=None, contours=False):
    values, norm = (rho, linear) if linear else (color_scale(rho), Normalize(0., 1.))
    image = ax.pcolormesh(x, s, values, norm=norm, cmap=case['cmap'], shading='auto', rasterized=True)
    if contours:
        lines = ax.contour(x, s, rho, levels=CONTOURS, colors='k', linewidths=.5, alpha=.6)
        ax.clabel(lines, CLABEL_LEVELS, fontsize=CLABEL_SIZE, fmt='%g', inline_spacing=2)
        for level, point in CLABEL_MANUAL.items():
            ax.clabel(lines, [level], fontsize=CLABEL_SIZE, fmt='%g', inline_spacing=2, manual=[point])
    ax.plot(case['analytic']['x'], case['analytic']['tau']*case['sound'], 'k--', lw=1.2,
            label='Analytic boundary')
    ax.plot(case['curves']['liquid_x'], case['curves']['liquid_s'], color='#f2c200', lw=1.4,
            label='Liquid-side hydro fit')
    ax.set(xlim=xlim, ylim=slim, xlabel=r'$x/R$', ylabel=r'$c_{s0}\tau/R$')
    return image


CMAPS = dict(
    RdBu_r='RdBu_r',
    cividis='cividis',
    # RdBu_r's blue half, then white through yellow to dark gold.
    BuYl=LinearSegmentedColormap.from_list('BuYl', [
        '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
        '#fff3b0', '#ffe066', '#f5c400', '#c79a00', '#7a5c00']))


def main(run, zoom=False, linear=None, cmap='RdBu_r', columns=None, gammas=None, contours=False):
    run = Path(run)
    folders = sorted((f for f in run.glob('gamma_*') if (f/'profiles.npz').exists()), key=order)
    if gammas:
        folders = [f for f in folders if f.name.removeprefix('gamma_') in gammas]
    suffix_name = 'max' if linear == 0. else f'{linear:g}' if linear else ''
    if linear == 0.:
        linear = max(read_npz(f/'profiles.npz')['rho'].max() for f in folders)
    norm = TwoSlopeNorm(vmin=0., vcenter=1., vmax=linear) if linear else None
    columns = columns or len(folders)
    rows = -(-len(folders)//columns)*(1+zoom)  # each zoom row sits under its row of panels
    fig, axes = plt.subplots(rows, columns, figsize=(3.6*columns+1.2, 3.7*rows),
                             constrained_layout=True, squeeze=False)
    zx, zs = np.linspace(*ZOOM_X, 401), np.linspace(*ZOOM_S, 201)
    used = []
    for i, folder in enumerate(folders):
        final = json.loads((folder/'observables.json').read_text())['final']
        case = dict(cmap=CMAPS[cmap], sound=final['sound'], analytic=read_npz(folder/'reference.npz'),
                    curves=read_npz(folder/(final['stage']+'_curves.npz')))
        p = read_npz(folder/'profiles.npz')
        row, column = (1+zoom)*(i//columns), i % columns
        ax = axes[row, column]
        image = panel(ax, p['x'], p['s'], p['rho'], case, (0, 1.12), (0, 1.08), norm, contours)
        ax.plot(0., final['tau_c']*final['sound'], 'wo', ms=4, mec='k', mew=.5)
        tag = final['gamma']
        ax.set_title(rf"$\gamma={r'\infty' if tag == 'inf' else tag}$")
        used.append(ax)
        if not zoom:
            continue
        corner = evaluate(read_npz(folder/(final['stage']+'.npz')), zx, zs)[0]
        ax = axes[row+1, column]
        panel(ax, zx, zs, corner, case, ZOOM_X, ZOOM_S, norm, contours)
        ax.set_title(rf"slit endpoint, max $\rho/\rho_0$ = {final['maximum_density']:.0f}", fontsize=9)
        used.append(ax)
    for ax in axes.ravel():
        ax.set_visible(ax in used)
    axes[0, 0].legend(fontsize=7, loc='upper right', facecolor='white', framealpha=.9)
    bar = fig.colorbar(image, ax=used, aspect=20*rows, shrink=.8 if zoom else 1., label=r'$\rho/\rho_0$',
                       extend='max' if linear else 'neither')
    if not linear:
        bar.set_ticks(list(color_scale(TICKS))+[1.], labels=[f'{t:g}' for t in TICKS]+[r'$\infty$'])
    bar.ax.tick_params(labelsize=8)
    name = 'hydro-density-linear'+suffix_name if linear else 'hydro-density'
    name += '' if cmap == 'RdBu_r' else '-'+cmap
    name += '-contours' if contours else ''
    for suffix in ('png', 'pdf'):
        fig.savefig(run/(name+'.'+suffix), dpi=180)
    print('wrote', run/(name+'.png'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', nargs='?', default='runs/slit-4g')
    p.add_argument('--zoom', action='store_true', help='add a row zoomed on the slit endpoint')
    p.add_argument('--linear', type=float, nargs='?', const=0., metavar='VMAX',
                   help='for comparison: linear on each side of rho0 up to VMAX (default: largest density), '
                        'written to hydro-density-linear<VMAX or max>.png')
    p.add_argument('--cmap', choices=CMAPS, default='RdBu_r',
                   help='colormap; non-default choices add -<cmap> to the file name')
    p.add_argument('--columns', type=int, help='panels per row (default: all in one row)')
    p.add_argument('--gamma', nargs='+', help='couplings to plot, as in the folder names (default: all)')
    p.add_argument('--contours', action='store_true',
                   help='add labelled constant-density lines; adds -contours to the file name')
    args = p.parse_args()
    main(args.run, args.zoom, args.linear, args.cmap, args.columns, args.gamma, args.contours)
