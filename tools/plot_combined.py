"""3D density surfaces above the matching 2D density maps (read-only).

Usage: python tools/plot_combined.py runs/slit-4g [--gamma 100 1 0.01] [--contours] [--zmax 3.4] [--width 7]

Top row: plot_surface.py panels. Bottom row: plot_density.py panels on the
same rho/(1+rho) color scale, so one colorbar serves both rows; they share
the time axis, so only the first is labelled. The figure is drawn at its
printed width (--width, inches, default 7 for \\linewidth), so font sizes
are the sizes on the page. Writes hydro-combined.png and .pdf into the run
folder; nothing else changes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.contour import ContourSet
import plot_density
import plot_surface

# Point sizes on the page.
TICK, LABEL, TITLE, SMALL = 10, 10.5, 11.5, 7.5
CONTOURS = (.5, 1., 1.5, 2.)


def style_top(ax, last):
    # x and tau scales are read off the bottom row; the rho axis is labelled once, on the right.
    ax.title.set_fontsize(TITLE)
    ax.title.set_y(.9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='z', labelsize=TICK, pad=-1)
    if last:
        ax.zaxis.label.set_fontsize(LABEL)
        ax.zaxis.labelpad = -3
    else:
        ax.set_zlabel('')
        ax.tick_params(axis='z', labelleft=False, labelright=False)
        ax.set_zticklabels([])
    ax.set_box_aspect(plot_surface.BOX, zoom=1.22)


def style_bottom(ax, first):
    ax.tick_params(labelsize=TICK, length=3, pad=2)
    ax.set_xticks([0, .5, 1.])
    ax.set_yticks([0, .5, 1.])
    ax.xaxis.label.set_fontsize(LABEL)
    if first:
        ax.yaxis.label.set_fontsize(LABEL)
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)


def main(run, gammas, contours=False, zmax=3.4, width=7.):
    run = Path(run)
    n = len(gammas)
    plot_density.CONTOURS = CONTOURS
    plot_density.CLABEL_SIZE = SMALL       # set before clabel so the inline gaps fit the text
    # Small loops at the endpoint: label them at their tops, clear of the boundary line.
    plot_density.CLABEL_LEVELS = [.5, 1.]
    plot_density.CLABEL_MANUAL = {1.5: (.98, .135), 2.: (.98, .065)}
    fig = plt.figure(figsize=(width, width*.72))
    # The 3D row keeps a right margin for the rho axis; the maps use the full width.
    # Negative hspace: the 3D axes reserve room below the box that holds no labels here.
    # The strip below the maps holds the horizontal colorbar and the legend.
    upper = fig.add_gridspec(1, n, left=.045, right=.945, bottom=.52, top=1.03, wspace=.04)
    lower = fig.add_gridspec(1, n, left=.065, right=.985, bottom=.115, top=.61, wspace=.04)
    grid = [upper, lower]
    top, bottom = [], []
    for i, tag in enumerate(gammas):
        folder = run/('gamma_'+tag)
        ax = fig.add_subplot(grid[0][0, i], projection='3d')
        plot_surface.surface(ax, plot_surface.load(folder), zmax)
        style_top(ax, last=i == n-1)
        top.append(ax)
        final = json.loads((folder/'observables.json').read_text())['final']
        case = dict(cmap='RdBu_r', sound=final['sound'], analytic=plot_density.read_npz(folder/'reference.npz'),
                    curves=plot_density.read_npz(folder/(final['stage']+'_curves.npz')))
        p = plot_density.read_npz(folder/'profiles.npz')
        ax = fig.add_subplot(grid[1][0, i], sharey=bottom[0] if bottom else None)
        plot_density.panel(ax, p['x'], p['s'], p['rho'], case, (0, 1.12), (0, 1.08), contours=contours)
        # The dark region already shows the numerical vacuum, so overlay only the analytic
        # boundary; the fitted numerical boundary is compared quantitatively in hydro-boundary.
        for line in ax.get_lines():
            if line.get_label() == 'Liquid-side hydro fit':
                line.remove()
            elif line.get_label() == 'Analytic boundary':
                line.set(color='#f2c200', linestyle='-', linewidth=2., label='Analytic boundary')
        for artist in ax.collections:
            if isinstance(artist, ContourSet):
                artist.set_linewidth(.85)
        ax.plot(0., final['tau_c']*final['sound'], 'wo', ms=3, mec='k', mew=.5)
        ax.set_aspect('equal')
        style_bottom(ax, first=not bottom)
        bottom.append(ax)
    # Colorbar under the maps.
    left, right = bottom[0].get_position(), bottom[-1].get_position()
    x0 = left.x0+.06                       # room for the rho/rho0 label on the bar's left
    cax = fig.add_axes([x0, .035, right.x1-x0, .022])
    bar = fig.colorbar(ScalarMappable(Normalize(0., 1.), 'RdBu_r'), cax=cax, orientation='horizontal')
    bar.set_ticks(list(plot_density.color_scale(plot_density.TICKS))+[1.],
                  labels=[f'{t:g}' for t in plot_density.TICKS]+[r'$\infty$'])
    bar.ax.tick_params(labelsize=TICK, length=3, pad=2)
    cax.text(-.015, .5, r'$\rho/\rho_0$', fontsize=LABEL, transform=cax.transAxes, ha='right', va='center')
    # One entry: it fits in the empty upper right of the first map.
    bottom[0].legend(fontsize=LABEL-1, loc='upper right', facecolor='white', framealpha=.9,
                     handlelength=1.6, borderpad=.35)
    # Tight bounding box: trims white space and keeps labels that reach past the edge.
    for suffix in ('png', 'pdf'):
        fig.savefig(run/('hydro-combined.'+suffix), dpi=300, bbox_inches='tight', pad_inches=.02)
    print('wrote', run/'hydro-combined.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', nargs='?', default='runs/slit-4g')
    p.add_argument('--gamma', nargs='+', default=['100', '1', '0.01'],
                   help='couplings, as in the folder names (default: 100 1 0.01)')
    p.add_argument('--contours', action='store_true', help='constant-density lines on the 2D maps')
    p.add_argument('--zmax', type=float, default=3.4, help='3D box height in rho/rho0')
    p.add_argument('--width', type=float, default=7., help='printed width in inches (default 7)')
    args = p.parse_args()
    main(args.run, args.gamma, args.contours, args.zmax, args.width)
