"""3D density surfaces rho(x, tau), colored by density (read-only).

Usage: python tools/plot_surface.py runs/slit-4g [--gamma inf 1 0.01] [--zmax 3.4]

Both signs of x and tau are shown (rho is even in each). The mesh is polar
around the origin and starts on the vacuum boundary, so the vacuum is a clean
hole; radial spacing is graded toward the boundary and angles are packed
toward the slit endpoints x = +-1, tau = 0. The analytic boundary is used only
to place the mesh: the plotted values are the final-stage numerical density at
the mesh nodes. Height and color both show rho; color uses the same scale as
plot_density.py, rho/(1+rho) on RdBu_r. The endpoint spikes diverge and are
left open where they leave the box at --zmax. Writes hydro-surface.png and
.pdf into the run folder; nothing else changes.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from efp_production_fem import basis

XMAX, SMAX = 1.3, 1.15
N_ANGLE, N_RADIAL = 128, 34
BOX = (1.25, 1.1, 1.0)
LIGHT = np.array([-.35, -.55, .76])/np.linalg.norm([-.35, -.55, .76])
TICKS = (0., .1, .25, .5, 1., 2., 4., 10.)


def read_npz(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def color_scale(rho):
    rho = np.asarray(rho, dtype=float)
    return rho/(1+rho)


def density_at(fields, x, s):
    # Point evaluation of rho = U_x (efp_production_fem.evaluate only does tensor grids).
    bx, dx = basis(fields['x_knots'], np.asarray(x))
    bs, ds = basis(fields['s_knots'], np.asarray(s))
    rows = csr_matrix(bs)@fields['coefficients']
    return np.asarray(csr_matrix(dx).multiply(rows).sum(axis=1)).ravel()


def polar_mesh(boundary_x, boundary_s):
    t = np.linspace(0., 2*np.pi, N_ANGLE+1)
    theta = t-.45*np.sin(2*t)                          # packed toward theta = 0, pi
    q = np.arctan2(abs(np.sin(theta)), abs(np.cos(theta)))  # angle folded into the first quadrant
    b_theta = np.arctan2(boundary_s, boundary_x)
    order = np.argsort(b_theta)
    r_in = np.interp(q, b_theta[order], np.hypot(boundary_x, boundary_s)[order])
    r_out = np.minimum(XMAX/np.maximum(np.cos(q), 1e-12), SMAX/np.maximum(np.sin(q), 1e-12))
    u = np.linspace(0., 1., N_RADIAL)**1.8
    r = r_in[None, :]+(r_out-r_in)[None, :]*u[:, None]
    return r*np.cos(theta)[None, :], r*np.sin(theta)[None, :]


def load(folder):
    final = json.loads((folder/'observables.json').read_text())['final']
    fields = read_npz(folder/(final['stage']+'.npz'))
    reference = read_npz(folder/'reference.npz')
    bx, bs = reference['x'], reference['tau']*final['sound']
    keep = (bx >= 0) & (bs >= 0)
    X, S = polar_mesh(np.r_[bx[keep], 1.], np.r_[bs[keep], 0.])
    rho = density_at(fields, abs(X).ravel(), abs(S).ravel()).reshape(X.shape)
    return dict(tag=final['gamma'], X=X, S=S, rho=np.maximum(rho, 0.))


def surface(ax, case, zmax):
    X, S, rho = case['X'], case['S'], case['rho']
    Z = np.minimum(rho, zmax)
    # Soft Lambert shading from cell normals in box-scaled coordinates.
    P = np.stack([X/(2*XMAX)*BOX[0], S/(2*SMAX)*BOX[1], Z/zmax*BOX[2]], axis=-1)
    n = np.cross(P[1:, 1:]-P[:-1, :-1], P[:-1, 1:]-P[1:, :-1])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)+1e-15
    n *= np.sign(n[..., 2:3]+1e-15)
    c = color_scale(rho)
    cell = (c[1:, 1:]+c[:-1, :-1]+c[1:, :-1]+c[:-1, 1:])/4
    faces = np.ones(rho.shape+(4,))
    faces[:-1, :-1, :3] = plt.get_cmap('RdBu_r')(cell)[..., :3]*(.78+.22*np.clip(n@LIGHT, 0, 1))[..., None]
    # Cells entirely above the box are dropped, so the spikes leave it as open tubes
    # cut along the rho = zmax level rather than ragged facets. Near the slit endpoint
    # a single cell can span the jump from the vacuum rim (rho ~ 0) to the divergence;
    # drawn, it is a vertical sheet inside the spike, so such cells are dropped too.
    corners = np.stack([rho[1:, 1:], rho[:-1, :-1], rho[1:, :-1], rho[:-1, 1:]])
    above = np.zeros(rho.shape, dtype=bool)
    above[:-1, :-1] = (corners.min(0) >= zmax) | ((corners.max(0) >= zmax) & (corners.min(0) < .6*zmax))
    faces[above] = 0.
    patches = ax.plot_surface(X, S, Z, facecolors=faces, rstride=1, cstride=1, shade=False,
                              linewidth=.25, antialiased=True)
    edges = np.tile([0., 0., 0., .22], (above[:-1, :-1].size, 1))
    edges[above[:-1, :-1].ravel()] = 0.
    patches.set_edgecolors(edges)
    ax.set(xlim=(-XMAX, XMAX), ylim=(-SMAX, SMAX), zlim=(0, zmax))
    ax.set_title(rf"$\gamma_0={r'\infty' if case['tag'] == 'inf' else case['tag']}$", y=.92, fontsize=13)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks(np.arange(0, zmax+1e-9, 1.))
    ax.set_xlabel(r'$x/R$', labelpad=-2, fontsize=11)
    ax.set_ylabel(r'$c_{s0}\tau/R$', labelpad=-2, fontsize=11)
    ax.set_zlabel(r'$\rho/\rho_0$', labelpad=-4, fontsize=11)
    ax.tick_params(labelsize=9, pad=-1)
    ax.view_init(elev=22, azim=-76)
    ax.set_box_aspect(BOX)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis._axinfo['grid']['color'] = (0, 0, 0, 0)


def main(run, gammas, zmax):
    run = Path(run)
    cases = [load(run/('gamma_'+tag)) for tag in gammas]
    fig = plt.figure(figsize=(5.4*len(cases)+1., 5.2), constrained_layout=True)
    axes = [fig.add_subplot(1, len(cases), i+1, projection='3d') for i in range(len(cases))]
    for ax, case in zip(axes, cases):
        surface(ax, case, zmax)
    bar = fig.colorbar(ScalarMappable(Normalize(0., 1.), 'RdBu_r'), ax=axes, shrink=.6, aspect=30,
                       label=r'$\rho/\rho_0$')
    bar.set_ticks(list(color_scale(TICKS))+[1.], labels=[f'{t:g}' for t in TICKS]+[r'$\infty$'])
    bar.ax.tick_params(labelsize=8)
    for suffix in ('png', 'pdf'):
        fig.savefig(run/('hydro-surface.'+suffix), dpi=180)
    print('wrote', run/'hydro-surface.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', nargs='?', default='runs/slit-4g')
    p.add_argument('--gamma', nargs='+', default=['inf', '1', '0.01'],
                   help='couplings, as in the folder names (default: inf 1 0.01)')
    p.add_argument('--zmax', type=float, default=3.4, help='box height in rho/rho0; the spikes leave the box here')
    args = p.parse_args()
    main(args.run, args.gamma, args.zmax)
