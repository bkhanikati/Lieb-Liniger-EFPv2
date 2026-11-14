"""Production EFP hydrodynamics for a list of background couplings.

Example: python efp_production.py --gamma inf 10 1 .1 .01
All paths are relative to this file unless --output is supplied. Runs are
serial and restartable. No historical study data or prescribed internal
boundary is needed: each coupling starts from a feasible coarse field.

Three adaptive coarse refinements and one inner bisection prepare a fine field;
two subsequent refinements target its numerical interface. The outer box
is fixed in acoustic coordinates. Every solve includes the vacuum action
and reports its achieved convex optimization gap and quadrature checks.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')
import numpy as np
from scipy.ndimage import binary_dilation
from efp_production_core import TonksEOS
from efp_production_fem import initial_axes, evaluate, solve_free, adapt_axes
from efp_production_eos import build_table, ProductionEOS, validate_table
from efp_production_reference import reference

HERE = Path(__file__).resolve().parent
VERSION = 1


def load_fields(path):
    with np.load(path) as f:
        return {key:f[key] for key in f.files}


def bisect_inner(axis):
    centers = (axis[:-1]+axis[1:])/2
    return np.sort(np.r_[axis, centers[centers < 2.]])


def refine_interface(fields, add=64):
    """Nested, budgeted refinement using only the current hydro field.

    Protect both cusp neighborhoods; prioritize wide cells in the low-
    density liquid and interface. If a time-fitting band has many tiny
    cells, keep an explicit budget instead of refining that band without
    limit. Tensor splines insert full rows/columns, so budgets matter.
    """
    x, s = fields['x_axis'], fields['s_axis']
    xc, sc = (x[:-1]+x[1:])/2, (s[:-1]+s[1:])/2
    rho = evaluate(fields, xc, sc)[0]
    ar = evaluate(fields, [0.], s)[0][:, 0]
    tip = float(s[np.flatnonzero(ar >= .02)[0]])
    wet = rho >= .02
    edge = binary_dilation(wet, iterations=2)&binary_dilation(~wet, iterations=2)
    band = (edge|((rho >= .02)&(rho <= .3))) & (xc[None, :] < 1.1)&(sc[:, None] < 1.2*tip)
    new_axes, monitor = [], dict(numerical_tip=tip)
    for name, axis, projected in (('x',x,band.any(axis=0)),('s',s,band.any(axis=1))):
        center, width = (axis[:-1]+axis[1:])/2, np.diff(axis)
        chosen = set(range(min(8, len(center))))
        if name == 'x':
            slit = np.searchsorted(axis, 1.)
            chosen.update(range(max(0,slit-8),min(len(center),slit+8)))
        else:
            for density in (.001, .02):
                j = int(np.flatnonzero(ar >= density)[0])
                chosen.update(range(max(0,j-2),min(len(center),j+2)))
            low = max(0, np.flatnonzero(ar >= 1e-4)[0]-2)
            high = min(len(center), np.flatnonzero(ar >= .24)[0]+2)
            for j in np.arange(low, high)[np.argsort(width[low:high])[::-1]]:
                if len(chosen) >= add//2:
                    break
                chosen.add(int(j))
        if len(chosen) > add:
            raise ValueError('Refinement budget too small for protected cells')
        score = np.where(projected & (center < 2.), width, -np.inf)
        for j in np.argsort(score, kind='stable')[::-1]:
            if len(chosen) >= add:
                break
            if not np.isfinite(score[j]):
                raise ValueError('Refinement budget exceeds eligible interface cells')
            chosen.add(int(j))
        nodes = np.sort(np.r_[axis, center[sorted(chosen)]])
        assert np.array_equal(nodes[nodes >= 2.], axis[axis >= 2.])
        new_axes.append(nodes)
        monitor['inserted_'+name] = center[sorted(chosen)]
    return tuple(new_axes), monitor


SLIT_FIT_NODES = 12  # the widest liquid_curve window ends 12 knots past rho=.02


def slit_bands(profiles, axis, eligible):
    """Lines whose rho=.02 to rho=.8 band holds too few knots to be fitted.

    Each row of profiles is the density along one line, sampled at the
    knots of axis. Returns (band width, first knot, last knot) per line.
    """
    bands = []
    for line in np.flatnonzero(eligible):
        r = profiles[line]
        wet = np.flatnonzero(r >= .02)
        if r[0] >= .02 or not len(wet):
            continue
        j = int(wet[0])
        dense = np.flatnonzero(r[j:] > .8)
        k = j+int(dense[0]) if len(dense) else len(r)-1
        if k-j >= SLIT_FIT_NODES:
            continue
        low, high = max(0, j-1), min(len(axis)-1, k+1)
        bands.append((float(axis[high]-axis[low]), low, high))
    return sorted(bands, reverse=True)


def refine_slit(fields, add=64):
    """Opt-in nested refinement of the slit-endpoint corner.

    Near (x,s)=(1,0) the density rises from .02 to above .8 within a few
    x-knots, so every horizontal liquid-side window is rejected there.
    A time row is deficient when fewer than SLIT_FIT_NODES knots separate
    its rho=.02 crossing from its first knot above .8; columns 1/2<x<1 are
    tested the same way along s. Cells spanning deficient bands are
    bisected, widest band first (closest to resolved), while each line
    fits within the budget. Only the numerical density enters; no
    analytic curve or time selects cells. Nodes beyond 2 are unchanged.
    """
    x, s = fields['x_axis'], fields['s_axis']
    rho = evaluate(fields, x, s)[0]
    rows = slit_bands(rho, x, (s > 0.)&(s < 1.))
    columns = slit_bands(rho.T, s, (x > .5)&(x < 1.))
    new_axes, monitor = [], dict(deficient_rows=len(rows), deficient_columns=len(columns))
    for name, axis, bands in (('x', x, rows), ('s', s, columns)):
        center = (axis[:-1]+axis[1:])/2
        chosen = set()
        for _, low, high in bands:
            cells = {j for j in range(low, high) if center[j] < 2.}
            if len(chosen | cells) <= add:
                chosen |= cells
        nodes = np.sort(np.r_[axis, center[sorted(chosen)]])
        assert np.array_equal(nodes[nodes >= 2.], axis[axis >= 2.])
        new_axes.append(nodes)
        monitor['inserted_'+name] = center[sorted(chosen)]
    if not len(monitor['inserted_x'])+len(monitor['inserted_s']):
        raise ValueError('Slit refinement found no deficient band within budget')
    return tuple(new_axes), monitor


def run_case(gamma, output, config):
    if config['n'] < 16:
        raise ValueError('Use n>=16: smaller coarse meshes do not resolve the inherited outer trace sufficiently')
    tag = 'inf' if np.isinf(gamma) else f'{gamma:g}'
    folder = output/('gamma_'+tag)
    folder.mkdir(parents=True, exist_ok=True)
    eos = TonksEOS() if np.isinf(gamma) else ProductionEOS(gamma, output/'eos_table.npz')
    expected = dict(version=VERSION, gamma=tag, **config)
    cfg = folder/'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != expected:
        raise ValueError(f'Existing configuration differs in {folder}; choose a new output directory')
    cfg.write_text(json.dumps(expected, indent=2)+'\n')
    ref, curve = reference(gamma, None if np.isinf(gamma) else eos)
    (folder/'reference.json').write_text(json.dumps(ref, indent=2)+'\n')
    np.savez_compressed(folder/'reference.npz', **curve)
    x, s = initial_axes(config['n'], config['extent'])
    previous = None
    records = []
    names = ['coarse']+[f'bootstrap_{k}' for k in range(1,config['bootstrap']+1)]+['base']+[
        f'refine_{k}' for k in range(1,config['refinements']+1)]+[
        f'slit_{k}' for k in range(1,config.get('slit_refinements',0)+1)]
    for stage in names:
        monitor = None
        if stage.startswith('bootstrap'):
            number = int(stage.split('_')[-1])
            (x,s),monitor = adapt_axes(previous,add=20,focused_tip=(number==config['bootstrap']))
        elif stage=='base':
            x, s = bisect_inner(x), bisect_inner(s)
        elif stage.startswith('refine'):
            (x,s), monitor = refine_interface(previous, config['add'])
        elif stage.startswith('slit'):
            (x,s), monitor = refine_slit(previous, config['slit_add'])
        record_path, field_path = folder/(stage+'.json'), folder/(stage+'.npz')
        if record_path.exists() and field_path.exists():
            saved = json.loads(record_path.read_text())
            result, fields = saved['result'], load_fields(field_path)
            assert np.array_equal(fields['x_axis'], x) and np.array_equal(fields['s_axis'], s)
            assert np.isclose(result['sound'], eos.sound, rtol=0, atol=1e-13)
            print(f'REUSE gamma={tag} {stage}', flush=True)
        else:
            print(f'START gamma={tag} {stage}: {len(x)-1} x {len(s)-1} cells', flush=True)
            barriers = None if previous is None else (1e-8,1e-9)
            result, fields, history = solve_free(x,s,eos,ref['alpha'],ref['cs0'],
                initial=previous,barriers=barriers,gap_atol=config['gap_atol'])
            result.update(gamma=tag,stage=stage,production_version=VERSION,
                analytic_action=ref['action'],
                relative_action_error=result['corrected_action_rechecked']/ref['action']-1,
                analytic_internal_boundary_used=False)
            # Write the field before its completion record. An interrupted
            # write cannot make a stage look complete to the restart logic.
            np.savez_compressed(field_path, **fields)
            record_path.write_text(json.dumps(dict(result=result,history=history),indent=2)+'\n')
            if monitor is not None:
                np.savez_compressed(folder/(stage+'_mesh.npz'), **monitor)
            print('COMPLETE '+json.dumps(dict(gamma=tag,stage=stage,
                coefficients=result['coefficients'],action=result['corrected_action_rechecked'],
                relative_action_error=result['relative_action_error'],
                seconds=result['elapsed_seconds'],peak_mib=result['process_peak_rss_mib'])),flush=True)
        previous = fields
        records.append(result)
        (folder/'convergence.json').write_text(json.dumps(records,indent=2)+'\n')
        gc.collect()
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gamma', nargs='+', type=float, default=[np.inf,10.,1.,.1,.01])
    p.add_argument('--output', type=Path, default=HERE/'production-output')
    p.add_argument('--n', type=int, default=64, help='coarse inner intervals per axis; minimum 16')
    p.add_argument('--extent', type=float, default=13.)
    p.add_argument('--bootstrap', type=int, default=3, help='coarse adaptive preparations before inner bisection')
    p.add_argument('--refinements', type=int, default=2)
    p.add_argument('--add', type=int, default=64)
    p.add_argument('--gap-atol', type=float, default=1e-6)
    p.add_argument('--slit-refinements', type=int, default=0,
                   help='opt-in slit-endpoint passes after --refinements; 0 keeps the original schedule')
    p.add_argument('--slit-add', type=int, default=64, help='knot budget per axis for each slit pass')
    p.add_argument('--solve-only', action='store_true', help='skip diagnostics and plotting')
    args = p.parse_args()
    if any(g <= 0 or np.isnan(g) for g in args.gamma):
        p.error('Couplings must be positive, or inf for TG')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(np.isfinite(g) for g in args.gamma):
        build_table(args.output/'eos_table.npz')
        check_path = args.output/'eos_validation.json'
        if not check_path.exists():
            check_path.write_text(json.dumps(validate_table(args.output/'eos_table.npz'),indent=2)+'\n')
    config = dict(n=args.n,extent=args.extent,bootstrap=args.bootstrap,
                  refinements=args.refinements,add=args.add,gap_atol=args.gap_atol)
    # Recorded only when used, so original-schedule outputs stay comparable.
    if args.slit_refinements:
        config.update(slit_refinements=args.slit_refinements,slit_add=args.slit_add)
    for gamma in args.gamma:
        run_case(gamma,args.output,config)
    if not args.solve_only:
        from efp_production_results import make_results
        make_results(args.output,args.gamma)


if __name__ == '__main__':
    main()
