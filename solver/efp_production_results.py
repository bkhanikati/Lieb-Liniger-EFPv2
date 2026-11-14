"""Produce figures, numerical tables and convergence diagnostics for all couplings.

This module reads completed solves. Hydro boundary extraction is performed
before the analytic reference is used to compute comparison errors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm
from efp_production_core import TonksEOS
from efp_production_fem import evaluate
from efp_production_eos import ProductionEOS
from efp_production_reference import curve_at_time
from efp_production_diagnostics import extract


def read_npz(path):
    with np.load(path) as f:
        return {key:f[key] for key in f.files}


def analyze_case(output,gamma):
    tag = 'inf' if np.isinf(gamma) else f'{gamma:g}'
    folder = output/('gamma_'+tag)
    config = json.loads((folder/'config.json').read_text())
    eos = TonksEOS() if np.isinf(gamma) else ProductionEOS(gamma,output/'eos_table.npz')
    reference = json.loads((folder/'reference.json').read_text())
    analytic = read_npz(folder/'reference.npz')
    # Only the fine initial solution and interface refinements enter the
    # reported convergence table. Coarse bootstrap records remain saved.
    base = 'base'
    stages = [base]+[f'refine_{i}' for i in range(1,config['refinements']+1)]+[
        f'slit_{i}' for i in range(1,config.get('slit_refinements',0)+1)]
    records, extracted = [], []
    for stage in stages:
        saved = json.loads((folder/(stage+'.json')).read_text())['result']
        fields = read_npz(folder/(stage+'.npz'))
        diagnostics, arrays = extract(fields,eos,gamma)
        closing = diagnostics['closing']
        tau = arrays['liquid_s']/eos.sound
        exact_x = curve_at_time(analytic,tau)
        arrays['analytic_x_at_liquid_times'] = exact_x
        valid = np.isfinite(arrays['liquid_x']) & (tau<=reference['tau_c'])
        difference = arrays['liquid_x'][valid]-exact_x[valid]
        result = dict(gamma=tag,stage=stage,coefficients=saved['coefficients'],sound=eos.sound,
            analytic_sound=reference['cs0'],analytic_action=reference['action'],
            action=saved['corrected_action_rechecked'],action_relative_error=saved['relative_action_error'],
            exterior_action=saved['exterior_action'],tau_c=closing['physical_tip'],
            analytic_tau_c=reference['tau_c'],tau_c_relative_error=closing['physical_tip']/reference['tau_c']-1,
            closing=closing,thresholds=diagnostics['thresholds'],
            action_optimization_gap=saved['optimization']['full_action_gap_bound_for_quadrature_problem'],
            quadrature_q8_minus_q6=saved['quadrature_rechecks_inside']['8']-saved['quadrature_rechecks_inside']['6'],
            curve_valid_time_fraction=float(np.count_nonzero(valid)*np.diff(tau)[0]/reference['tau_c']),
            curve_rms_error=float(np.sqrt(np.mean(difference**2))) if np.any(valid) else None,
            curve_max_error=float(np.max(abs(difference))) if np.any(valid) else None,
            elapsed_seconds=saved['elapsed_seconds'],peak_process_mib=saved['process_peak_rss_mib'],
            maximum_density=saved['density_max_checked'],
            direct_bethe_table_density_max=eos.metadata.get('table_density_max'),
            high_density_eos_tail_used=bool('table_density_max' in eos.metadata and
                saved['density_max_checked']>eos.metadata['table_density_max']))
        if records:
            common = valid & np.isfinite(extracted[-1]['liquid_x'])
            delta = arrays['liquid_x'][common]-extracted[-1]['liquid_x'][common]
            result['change_from_previous'] = dict(action=result['action']-records[-1]['action'],
                tau_c_relative=result['tau_c']/records[-1]['tau_c']-1,
                curve_rms=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                curve_max=float(np.max(abs(delta))) if len(delta) else None)
        records.append(result);extracted.append(arrays)
        (folder/(stage+'_observables.json')).write_text(json.dumps(result,indent=2)+'\n')
        np.savez_compressed(folder/(stage+'_curves.npz'),**arrays)
    common = np.logical_and.reduce([np.isfinite(a['liquid_x']) for a in extracted])
    for record, a in zip(records,extracted):
        delta = a['liquid_x'][common]-a['analytic_x_at_liquid_times'][common]
        record['curve_rms_on_common_points'] = float(np.sqrt(np.mean(delta**2))) if len(delta) else None
    final = records[-1]
    # Include preparation solves in the cost, not just the last refinement.
    # RSS is the process high-water mark, so serial cases may inherit an
    # earlier peak; it is an upper bound on each individual case's need.
    history = json.loads((folder/'convergence.json').read_text())
    final['total_solve_seconds'] = float(sum(r['elapsed_seconds'] for r in history))
    final['stages_completed'] = len(history)
    final['closing_stability_pass'] = bool(len(records)>1 and closing['qualified'] and
        abs(final['change_from_previous']['tau_c_relative'])<.001 and closing['selected_spread_relative']<.001)
    final['uniform_curve_stability_pass'] = bool(len(records)>1 and
        final['change_from_previous']['curve_max'] is not None and final['change_from_previous']['curve_max']<.001)
    # Density/current plotting arrays are in acoustic coordinates. Velocity
    # is deliberately omitted in nearly empty cells, where j/rho is unstable.
    grid_x,grid_s = np.linspace(0.,1.2,481),np.linspace(0.,1.15,461)
    density,k = evaluate(fields,grid_x,grid_s)
    plot_data = dict(x=grid_x,s=grid_s,rho=density,current=-eos.sound*k,
        velocity=np.divide(-eos.sound*k,density,out=np.full_like(density,np.nan),where=density>1e-4))
    np.savez_compressed(folder/'profiles.npz',**plot_data)
    (folder/'observables.json').write_text(json.dumps(dict(final=final,convergence=records),indent=2)+'\n')
    return dict(tag=tag,eos=eos,reference=reference,analytic=analytic,records=records,
                arrays=extracted[-1],profiles=plot_data)


def make_results(output,gammas):
    output = Path(output)
    cases = [analyze_case(output,g) for g in gammas]
    colors = plt.get_cmap('viridis')(np.linspace(.05,.85,len(cases)))
    rows = int(np.ceil((len(cases)+1)/3))
    fig,axes = plt.subplots(rows,3,figsize=(13.2,4.25*rows),constrained_layout=True)
    flat = axes.ravel()
    for ax,case in zip(flat,cases):
        p,a,analytic = case['profiles'],case['arrays'],case['analytic']
        image = ax.pcolormesh(p['x'],p['s'],np.maximum(p['rho'],1e-5),
            norm=LogNorm(vmin=1e-4,vmax=4),cmap='cividis',shading='auto',rasterized=True)
        ax.plot(analytic['x'],analytic['tau']*case['eos'].sound,'k--',lw=1.6,label='Analytic boundary')
        ax.plot(a['contour_0.001'],a['contour_s'],color='#ff6767',lw=.8,ls=':',label=r'$\rho=10^{-3}$ contour')
        ax.plot(a['liquid_x'],a['liquid_s'],color='white',lw=1.5,label='Liquid-side hydro fit',
                path_effects=[path_effects.Stroke(linewidth=2.5,foreground='#333333'),path_effects.Normal()])
        final = case['records'][-1]
        ax.plot(0.,final['tau_c']*case['eos'].sound,'wo',ms=4)
        label = r'\infty' if case['tag']=='inf' else case['tag']
        ax.set(title=rf'$\gamma={label}$',xlim=(0,1.12),ylim=(0,1.08),xlabel=r'$x/R$',ylabel=r'$c_{s0}\tau/R$')
    for ax in flat[len(cases):-1]:
        ax.set_visible(False)
    overlay = flat[-1]
    for case,color in zip(cases,colors):
        a = case['arrays'];ref=case['analytic']
        label = r'$\infty$' if case['tag']=='inf' else case['tag']
        overlay.plot(ref['x'],ref['tau']*case['eos'].sound,ls='--',lw=1,color=color)
        overlay.plot(a['liquid_x'],a['liquid_s'],lw=1.8,color=color,label=rf'$\gamma=$'+label)
    overlay.set(xlim=(0,1),ylim=(0,1.05),xlabel=r'$x/R$',ylabel=r'$c_{s0}\tau/R$',
                title='Hydro (solid), analytic (dashed)')
    overlay.legend(fontsize=9);overlay.grid(alpha=.2)
    flat[0].legend(fontsize=7,loc='upper right',facecolor='white',framealpha=.9)
    fig.colorbar(image,ax=list(flat[:len(cases)]),label=r'$\rho/\rho_0$ (log scale; clipped above 4)',shrink=.85)
    fig.suptitle('EFP hydrodynamics with a freely determined vacuum region',fontsize=15)
    for suffix in ('png','pdf'):
        fig.savefig(output/('hydro-profiles.'+suffix),dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.2,4.1),constrained_layout=True)
    for case,color in zip(cases,colors):
        records=case['records'];r=records[-1];a=case['arrays'];label=case['tag']
        count=np.array([v['coefficients'] for v in records])/1000
        axes[0].plot(count,[100*v['action_relative_error'] for v in records],'-o',color=color,label=label,ms=4)
        axes[1].plot(count,[100*v['tau_c_relative_error'] for v in records],'-o',color=color,ms=4)
        fit_low,fit_high=r['closing']['selected_range']
        fit_limits=100*(np.array([fit_low,fit_high])/case['eos'].sound/r['analytic_tau_c']-1)
        fit_center=100*r['tau_c_relative_error']
        # All cases have the same final coefficient count. Colored caps
        # make each fitting range visible even where their bars overlap.
        axes[1].errorbar(count[-1],fit_center,
            yerr=np.array([[fit_center-fit_limits[0]],[fit_limits[1]-fit_center]]),
            fmt='none',ecolor=color,elinewidth=1.1,capsize=4,zorder=3)
        axes[2].plot(a['liquid_s']/case['eos'].sound/r['analytic_tau_c'],
            a['liquid_x']-a['analytic_x_at_liquid_times'],color=color,lw=1,label=label)
    axes[0].set(xlabel='Displacement coefficients (thousands)',ylabel='Action excess (%)',title='Action convergence')
    axes[1].set(xlabel='Displacement coefficients (thousands)',ylabel='Closing-time error (%)',title='Closing time and fit spread')
    axes[2].set(xlabel=r'$\tau/\tau_c^{\rm analytic}$',ylabel=r'$(x_{\rm hydro}-x_{\rm analytic})/R$',title='Error on accepted liquid-side fits')
    axes[2].axhspan(-.001,.001,color='.8',alpha=.35)
    for ax in axes:
        ax.axhline(0,color='k',lw=.7);ax.grid(alpha=.2)
    axes[0].legend(title=r'$\gamma$',fontsize=8)
    for suffix in ('png','pdf'):
        fig.savefig(output/('hydro-convergence.'+suffix),dpi=180)
    plt.close(fig)
    summary=[case['records'][-1] for case in cases]
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Production EFP hydrodynamics','',
        'Units: R=rho0=m=hbar=1. The action includes all four quadrants, the vacuum contribution and the leading exterior correction. '
        'The box is 0<=x,cs0*tau<=13 for the default run. Analytic dipole data fix the outer boundary; the internal vacuum boundary is free.','',
        '| gamma | Hydro action | Analytic action | Action error | Hydro tau_c | Analytic tau_c | tau_c error | Selected fit spread |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['gamma']} | {r['action']:.10f} | {r['analytic_action']:.10f} | {100*r['action_relative_error']:+.5f}% | "
            f"{r['tau_c']:.9f} | {r['analytic_tau_c']:.9f} | {100*r['tau_c_relative_error']:+.5f}% | {100*r['closing']['selected_spread_relative']:.5f}% |")
    lines+=['','Closing times are EOS-informed liquid-side extrapolations. The selected spread is the full range of the accepted fits, '
        'not a confidence interval or proof of accuracy. Every tested window, including rejected ones, is saved in each observables.json.', '',
        '| gamma | Last mesh change in tau_c | Qualified extraction | Closing stability <0.1% | Local curve change, max / R | Common-point curve RMS error / R |',
        '|---|---:|---|---|---:|---:|']
    for r in summary:
        c=r.get('change_from_previous',{})
        lines.append(f"| {r['gamma']} | {100*c.get('tau_c_relative',float('nan')):+.5f}% | {r['closing']['qualified']} | "
            f"{r['closing_stability_pass']} | {c.get('curve_max')} | {r['curve_rms_on_common_points']} |")
    lines+=['','A small closing-time error does not establish uniform accuracy of the complete curve. Horizontal fits omit unresolved cusp neighborhoods. '
        'All comparisons retain the raw density contours and fit acceptance flags. Optimization gaps, quadrature differences, density ranges, '
        'runtime and memory are in summary.json and the per-stage records.','',
        f"Total recorded solve time, including coarse preparations: {sum(r['total_solve_seconds'] for r in summary)/60:.1f} minutes. "
        f"Largest process memory high-water mark: {max(r['peak_process_mib'] for r in summary)/1024:.2f} GiB. "
        'These are serial CPU runs; EOS construction and postprocessing are additional.','',
        'The EOS uses direct Bethe quadrature with a validated high-density weak-coupling continuation. Its coefficients are from '
        '[Prolhac (2016)](https://arxiv.org/abs/1610.08912). See eos_validation.json for direct overlap checks.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'production-output')
    p.add_argument('--gamma',nargs='+',type=float,default=[np.inf,10.,1.,.1,.01])
    args=p.parse_args();make_results(args.output,args.gamma)
