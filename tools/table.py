"""LaTeX rows of the convergence table (Table "hydro-convergence" in the paper).

Usage: python tools/table.py results/slit-4g

Compares the final mesh (slit_3) with the preceding one (slit_2) for each
coupling. Reads only <run>/gamma_*/observables.json.
"""
import json
import sys
from pathlib import Path

GAMMAS = ('inf', '100', '10', '1', '0.01')


def percent(value):
    text = f'{100*value:.4f}'
    return '$<0.0001$' if float(text) == 0 else text


def main(run):
    run = Path(run)
    for tag in GAMMAS:
        recs = {r['stage']: r for r in json.loads((run/f'gamma_{tag}'/'observables.json').read_text())['convergence']}
        last, prev = recs['slit_3'], recs['slit_2']
        cells = [r'$\infty$' if tag == 'inf' else f'${tag}$',
                 percent(abs(last['action']/prev['action']-1)),
                 percent(last['action_relative_error']),
                 percent(abs(last['change_from_previous']['tau_c_relative'])),
                 f"{1e3*last['curve_rms_error']:.3f}",
                 f"{1e3*last['curve_max_error']:.3f}",
                 f"{100*last['curve_valid_time_fraction']:.1f}"]
        print(' & '.join(cells)+r' \\'+f"   % {last['coefficients']} vs {prev['coefficients']} coefficients")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/slit-4g')
