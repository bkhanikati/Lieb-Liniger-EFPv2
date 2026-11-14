"""Coupling-independent extraction of numerical EFP observables.

Extraction receives the numerical field and EOS only. Analytic curves
and actions enter later, when results are compared and plotted.
"""
import numpy as np
from scipy.optimize import brentq
from efp_production_fem import evaluate
LIQUID_WINDOWS = ((3, 8), (4, 10), (5, 12))


def chemical_potential(eos, rho):
    """Remove the background subtraction from V', including its scale."""
    return eos.sound**2*(eos.derivative(rho)-eos.derivative(0.))


def radial_basis(eos, rho, n):
    """Regular radial modes of the hodograph PDE, evaluated by quadrature.

    For TG, F_n=pi**(2*n)*rho**(2*n)/(2*n+1)!; hence this reduces to
    the familiar polynomial in rho**2. For LL the EOS corrections are
    retained instead of assuming an even polynomial in rho.
    """
    rho = np.asarray(rho)
    if n == 0:
        return np.ones_like(rho)
    if n == 1:
        if hasattr(eos, 'energy'):
            return (eos.energy(rho)-eos.energy(0.))/rho
        return np.pi**2*rho**2/6
    z, w = np.polynomial.legendre.leggauss(32)
    z, w = (z+1)/2, w/2
    r = rho[..., None]*z
    return rho*np.sum(w*(1-z)*eos.sound**2*eos.second(r)*radial_basis(eos, r, n-1), axis=-1)


def intercept_fit(columns, coordinate):
    """Scale columns before least squares; return intercept and residual."""
    matrix = np.column_stack(columns)
    scale = np.max(abs(matrix), axis=0)
    coeff = np.linalg.lstsq(matrix/scale, coordinate, rcond=None)[0]/scale
    residual = matrix@coeff-coordinate
    return float(coeff[0]), float(np.sqrt(np.mean(residual**2)))


def liquid_curve(fields, eos, times=None):
    """Horizontal liquid-side extrapolations with explicit validity flags.

    From the first knot with rho>=.02, skip 3, 4 or 5 more knots and fit
    over 5, 6 or 7 nodes respectively. This avoids fitting the artificially
    tiny density in the discrete vacuum. Linear and quadratic fits in mu
    expose sensitivity to curvature and mesh oscillations. The reported
    central curve is the median of the three LINEAR fits, chosen before
    refinement. Quadratic fits contribute to the sensitivity envelope.

    A fit is rejected if its sample is nonmonotone, exceeds rho=.8,
    extrapolates by more than twice its sampled spatial width, or leaves
    [0,1]. These reject unresolved cusp neighborhoods rather than filling
    them with an analytic curve. No error estimate follows merely from a
    small least-squares residual.
    """
    times = np.linspace(.005, 1.05, 420) if times is None else np.asarray(times)
    x = fields['x_axis']
    rho = evaluate(fields, x, times)[0]
    candidates = np.full((len(times), len(LIQUID_WINDOWS), 2), np.nan)
    sampled_rho = np.full((len(times), len(LIQUID_WINDOWS), 2), np.nan)
    for row, r in enumerate(rho):
        if r[0] >= .02:
            continue
        crossing = np.flatnonzero(r >= .02)
        if not len(crossing):
            continue
        j = int(crossing[0])
        for window, (a, b) in enumerate(LIQUID_WINDOWS):
            xx, rr = x[j+a:j+b], r[j+a:j+b]
            if len(xx) != b-a or np.any(np.diff(rr) <= 0) or rr[-1] > .8:
                continue
            mu = chemical_potential(eos, rr)
            sampled_rho[row, window] = rr[[0, -1]]
            for degree in (1, 2):
                intercept, _ = intercept_fit([mu**p for p in range(degree+1)], xx)
                if 0 <= intercept <= 1 and 0 <= xx[0]-intercept <= 2*(xx[-1]-xx[0]):
                    candidates[row, window, degree-1] = intercept
    center, low, high = [np.full(len(times), np.nan) for _ in range(3)]
    for row, values in enumerate(candidates):
        # Require every linear window for a central estimate. Preserve all
        # individual fits in the output even when this requirement fails.
        if np.all(np.isfinite(values[:, 0])):
            center[row] = np.median(values[:, 0])
            low[row], high[row] = np.nanmin(values), np.nanmax(values)
    return dict(s=times, x=center, fit_low=low, fit_high=high,
                fit_candidates=candidates, sampled_density=sampled_rho)


def axis_crossing(fields, density):
    axis = fields['s_axis']
    rho = evaluate(fields, [0.], axis)[0][:,0]
    candidates = np.flatnonzero(rho >= density)
    if not len(candidates) or candidates[0] == 0:
        raise ValueError(f'Axis crossing not bracketed at density {density}')
    j = candidates[0]
    return float(brentq(lambda s:evaluate(fields,[0.],[s])[0][0,0]-density,
                        axis[j-1],axis[j],xtol=2e-13))


def closing_time(fields, eos, gamma):
    """Fit regular EOS modes and select resolved, stable liquid windows.

    Test the same catalogue for all couplings, supplemented by c-scaled
    low-density windows at weak coupling. A window is admissible only if
    it spans >=6 mesh intervals, starts >=2 intervals beyond rho=.02,
    and orders 2 and 3 agree within .05% of the inferred closing time.
    The reported estimate is the median of both orders in the three
    lowest-density admissible windows (or fewer if fewer qualify).
    No comparison with the analytic closing time enters this selection.
    If no window qualifies, return a flagged provisional estimate.
    """
    windows = [(.01,.04),(.02,.08),(.04,.12),(.06,.16),(.08,.20),(.10,.24),(.12,.28)]
    if gamma < 1:
        windows += [(a*gamma,b*gamma) for a,b in ((.04,.12),(.08,.2),(.12,.28))]
    windows = sorted(set(windows),key=lambda w:w[1])
    axis = fields['s_axis']
    origin = axis_crossing(fields,.02)
    origin_index = np.interp(origin,axis,np.arange(len(axis)))
    fits, choices = [], []
    for low, high in windows:
        density = np.linspace(low,high,31)
        positions = np.array([axis_crossing(fields,r) for r in density])
        indices = np.interp(positions[[0,-1]],axis,np.arange(len(axis)))
        span, clearance = float(indices[1]-indices[0]),float(indices[0]-origin_index)
        columns = [radial_basis(eos,density,n) for n in range(4)]
        pair = []
        for degree in (2,3):
            intercept,residual = intercept_fit(columns[:degree+1],positions)
            record = dict(low=low,high=high,degree=degree,scaled_tip=intercept,
                physical_tip=intercept/eos.sound,fit_rms=residual,mesh_span=span,
                liquid_clearance_cells=clearance,resolved=bool(span>=3))
            fits.append(record);pair.append(record)
        center = np.mean([f['scaled_tip'] for f in pair])
        disagreement = abs(pair[1]['scaled_tip']-pair[0]['scaled_tip'])/max(center,1e-12)
        admissible = bool(span>=6 and clearance>=2 and disagreement<=5e-4 and center>0)
        for f in pair:
            f.update(order_relative_difference=float(disagreement),admissible=admissible)
        if admissible:
            choices.append(pair)
    selected = [f for pair in choices[:3] for f in pair]
    qualified = bool(selected)
    if not selected:
        # Always expose failure of the acceptance test; a provisional value
        # allows plots and a reproducible report instead of silently failing.
        options = [f for f in fits if f['resolved'] and f['scaled_tip']>0]
        if not options:
            options = [f for f in fits if f['scaled_tip']>0]
        selected = sorted(options,key=lambda f:(f['order_relative_difference'],-f['mesh_span']))[:2]
    values = [f['scaled_tip'] for f in selected]
    all_resolved = [f['scaled_tip'] for f in fits if f['resolved'] and f['scaled_tip']>0]
    for f in fits:
        f['selected'] = any(f is item for item in selected)
    value = float(np.median(values))
    return dict(scaled_tip=value,physical_tip=value/eos.sound,qualified=qualified,
        selected_range=[float(min(values)),float(max(values))],
        selected_spread_relative=float(np.ptp(values)/value),
        all_resolved_range=[float(min(all_resolved)),float(max(all_resolved))] if all_resolved else None,
        fits=fits,method='median over orders 2/3 in up to three lowest-density admissible EOS-mode windows')


def extract(fields,eos,gamma):
    """Closing time, raw threshold contours and liquid-side curve fits."""
    closing = closing_time(fields,eos,gamma)
    s = np.linspace(0.,1.15,801)
    x = np.unique(np.r_[np.linspace(0.,1.1,2201),fields['x_axis'][fields['x_axis']<1.1]])
    rho = evaluate(fields,x,s)[0]
    arrays = dict(contour_s=s)
    thresholds = {}
    for cutoff in (1e-4,1e-3,1e-2):
        first = np.argmax(rho>=cutoff,axis=1)
        curve = x[first].copy()
        rows = np.flatnonzero(first>0)
        left = first[rows]-1
        curve[rows] = x[left]+(cutoff-rho[rows,left])/(rho[rows,first[rows]]-rho[rows,left])*(x[first[rows]]-x[left])
        curve[0] = 1.
        tip = axis_crossing(fields,cutoff)
        key = f'{cutoff:g}'
        arrays['contour_'+key] = curve
        thresholds[key] = dict(scaled_tip=tip,physical_tip=tip/eos.sound)
    liquid = liquid_curve(fields,eos,times=np.linspace(.002,1.05,500))
    arrays.update({'liquid_'+key:value for key,value in liquid.items()})
    return dict(closing=closing,thresholds=thresholds), arrays
