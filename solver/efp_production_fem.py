"""Finite-element implementation of free-boundary EFP hydrodynamics.

This production library contains the verified quadratic-spline solver,
without historical study drivers or saved-data dependencies. Its numerical
routines are unchanged from the preceding validated calculations.
"""
from __future__ import annotations
import platform
import resource
import time
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.sparse.linalg import spsolve
from efp_production_core import geometric_tail, minimize, action_terms, exterior_angular_factor


def knots(axis, slit=False):
    """Degree-two clamped knots; a repeated x=1 knot permits the slit cusp."""
    inner = np.asarray(axis)[1:-1]
    if slit:
        inner = np.sort(np.r_[inner, 1.])  # C0 at the singular slit endpoint.
    return np.r_[np.repeat(axis[0], 3), inner, np.repeat(axis[-1], 3)]


def greville(t):
    """Collocation sites for quadratic splines (averages of two knots)."""
    return (t[1:-2]+t[2:-1])/2


def coefficient_derivative(t):
    """Map quadratic coefficients to linear derivative coefficients.

    Every derivative basis function is nonnegative. Thus nonnegative
    scaled adjacent coefficient differences guarantee rho=u_x>=0 globally.
    This sufficient condition is more restrictive than pointwise positivity
    at a fixed resolution; refinement must check the remaining mesh bias.
    """
    n = len(t)-3
    scale = 2/(t[3:-1]-t[1:-3])
    return sparse.diags((-scale, scale), (0, 1), shape=(n-1, n), format='csr')


def basis(t, points):
    """Sparse displacement and first-derivative evaluation matrices."""
    b = sparse.csr_matrix(BSpline.design_matrix(points, t, 2))
    d = sparse.csr_matrix(BSpline.design_matrix(points, t[1:-1], 1)) @ coefficient_derivative(t)
    return b, d.tocsr()


def gauss_axis(axis, order):
    """Interior Gauss nodes and physical weights on each mesh interval."""
    z, w = np.polynomial.legendre.leggauss(order)
    h = np.diff(axis)
    return (axis[:-1,None]+h[:,None]*(z+1)/2).ravel(), (h[:,None]*w/2).ravel()


def operators(x, s, order):
    """Tensor maps to u_x,u_s with x varying fastest in flattened arrays."""
    tx, ts = knots(x, True), knots(s)
    xq, wx = gauss_axis(x, order)
    sq, ws = gauss_axis(s, order)
    bx, dx = basis(tx, xq)
    bs, ds = basis(ts, sq)
    gx = sparse.kron(bs, dx, format='csr')
    gt = sparse.kron(ds, bx, format='csr')
    return gx, gt, np.kron(ws, wx), (tx, ts)


def asymptotic_u(x, s, alpha, sound_ratio=1.):
    """Background plus known far-field dipole, used only at large radius.

    s uses the tabulated EOS sound speed; sound_ratio converts it to the
    independently calculated exact acoustic coordinate in the denominator.
    """
    r2 = np.asarray(x)**2+(sound_ratio*np.asarray(s))**2
    return np.asarray(x)-np.divide(2*alpha*np.asarray(x), r2,
                                  out=np.zeros(np.broadcast_shapes(np.shape(x),np.shape(s))), where=r2>0)


def constraints(tx, ts, extent, alpha, sound_ratio=1.):
    """Feasible initial field, fixed boundary coefficients and positivity map.

    On x=0 set u=0. On the slit s=0, 0<=x<=1 set u=0 and also u_s=0,
    by fixing the first two time coefficient rows: u=O(s**2) there avoids
    a nonintegrable kinetic term. For x>1 the lower edge remains free and
    zero current is its natural variational condition. Outer u is fixed by
    the asymptotics; rho and velocity are not prescribed independently.
    """
    xg, sg = greville(tx), greville(ts)
    nx, ns = len(xg), len(sg)
    xx, ss = np.meshgrid(xg, sg)
    full = xx-np.minimum(xx, 1.)*.3/(ss+.3)
    fixed = np.zeros_like(full, dtype=bool)
    fixed[:,0] = True
    fixed[:,-1] = True
    fixed[-1,:] = True
    fixed[:2,xg<=1] = True
    full[:2,xg<=1] = 0.
    # Consistent Dirichlet traces: interpolate the same far field on both
    # outer edges. No normal derivative is independently prescribed.
    bx, _ = basis(tx, xg)
    bs, _ = basis(ts, sg)
    full[-1,:] = spsolve(bx, asymptotic_u(xg, extent, alpha, sound_ratio))
    full[:,-1] = spsolve(bs, asymptotic_u(extent, sg, alpha, sound_ratio))
    full[:,0] = 0.
    c = sparse.kron(sparse.eye(ns), coefficient_derivative(tx), format='csr')
    # Integral of the nonnegative density basis, so sum(cw)=box area.
    wx = (tx[3:-1]-tx[1:-3])/2
    ws = (ts[3:]-ts[:-3])/3
    cw = np.kron(ws, wx)
    return fixed.ravel(), full.ravel(), c, cw


def initial_axes(n=64, extent=13.):
    """Endpoint-graded inner axes and a geometric continuation to the box."""
    z = np.linspace(0, 1, n+1)
    xi = (1-np.cos(np.pi*z))/2
    si = z*z
    x = np.r_[xi, 1+geometric_tail(extent-1, np.diff(xi)[-1], n//2)]
    s = np.r_[si, 1+geometric_tail(extent-1, np.diff(si)[-1], n//2)]
    x[-1] = s[-1] = extent
    return x, s


def evaluate(fields, x, s):
    """Return rho and u_s on a tensor grid, with array shape (len(s),len(x))."""
    bx, dx = basis(fields['x_knots'], np.asarray(x))
    bs, ds = basis(fields['s_knots'], np.asarray(s))
    u = fields['coefficients']
    return np.asarray(bs@(dx@u.T).T), np.asarray(ds@(bx@u.T).T)


def insert_coefficients(fields, new_x, new_s):
    """Exact knot insertion: a nested mesh retains every preceding field."""
    tx, ts = fields['x_knots'], fields['s_knots']
    ux = BSpline(tx, fields['coefficients'].T, 2, axis=0)
    for x in np.setdiff1d(new_x, fields['x_axis']):
        ux = ux.insert_knot(float(x))
    us = BSpline(ts, ux.c.T, 2, axis=0)
    for s in np.setdiff1d(new_s, fields['s_axis']):
        us = us.insert_knot(float(s))
    return us.c


def solve_free(x, s, eos, alpha, exact_sound, order=4, initial=None, barriers=None, gap_atol=1e-9):
    """Minimize over density/current and the unknown vacuum region together.

    The variables are displacement coefficients; continuity is automatic.
    No internal boundary is supplied. Nearly flat u identifies the vacuum
    as the positivity constraint becomes active when the barrier vanishes.
    The action includes that region and is multiplied by four for symmetry.

    initial, if supplied, must be on a nested mesh of the SAME box. Exact
    knot insertion preserves its field and external trace. The order+2 and
    order+4 rechecks integrate the final field more accurately; they do not
    reoptimize it. The exterior correction is only the leading dipole tail.
    """
    start = time.perf_counter()
    extent = float(x[-1])
    gx, gs, w, (tx, ts) = operators(x, s, order)
    fixed, full, c, cw = constraints(tx, ts, extent, alpha, exact_sound/eos.sound)
    if initial is not None:
        transferred = insert_coefficients(initial, x, s).ravel()
        # Adaptation leaves outer cells untouched, but the boundary spline
        # representation is re-interpolated. Verify this trace change is tiny.
        transfer_error = float(np.max(abs(transferred[fixed]-full[fixed])))
        if transfer_error > 1e-6:
            raise RuntimeError(f'Boundary transfer changed by {transfer_error}')
        # Preserve the original boundary spline exactly as well as the field:
        # only the interior approximation space changes between mesh passes.
        full = transferred
    else:
        transfer_error = 0.
    upper = np.full_like(full, extent)
    options = {} if barriers is None else dict(barriers=barriers)
    options['gap_atol'] = gap_atol
    full, history, opt = minimize(gx, gs, w, fixed, full, c, cw, eos, upper, **options)
    rho, k = gx@full, gs@full
    action = float(4*eos.sound*(w@action_terms(rho,k,eos)))
    fields = dict(x_axis=x, s_axis=s, x_knots=tx, s_knots=ts,
                  coefficients=full.reshape(len(ts)-3,len(tx)-3))
    refined = {}
    maxrho = float(rho.max())
    for q in (order+2, order+4):
        qx, qw = gauss_axis(x,q)
        qs, sw = gauss_axis(s,q)
        rr, kk = evaluate(fields,qx,qs)
        if rr.min() <= 0:
            raise RuntimeError('Nonpositive density at recheck points')
        refined[str(q)] = float(4*eos.sound*(sw@action_terms(rr,kk,eos)@qw))
        maxrho = max(maxrho,float(rr.max()))
    tail = float(alpha**2*exact_sound*exterior_angular_factor(extent,extent*exact_sound/eos.sound))
    opt['full_action_gap_bound_for_quadrature_problem'] = 4*eos.sound*opt['quarter_liquid_gap_bound']
    xc, sc = (x[:-1]+x[1:])/2, (s[:-1]+s[1:])/2
    rc, kc = evaluate(fields,xc,sc)
    fields.update(rho_cell=rc, current_cell=-eos.sound*kc, x_centers=xc, s_centers=sc)
    result = dict(model=eos.name, extent=extent, sound=eos.sound, alpha=alpha,
                  exact_sound=exact_sound, cells=(len(x)-1)*(len(s)-1),
                  coefficients=full.size, x_intervals=len(x)-1, s_intervals=len(s)-1,
                  quadrature_order=order, action_inside=action, exterior_action=tail,
                  corrected_action=action+tail, quadrature_rechecks_inside=refined,
                  corrected_action_rechecked=refined[str(order+4)]+tail,
                  optimization=opt, density_max_checked=maxrho,
                  density_within_eos_table=bool(not hasattr(eos,'rmax') or maxrho<eos.rmax),
                  coefficient_density_min=float((c@full).min()),
                  boundary_transfer_max_error=transfer_error,
                  elapsed_seconds=time.perf_counter()-start,
                  process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                    (1024**2 if platform.system()=='Darwin' else 1024), eos_metadata=eos.metadata)
    if not result['density_within_eos_table']:
        raise RuntimeError('Density exceeds EOS table')
    return result, fields, history


def adapt_axes(fields, add=20, focused_tip=False):
    """Nested refinement: reserve cells for the interface as well as gradients.

    All decisions use numerical fields. The analytic arctic curve is absent.
    Outer layers beyond 2 are left unchanged. Keeping all existing nodes
    protects the endpoint and avoids sacrificing the closing-point resolution.
    """
    x, s = fields['x_axis'], fields['s_axis']
    xc, sc = fields['x_centers'], fields['s_centers']
    r, k = fields['rho_cell'], fields['current_cell']
    # Soften the endpoint divergence to prevent it taking the whole budget.
    rs, rx = np.gradient(np.arcsinh(r),sc,xc,edge_order=2)
    ks, kx = np.gradient(np.arcsinh(k),sc,xc,edge_order=2)
    gradx = np.sqrt(np.sum((rx*rx+kx*kx)*np.diff(s)[:,None],axis=0))*np.diff(x)
    grads = np.sqrt(np.sum((rs*rs+ks*ks)*np.diff(x)[None,:],axis=1))*np.diff(s)
    wet = r > .02
    interface = binary_dilation(wet,iterations=2)&binary_dilation(~wet,iterations=2)
    interface &= (sc[:,None]<1.5)&(xc[None,:]<1.5)
    edgex = np.sum(interface*np.diff(s)[:,None],axis=0)*np.diff(x)
    edges = np.sum(interface*np.diff(x)[None,:],axis=1)*np.diff(s)
    # Explicitly protect the emerging tip on the symmetry axis. Its density
    # gradients need not be large, and the interface's horizontal width shrinks.
    axis_r, _ = evaluate(fields,np.array([0.]),sc)
    crossings = np.flatnonzero((axis_r[:-1,0]<.02)&(axis_r[1:,0]>=.02))
    tip = float(sc[crossings[0]]) if len(crossings) else None
    selected = []
    monitors = dict(gradient_x=gradx,gradient_s=grads,interface_x=edgex,interface_s=edges)
    for axis, grad, edge, coordinate in ((x,gradx,edgex,'x'),(s,grads,edges,'s')):
        center = (axis[:-1]+axis[1:])/2
        eligible = center < 2.
        choose = set()
        if tip is not None:
            if focused_tip and coordinate=='s':
                # Refining the widest cells in a broad band can repeatedly
                # miss the actual tip cell. Explicitly split the cells that
                # bracket the small-density crossing and their neighbours.
                for threshold in (1e-4,.001,.02):
                    above=np.flatnonzero(axis_r[:,0]>=threshold)
                    if len(above):
                        j=int(above[0])
                        choose.update(range(max(0,j-2),min(len(center),j+2)))
            else:
                protect = ((center<.12) if coordinate=='x' else abs(center-tip)<.10)&eligible
                candidates = np.flatnonzero(protect)
                for i in candidates[np.argsort(np.diff(axis)[candidates])[-min(4,len(candidates)):]]:
                    choose.add(int(i))
        for score, quota in ((edge,add//2),(grad,add)):
            ranking = np.argsort(np.where(eligible,gaussian_filter1d(score,.7),-np.inf))[::-1]
            target = min(add, len(choose)+quota)
            for i in ranking:
                if len(choose)>=target or not eligible[i]:
                    break
                choose.add(int(i))
        nodes = np.sort(np.r_[axis,center[sorted(choose)]])
        selected.append(nodes)
        monitors[f'inserted_{coordinate}'] = center[sorted(choose)]
    monitors['numerical_tip_monitor'] = np.nan if tip is None else tip
    return tuple(selected), monitors
