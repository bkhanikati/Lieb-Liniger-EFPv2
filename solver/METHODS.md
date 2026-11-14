# Production EFP hydrodynamics for the Lieb–Liniger model

> Note: this is the solver's original method description. Commands below
> assume you are inside `solver/`, and the default couplings differ from the
> paper. For the paper's runs (with `--slit-refinements 3 --slit-add 96`),
> follow the top-level README.md.

One command solves the free-boundary hydrodynamic problem for five
background couplings, generates density/boundary figures, and reports the
action and closing time together with refinement and fitting diagnostics.
No previous study directory or precomputed hydrodynamic field is required.

The default couplings are infinity (Tonks–Girardeau), 10, 1, 0.1 and 0.01.
Units are R=rho0=m=hbar=1. The computational quadrant is
0<=x<=13, 0<=s=cs0*tau<=13. The program uses CPU only; large solves run
sequentially. The tested platform is an Apple M1 Mac with 16 GB RAM.
The standard Unix `resource` module is used, so macOS/Linux are supported.

## Install and run

Use Python 3.11 or a compatible version in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python efp_production.py --gamma inf 10 1 0.1 0.01
python check_efp_production.py
```

The first run constructs and validates the EOS cache, then solves every
coupling from a feasible coarse field. Complete stages are checkpointed.
Running the same command again verifies and reuses them and regenerates
the observables and plots. It does **not** silently recompute completed
optimization stages.

For a genuinely fresh reproduction while retaining the bundled results:

```sh
python efp_production.py --gamma inf 10 1 0.1 0.01 --output fresh-output
python check_efp_production.py --output fresh-output
```

For a small installation test, which is not an accurate calculation:

```sh
python efp_production.py --gamma 0.01 --n 16 --bootstrap 0 --refinements 0 --output smoke-output --solve-only
```

Generate only the figures and observables from completed checkpoints:

```sh
python efp_production_results.py --gamma inf 10 1 0.1 0.01
```

All outputs default to `production-output/` next to the scripts. `--output`
chooses another directory. Do not mix different mesh configurations in one
directory: the program checks the saved configuration and refuses a
mismatch. After an interruption, rerun the same command; completed stages
are retained, but an interrupted Newton stage is restarted from its
preceding completed field. Numerical-library changes may affect last digits
or adaptive decisions near equal monitor values.

## Files to read

| Python file | Role |
|---|---|
| `efp_production.py` | User interface, restart handling, coupling loop and adaptive mesh selection. |
| `efp_production_core.py` | Convex action and feasible sparse Newton minimizer. |
| `efp_production_fem.py` | Quadratic spline discretization, boundary conditions, exact mesh transfer and solve. |
| `efp_production_eos.py` | Shared Bethe EOS table, scaling to arbitrary c, and controlled high-density continuation. |
| `efp_production_reference.py` | Analytic action, dipole and curve, with integral-equation resolution checks. |
| `efp_production_diagnostics.py` | Numerical closing time, threshold contours and liquid-side boundary fits. |
| `efp_production_results.py` | Tables, density maps, comparison plots and convergence summaries. |
| `check_efp_production.py` | Independent thermodynamic, transfer and completed-field checks. |

These eight files import no historical study scripts. They are commented
and can be copied together with requirements.txt. Saved outputs are useful
for inspection and fast reproduction of figures, but are not required for
a fresh solve. No Mathematica installation or network access is needed
after the Python dependencies have been installed.

## Hydrodynamic problem

The unknown is particle displacement u, not velocity:

    rho = u_x,    j = -cs0*u_s,    v = j/rho.
    V(rho) = [epsilon(rho)-epsilon(1)-mu(1)*(rho-1)]/cs0^2.
    S_box = 4*cs0 * integral [u_s^2/(2*u_x) + V(u_x)] dx ds.

The factor four uses the two reflection symmetries. The finite vacuum
contribution V(0) is retained. Spatially nondecreasing spline coefficients
guarantee rho>=0 everywhere; a vanishing log barrier approaches vacuum.
Continuity holds identically in the displacement representation.

At x=0 set u=0. On s=0, 0<=x<=1, set u=0 and fix the first two temporal
coefficient rows there, ensuring u=O(s^2) and an integrable kinetic term.
For x>1 the bottom boundary is free and has the natural zero-current
condition. On the distant top/right edges prescribe the dipole trace
u-x=-2*alpha*x/[x^2+(cs_exact/cs_numerical*s)^2]. Density and velocity are
not prescribed independently there. The analytic internal vacuum boundary
and analytic closing time are never imposed.

The only analytic EFP input to optimization is the outer dipole amplitude.
The sound speed is computed from the EOS. The final action adds the leading
exterior dipole contribution. For a square in exact acoustic coordinates
this is (pi+2)*alpha^2*cs0/L^2; the code accounts for the small difference
between interpolated and directly computed sound speeds.

## Mesh and stopping rules

The initial coarse mesh has 64 graded intervals in each inner coordinate,
plus a geometric outer continuation. Three inexpensive adaptive
preparations add 20 knots per axis each; the last explicitly protects the
numerical closing tip. Bisecting every interval with center below 2 then
creates the fine initial field (`base`). Two subsequent passes add 64
knots per axis, concentrating on the slit endpoint, closing tip and widest
intervals along the numerical interface. Every preceding knot and its
outer trace are preserved. This is the validated preparation sequence,
packaged as one production run.

Tensor splines add complete rows/columns, not isolated triangles. The
finite knot budget keeps memory practical. Selection uses numerical
density/current fields only. No analytic curve or time sets the mesh.

The first solve decreases the barrier from 1e-3 to 1e-9. Later exact knot
transfers use 1e-8 and 1e-9. Sparse Newton solves use diagonal scaling and
symmetric minimum-degree ordering. The final stopping rule combines a
small Newton decrement with a measured convex objective-gap bound and a
1e-6 absolute allowance in the quarter objective for gradient cancellation.
The achieved bound is saved; the allowance is not claimed as the error.
This bound concerns the discrete quadrature problem, not continuum shape
accuracy. Actions are re-integrated with orders 6 and 8 after order-4
optimization. Iteration history retains the legacy `quarter_liquid_action`
key; despite its name, that quantity includes the vacuum contribution.

## EOS at small gamma and high density

The microscopic interaction c stays fixed while density varies. With
rho0=1, c equals the specified background gamma. A single table at c=1
serves every finite coupling by

    rho=c*n,  mu=c^2*M(n),  epsilon=c^3*E(n).

Bethe quadrature order increases with Q/c so the Lorentzian kernel remains
resolved. Cubic Hermite interpolation uses both mu and its derivative,
obtained from the dressed sound speed. Integrating this same interpolant
keeps energy, chemical potential and compressibility consistent.

For local gamma below approximately 0.00197, use the weak-coupling energy
through gamma^(5/2), smoothly matched in mu and mu'. The matching correction
decays at infinity. The expansion coefficients are documented in
[Prolhac, arXiv:1610.08912](https://arxiv.org/abs/1610.08912) and
[Ristivojevic, arXiv:1905.13705](https://arxiv.org/abs/1905.13705).
This is an EOS approximation with direct Bethe overlap checks, not an
unchecked mean-field replacement. `eos_validation.json` records off-grid
and high-density checks using higher quadrature orders. The code never
tries to resolve an arbitrarily narrow high-density kernel at one fixed
quadrature order.

The observables explicitly report `direct_bethe_table_density_max` and
`high_density_eos_tail_used`. The legacy stage flag
`density_within_eos_table` includes the validated continuation; it does
not mean that every density was inside the direct Bethe table.

The analytic reference is independently checked by doubling the number
of response-equation modes. A separate higher-order Bethe background
check verifies its sound speed. These references are then compared with
hydro results; only the allowed far-field data enter the hydro solve.

## Closing-time and boundary extraction

A finite mesh and barrier smear the density onset. A fixed small-density
contour is saved as a diagnostic, but the closing time is extrapolated
from the liquid side using the regular modes of the hodograph equation.
If a(r)=mu'(r)/r, then

    T_rr + 2*T_r/r + a(r)*T_yy = 0,
    F0=1,
    F_n(r) = integral_0^r (1-z/r)*mu'(z)*F_(n-1)(z) dz.

Thus T=T0-F1*T0''+F2*T0''''-..., and F1=epsilon(r)/r. Fit the axis time
against F0,F1,F2 and F0,F1,F2,F3 with a free intercept. At TG these are
polynomials in rho^2; at finite coupling they retain the EOS crossover.

The same density-window catalogue is tested for all couplings, with extra
c-scaled dilute windows at weak coupling for sensitivity diagnostics.
Windows inside the rho=.02 onset exclusion band cannot qualify. A window is admissible when it
spans at least six mesh intervals, begins at least two intervals beyond
the rho=.02 crossing, and the two fit orders agree within .05% of the
inferred closing time. The estimate is the median of both orders in up to
three lowest-density admissible windows. All rejected windows and their
reasons remain in the data. If no window qualifies, the estimate is
explicitly marked provisional. No analytic closing time selects a window.

For horizontal boundary fits, the local fixed-time expansion is
x=x_edge-mu(rho)*T0'(y_edge)+O(rho^4). Use actual liquid mesh nodes, varying
the skipped transition cells and fitting interval. The central estimate
is a median of three linear chemical-potential fits; quadratic fits
measure sensitivity. Fits failing the same monotonicity, density-range
and extrapolation checks as in the validated boundary procedure are
omitted. There is no invented interpolation through unresolved cusp
neighborhoods. The slit endpoint is known; the axis closing time is fitted
separately.

Selected fitting spreads are diagnostic ranges, not confidence intervals.
The program separately reports the last mesh change, comparison with the
analytic answer, accepted curve coverage, and the full fitting catalogue.
It does not infer uniform curve accuracy from a small action or tip error.

## Outputs

- `RESULTS.md`, `summary.json`: final numbers and numerical qualifications.
- `hydro-profiles.png/.pdf`: five density maps and a combined boundary plot.
- `hydro-convergence.png/.pdf`: action, closing time and spatial curve errors.
- `gamma_*/reference.json/.npz`: analytic values and their resolution checks.
- `gamma_*/coarse`, `bootstrap_*`, `base`, `refine_*` JSON/NPZ pairs:
  iteration records, axes, coefficients, density and current fields.
- `gamma_*/observables.json`, `*_observables.json`, `*_curves.npz`:
  extraction details, fitting windows and successive-mesh comparisons.
- `gamma_*/profiles.npz`: sampled density, current and velocity. Velocity
  is NaN for rho<=1e-4, where division by tiny density is unreliable.
- `eos_validation.json`, `production_checks.json`: numerical checks.

The density figures clip their color scale above rho/rho0=4 so the liquid
and vacuum edge remain visible; the saved density values are not clipped.
The plotted time is acoustic time s=cs0*tau. Tables give physical tau_c in
the stated R=rho0=m=hbar=1 units.
