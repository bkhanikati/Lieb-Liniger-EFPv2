# Emptiness formation in quantum fluids: hydrodynamic numerics

Code and data for the numerical results in "Alexander G. Abanov and Boris A. Khanikati, *Emptiness formation in quantum fluids: a hydrodynamic solution*."
> 
The solver minimizes the dispersionless hydrodynamic action of the Lieb–Liniger
gas for the emptiness formation probability (EFP) instanton. Methods are described in Appendix D of the paper
and in more detail in [solver/METHODS.md](solver/METHODS.md).

## Contents

| Path | Description|
|---|---|
| `solver/efp_production.py` | Command line, mesh schedule, restarts |
| `solver/efp_production_core.py` | Discrete action, log-barrier sparse Newton minimizer |
| `solver/efp_production_fem.py` | Quadratic tensor B-splines, positivity constraints, mesh adaptation |
| `solver/efp_production_eos.py` | Bethe-ansatz equation of state table and weak-coupling continuation |
| `solver/efp_production_reference.py` | Analytic far-field amplitude, action and vacuum boundary |
| `solver/efp_production_diagnostics.py` | Vacuum-boundary and closing-time extraction |
| `solver/efp_production_results.py` | Observables, convergence records, field profiles |
| `solver/check_efp_production.py` | EOS, spline and normalization checks |
| `tools/plot_combined.py` | 3D density surfaces and density maps (density figure) |
| `tools/plot_boundary.py` | Numerical vs analytic vacuum boundary (boundary figure) |
| `tools/plot_density.py`, `tools/plot_surface.py` | Panels used by `plot_combined.py`; also runnable alone |
| `tools/table.py` | Rows of the convergence table |
| `results/slit-4g/` | Summary output of the runs used in the paper (see below) |

## Install

Python ≥ 3.11 on Linux or macOS (the solver uses the Unix `resource`
module; on Windows use WSL).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

## Reproduce the paper

**Table only (seconds, uses the included results):**

```sh
python tools/table.py results/slit-4g
```

**Boundary figure (seconds, uses the included results):**

```sh
python tools/plot_boundary.py results/slit-4g
```

**Full calculation.** The paper uses γ₀ = ∞, 100, 10, 1, 0.01:

```sh
python solver/efp_production.py --gamma inf 100 10 1 0.01 \
    --slit-refinements 3 --slit-add 96 --output runs/slit-4g
```

All other options are at their defaults (64 + 32 coarse intervals per axis,
box L = 13, three bootstrap passes, two interface refinements of 64 knots).
Each coupling took 40–75 minutes of solve time on one CPU thread, with peak
memory of about 9 GiB; the equation-of-state table is built once on the first
finite-coupling run. Completed stages are checkpointed, so rerunning the same
command resumes after an interruption.

The run writes the field of every stage, `profiles.npz`, observables and
convergence records into `runs/slit-4g/gamma_*/`. The density figure needs
these full fields:

```sh
python tools/plot_combined.py runs/slit-4g --gamma 100 1 0.01 --contours
python tools/plot_boundary.py runs/slit-4g
python tools/table.py runs/slit-4g
python solver/check_efp_production.py --output runs/slit-4g
```

## Included results

`results/slit-4g/` holds the small outputs of the paper's runs, enough to
check every number in the table and the boundary figure without rerunning:

- `eos_table.npz`, `eos_validation.json`: the equation-of-state table and its
  check against independent Bethe quadratures.
- `summary.json`: final-mesh observables for every coupling.
- `gamma_*/config.json`: solver options of the run.
- `gamma_*/observables.json`, `convergence.json`: action, closing time,
  boundary errors and solver diagnostics on every mesh.
- `gamma_*/reference.json`, `reference.npz`: the analytic solution.
- `gamma_*/slit_3_curves.npz`: the extracted numerical vacuum boundary on the
  final mesh.

The full fields (about 20 MB per coupling for the final stage and
`profiles.npz`) are not included; rerun the solver to regenerate them.
<!-- TODO: or link a Zenodo record holding runs/slit-4g -->

## Units and conventions

R = ρ₀ = m = ħ = 1, so the microscopic coupling is c = γ₀. Time is reported
as s = c_s0 τ. Action errors are relative to the analytic action; boundary
errors are |x_num − x_analytic|/R at matching physical times.

## License and citation

<!-- TODO: license agreed with all authors of the solver code -->
If you use this code, please cite the paper above.
