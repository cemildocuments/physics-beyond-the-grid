# Physics beyond the grid

Reproducible experiments on separating continuum coefficients from numerical artifacts in simulation-based equation identification.

The project includes periodic transport, Burgers and logistic reaction–diffusion simulations, weak-form regression, polynomial refinement, independent spectral references, two-dimensional diffusion-tensor identification, and rank and shared-bias controls.

## Installation

Use Python 3.12 and create an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The implementation uses double-precision CPU arithmetic. Eight to twelve CPU processes are suitable; a single process also works. An 8 GB memory allocation is ample for the default configuration. Numba compiles the integration routines on their first use. Dependency installation requires internet access; the experiments generate their own data locally.

## Reproduce and verify

From the project directory:

```sh
python run_all.py --workers 12
```

This runs the analytic and algebraic tests, regenerates all experimental conditions into `reproduced/`, creates numerical summary tables, and compares the regenerated estimates with the reference results. The original `results/` directory supplies the numerical reference for this comparison. Change `--workers` to control CPU concurrency and `--output` to choose a different destination.

The numerical comparison checks coefficients, uncertainty intervals, transfer errors, tensor estimates and diagnostic results with absolute tolerance 1e-8 and relative tolerance 1e-7. Runtime and completion order may differ across executions. The original eight-process and twelve-process runs produced identical coefficient vectors in the recorded software environment.

## Contents

| Path | Purpose |
| --- | --- |
| `src/core.py` | One-dimensional simulators, weak operators, joint and coefficient extrapolation, uncertainty estimates, and independent reference solutions. |
| `src/run_benchmark.py` | Main factorial benchmark with deterministic seeds. |
| `src/extra_experiments.py` | Two-dimensional modal simulations, tensor fits, rank experiments, shared-bias control and convergence checks. |
| `src/export_examples.py` | Regenerates the representative raw field and per-grid coefficient table. |
| `src/analyze.py` | Complete numerical summaries and decision counts. |
| `run_all.py` | End-to-end reproduction and validation. |
| `verify_results.py` | Comparison of reproduced and reference numerical values. |
| `tests/test_core.py` | Analytic-symbol, weak-form, rank, exact-recovery and extrapolation-equivalence tests. |
| `results/main.jsonl` | All 9,720 one-dimensional coefficient estimates, interval bounds, decisions and transfer metrics. |
| `results/tensor.json` | All 1,200 diffusion-tensor estimates. |
| `results/diagnostics.json` | Rank, shared-bias and independent convergence results. |
| `results/summary.csv` | Median and quartile errors, grouped by method and noise. |
| `results/decisions.csv` | Positive, negligible and unresolved decision counts. |
| `results/tensor_summary.csv` | Tensor-error and eigenvalue-ratio summaries. |
| `results/refinement_curve.csv` | Per-grid diffusion estimates for the representative refinement example. |
| `data/representative_burgers.npz` | Field `u`, coordinates `x,t`, and generating values `nu,a,seed`. |

## Experimental design

The main benchmark has 40 trial blocks, three equation families, three diffusivities (0, 0.002, 0.02) and three observation-noise ratios (0, 0.001, 0.01): 1,080 datasets. Each dataset uses eight initial conditions at five grids (32, 48, 64, 96, 128 points) and two dissipation settings. All methods use a 12-term operator library. The two low-order settings belong to the same flux family. A fourth-order discretization supplies an independent accuracy control. Transfer uses withheld initial conditions and independent spectral solutions.

`kind` is 0 for transport, 1 for Burgers and 2 for logistic reaction with transport. `trial` is the seeded trial block. `noise` is the ratio of observation-noise standard deviation to clean-field standard deviation. `nu` is physical diffusion. `coeff` follows `LABELS` in `src/core.py`. `coefficient_error` is relative Euclidean coefficient error; `transfer_error` is the relative weak-equation residual on independent reference trajectories.

The refinement envelope combines approximate intervals from quadratic, cubic and coarse-grid-omitted fits. Its decisions use a fixed negligible-effect tolerance of 0.0005. An unresolved result remains an unresolved result in every summary. These intervals are empirical sensitivity measures rather than guarantees of physical truth.

The tensor experiment uses two known diffusion matrices, four initial conditions per environment and 40 trial blocks per matrix. Exact semidiscrete Fourier evolution separates spatial artifacts from temporal integration error. `case=0` is isotropic diffusion; `case=1` is oblique anisotropic diffusion. `D` is the full estimated tensor, `error` is relative Frobenius error, and `ratio` is its eigenvalue ratio when positive definite.

All fields are nondimensional. The data are generated independently on each grid. Paired seeds across conditions preserve controlled comparisons. Clean reference results and complete trial-level estimates are included for inspection and reuse under the MIT license.
