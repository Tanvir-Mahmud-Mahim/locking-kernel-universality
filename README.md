# lockkernel

Reference implementation for **a single oscillator response sets the
exponents of synchronization transitions**.

A population of oscillators with spread frequencies begins to synchronize when
the coupling passes a threshold, and the order parameter then grows from zero as
a power of the distance above it. That power has been derived separately for
each case in which it has been asked. This code shows that the cases are one
statement, and that the quantity deciding the exponent is the **locking kernel**
`W(u)`: the time averaged projection a single oscillator keeps on the collective
field, as a function of its detuning in units of the width of the locked region.

Everything here follows from one substitution. Replacing the coupling by the
locking bandwidth as the variable along the branch turns the self consistency
into

```
chiN(Omega) = Omega / H(Omega),     R(Omega) = H(Omega),
H(Omega)    = Omega * integral p(Omega u) W(u) du,
```

an **exact parametric solution** for any frequency distribution and any kernel.
Sweeping `Omega` upward from zero traces the whole branch with no root finding
and no cancellation near threshold, so the branch is computed rather than
fitted and the exponent is read off it directly.

## What the code establishes

| Result | Where |
|---|---|
| Exact parametric solution, against two closed forms at 30 digits | `scripts/01_parametric.py` |
| Threshold `chiN_c = 1/(p(0) m)`, `m` the area under the kernel | `parametric.threshold` |
| `beta = 1/(s-1)` for `1 < s < 3` and `1/2` for `s >= 3`, set by the kernel tail | `scripts/02_universality_line.py` |
| The amplitude for the whole line, not only the exponent | `parametric.amplitude_general` |
| Order of the transition from the sign of one nonlocal integral, with an exact tricritical point | `scripts/03_order_of_transition.py` |
| The time averaged kernel against the full undamped dynamics, 48 runs | `scripts/04_meanfield_check.py` |
| The solver against three independent exact references | `scripts/05_solver_validation.py` |
| Coupling spread reproduces both published network exponent families | `scripts/06_coupling_disorder.py` |
| The averaged kernel against a simulation with the disorder quenched | `scripts/09_quenched_simulation.py` |
| The marginal case `s = 3` and its logarithmic correction | `scripts/10_marginal_and_robustness.py` |

The two kernels that occur physically sit on opposite sides of the family.
Overdamped phase oscillators release a drifting member completely, so their
kernel has compact support and `beta = 1/2`. Oscillators that precess keep an
algebraically decaying projection at every detuning, so their kernel has tail
exponent `s = 2` and `beta = 1`. The factor two between the two thresholds and
the factor two between the two exponents have the same origin.

## Coupling spread, and the published network exponents

When oscillators do not all couple with the same strength, the kernel in the
self consistency is an average of the single oscillator response,

```
Wt(v) = (1/<k>) integral P(k) k W(v / k**eta) dk,
```

which is again a function of one variable, so the whole construction applies
unchanged. For `P(k) ~ k**-gamma` the tail of the average is
`s = min(s0, (gamma-2)/eta)`. Combined with `beta = 1/(s-1)` this gives

* `beta = 1/(gamma-3)` for `3 < gamma < 5` and `1/2` above, the published scale
  free result (Lee, Phys. Rev. E **72**, 026208, 2005);
* `beta = eta/(gamma-2-eta)`, the published degree dependent coupling result
  (Oh, Lee, Kahng and Kim, Phys. Rev. E **75**, 011104, 2007);
* and, for precessing oscillators, `beta = 1/(gamma-3)` for `3 < gamma < 4` and
  `beta = 1` above.

Both published families are reproduced numerically to between `2e-5` and `5e-4`
for five of the six published values, and to `9e-3` for the sixth, which sits in
the corner where the area under the averaged kernel is slowest to converge; their
separate regime boundaries turn out to be the same condition: the point at which
the second moment of the averaged kernel stops converging.

## Install

```bash
pip install -e .
pytest                       # 124 tests, about a minute
```

Requires Python 3.9 or later, with `numpy`, `scipy`, `mpmath` and `matplotlib`.

## Use

```python
import mpmath as mp
from lockkernel import kernels as K, lineshapes as L, parametric as P

mp.mp.dps = 25
line = L.gaussian(1.0)

P.threshold(line, K.conservative())              # threshold coupling
P.branch_point(line, K.conservative(), 1e-4)     # (chiN, R, eps) on the branch
P.extract_beta(line, K.power_tail(2.5), (-3, -4, -5, -6))[-1]   # -> 2/3
P.amplitude_general(line, K.power_tail(2.5))     # the amplitude as well
P.c_coefficient(line)                            # its sign decides the order

# a kernel produced by a spread of coupling strengths
ker = K.heterogeneous(K.kuramoto(), degree_exponent=3.8)
P.extract_beta(line, ker, (-3, -4, -5, -6))[-1]  # -> 1/(3.8-3) = 1.25
```

and for the class resolved dynamics,

```python
import numpy as np
from lockkernel import lineshapes as L
from lockkernel.ensemble import build_system
from lockkernel.cumulant import evolve_meanfield

sys_ = build_system(L.lorentzian(2 * np.pi), N=1e10, chiN=2.0 * np.pi)
s, z = evolve_meanfield(sys_, 200.0, t_eval=np.linspace(0, 200, 600))
```

## Layout

```
src/lockkernel/
    lineshapes.py   frequency distributions p(delta)
    kernels.py      locking kernels, the predicted exponent, coupling spread
    parametric.py   the exact solution, c, the amplitude, folds
    ensemble.py     detuning classes, with no population discarded
    cumulant.py     class resolved dynamics and the collective observables
    exact.py        the exact references used to validate the solver
scripts/            one script per result, each writing a record to data/
tests/              the validation suite
```

## How the numerics are validated

Nothing is asserted that is not checked against something independent.

* The parametric solution is checked against the two branches known in closed
  form, to better than `1e-28` over six decades of locking bandwidth.
* The coefficient `c` is computed twice, from its defining integral and as the
  extrapolated slope of the smoothed peak; the two agree to `1e-11`.
* The amplitude takes the closed values `1`, `pi/2`, `4/3` and `pi**2/4` on the
  Lorentzian, Gaussian, Student t and box lines, and the general form reproduces
  them along the whole line of exponents.
* The class resolved solver is checked against **exact diagonalisation of the
  full Hamiltonian** for small ensembles with several distinct detunings, against
  **exact dynamics in the collective spin space** for a few tens of emitters, and
  against the exact solution on a degenerate line, where the error falls like
  `1/N`. Structural identities of the equations are tested as well, since they
  catch a transposed index that agreement at short time does not.
* The time averaged kernel, which is an assumption rather than a theorem for an
  undamped system, is checked against the full dynamics along three independent
  axes of convergence, with the drift over each run recorded as a quasi
  stationarity diagnostic.
* The averaged kernel produced by coupling spread is checked against a
  simulation in which the weights and detunings are **drawn once and held
  fixed** and the kernel is measured from the trajectories rather than assumed,
  averaged over six independent draws of the disorder.

Two limitations are recorded rather than hidden, both in `RESULTS.md`. In an
undamped model the truncated cumulant hierarchy is secularly unstable at long
times, so the module provides physicality diagnostics that any long time use
must be gated on. And a conservative population with a broad spread of couplings
does not settle on the stationary branch, which the quenched simulation shows
directly.

## Reproducing

```bash
python scripts/01_parametric.py            # minutes
python scripts/02_universality_line.py     # minutes
python scripts/03_order_of_transition.py   # minutes
python scripts/04_meanfield_check.py       # tens of minutes, parallel
python scripts/05_solver_validation.py     # minutes
python scripts/06_coupling_disorder.py     # tens of minutes
python scripts/09_quenched_simulation.py   # tens of minutes
python scripts/10_marginal_and_robustness.py
python scripts/11_figures.py
python scripts/12_tables.py
python scripts/13_graphical_abstract.py
python scripts/14_letter_figures.py
```

Each writes a JSON record to `data/`. Every table in the paper is generated from
those records rather than typed, so a table cannot disagree with the run behind
it. The archived records and figures are deposited separately; see
`CITATION.cff`.

`14_letter_figures.py` also checks its own output and prints what it found. For
the cavity figure it reports whether any label is printed on drawn content,
whether all twelve edges of the box survive to the render, how much white space
each panel title has under it, and whether anything runs off the canvas. All
four are measured from the rendered pixels rather than from bounding boxes,
which a text artist on a three dimensional axes reports out of date.

## License

Apache License 2.0. See `LICENSE`, and `NOTICE` for the copyright statement.
