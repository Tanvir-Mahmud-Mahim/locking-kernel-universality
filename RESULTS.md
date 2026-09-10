# Results ledger

Every number here was computed in this repository and cross checked. Nothing is
quoted from memory. Scripts are in `scripts/`, raw output in `data/`, and each
number below names the script that produced it.

The claim of the paper is one formula. The order parameter exponent of a mean
field synchronization transition is set by the tail of the locking kernel,

    beta = 1/(s-1)  for 1 < s < 3,    beta = 1/2  for s >= 3,

for a kernel W ~ |u|^-s, with the crossover at s = 3 lying exactly where the
second moment of the kernel ceases to converge.

---

## 1. The exact parametric solution (`01_parametric.py`)

Substituting the locking bandwidth for the coupling as the parameter along the
branch turns the self consistency into

    chiN(Omega) = Omega / H(Omega),   R(Omega) = H(Omega),
    H(Omega)    = Omega int p(Omega u) W(u) du,

exactly, for any line and any kernel. The threshold is chiN_c = 1/(p(0) m) with
m the kernel mass, so the kernel enters the threshold only through its mass.

Verified at 30 significant digits:

| check | result |
|---|---|
| conservative Lorentzian closed form, worst error over six decades | **4.18e-31** |
| overdamped Lorentzian closed form, worst error | **9.69e-29** |
| threshold ratio K_c / chiN_c | **exactly 2**, every line |
| c from its integral vs the extrapolated slope of h | **1e-11** or better |

Amplitude at s = 2, A = pi p(0)^2 / c, against the exact branch:

| line | A predicted | A measured |
|---|---|---|
| Lorentzian | 1 | 0.9999980 |
| Gaussian | pi/2 = 1.5707963 | 1.5707927 |
| Student t (nu=3) | 4/3 = 1.3333333 | 1.3333308 |
| box | **pi^2/4 = 2.4674011** | 2.4673948 |

The box amplitude is exactly pi^2/4 independently of the box width, which is a
check on the whole construction since p(0) and c separately depend on it.

## 2. The line of universality classes (`02_universality_line.py`)

Tested at a fixed Gaussian line against the kernel family W_s = 1/(1+|u|^s),
25 digits, exponent read as a converged local slope with no fitting.

| s | beta measured | predicted | rel. dev |
|---|---|---|---|
| 1.5 | 1.9926305 | 2 | 3.7e-3 |
| 1.8 | 1.2498389 | 5/4 | 1.3e-4 |
| **2.0** | **0.99999108** | **1** | **8.9e-6** |
| 2.2 | 0.83336428 | 5/6 | 3.7e-5 |
| 2.5 | 0.66714445 | 2/3 | 7.2e-4 |
| 3.0 | 0.52113288 | 1/2 | 4.2e-2 (marginal, log corrections) |
| 3.5 | 0.50028801 | 1/2 | 5.8e-4 |
| 4.0 | 0.50000173 | 1/2 | 3.5e-6 |
| 6.0 | 0.50000000 | 1/2 | **3.0e-11** |
| overdamped kernel | 0.50000000 | 1/2 | < 1e-8 |
| Gaussian kernel | 0.50000000 | 1/2 | < 1e-8 |

Deviations are largest exactly where the derivation says they must be: near
s = 1 where the kernel mass barely converges, and at the marginal s = 3.

### The amplitude along the whole line

    R = A_s eps^(1/(s-1)),   A_s = p(0) m [ p(0) m / (C I_s) ]^(1/(s-1)),
    I_s = int [p(0)-p(d)] |d|^-s dd = (2/(s-1)) int_0^inf d^(1-s) [-p'(d)] dd.

The window 1 < s < 3 is exactly the window in which I_s converges: at s = 3 it
diverges at the origin, at s = 1 at infinity. Verified on three lines at
s = 1.5, 1.8, 2.0, 2.2, 2.5, best at s = 2 (2e-7) and degrading towards both
ends of the window as the subleading terms grow. Reduces to pi p(0)^2/c at
s = 2.

The by parts form is what makes it computable: the direct form subtracts two
numbers that agree to as many digits as the quadrature approaches the origin,
and evaluating it directly gives answers wrong by factors of 1e6.

## 3. Coupling disorder and the published network exponents (`06_coupling_disorder.py`)

With weights k ~ P(k), a field on each oscillator scaling as k^eta, and a
contribution weighted by k, the kernel in the self consistency is an average,

    Wt(v) = (1/<k>) int P(k) k W(v/k^eta) dk,

again a function of one variable, so the whole construction applies unchanged.
For P(k) ~ k^-gamma the tail of the average is

    s = min(s0, (gamma-2)/eta).

Both weight integrals were reduced to closed form (Beta and Gauss
hypergeometric) and checked against direct quadrature to **1e-15**.

| base | gamma | eta | s predicted | s measured | beta measured | published |
|---|---|---|---|---|---|---|
| overdamped | 3.4 | 1 | 1.4 | 1.4 | 2.4782864 | 1/(g-3) = 2.5 |
| overdamped | 3.8 | 1 | 1.8 | 1.8 | 1.2498886 | 1/(g-3) = 1.25 |
| overdamped | 4.4 | 1 | 2.4 | 2.4 | 0.7144167 | 1/(g-3) = 0.714286 |
| overdamped | 5.5 | 1 | 3.5 | 3.5 | 0.50024226 | 1/2 |
| conservative | 3.4 | 1 | 1.4 | 1.39831 | 2.4610525 | none |
| conservative | 3.8 | 1 | 1.8 | 1.76637 | 1.2783489 | none |

So the framework **reproduces the published scale free Kuramoto exponent**,
beta = 1/(gamma-3) for 3 < gamma < 5 and 1/2 above [Lee, PRE 72, 026208 (2005)],
to between 9e-5 and 5e-4 at gamma = 3.8, 4.4 and 5.5, and to 8.7e-3 at
gamma = 3.4, which sits in the corner where the area under the averaged kernel
is slowest to converge; and **reproduces the degree dependent coupling family**
beta = eta/(gamma-2-eta) [Oh, Lee, Kahng & Kim, PRE 75, 011104 (2007)].

It also **explains the crossover at gamma = 5**, which the literature reports
without explanation: it is the point at which s reaches 3, that is where the
second moment of the averaged kernel, and the integral I_s, cease to converge.

**The prediction.** The conservative base kernel already has s0 = 2, so disorder
can only soften it so far, and s = min(2, gamma-2). Hence

    beta = 1/(gamma-3) for 3 < gamma < 4,    beta = 1 for gamma >= 4.

The crossover moves from 5 to 4 and the saturated value from 1/2 to 1. In the
window 3 < gamma < 4 the two dynamics share an exponent: disorder dominates the
kernel tail and the single oscillator response stops mattering.

## 4. The order of the transition (`03_order_of_transition.py`)

The sign of c decides the order. For a two Gaussian line of separation a and
component width sigma:

- tricritical point at **a/sigma = 1.30692972772** (reproduced to 11 digits by
  an independent implementation of the whole machinery);
- sharp: a/sigma = 1.28 gives c = +0.106772 and a monotonic branch,
  a/sigma = 1.32 gives c = -0.0504961 and a fold;
- at a/sigma = 2.0 the upper turning point sits at chiN_c = 1.4739, the order
  parameter jumps there from 0 to **0.848**, and the synchronized state survives
  down to chiN = 0.8734, so the hysteresis spans a factor **1.688** in coupling.

## 5. The kernel against the full dynamics (`04_meanfield_check.py`)

The one step of the argument that is not a theorem. For a conservative
population the mean field kinetics is Vlasov like, so a time averaged kernel is
an ansatz. Tested directly against the class resolved mean field dynamics on a
Lorentzian line, 48 runs along three independent axes.

| axis | runs | max rel. dev | median rel. dev | max abs drift | drift signs |
|---|---|---|---|---|---|
| grid, M = 200..1600 | 16 | 6.50e-3 | 6.26e-4 | 3.39e-3 | +10 / -6 |
| run length, T = 50..400 | 16 | 6.78e-3 | 9.12e-4 | 7.58e-3 | +14 / -2 |
| free-region cutoff, 4..32 | 16 | 1.72e-3 | 2.20e-4 | 3.32e-3 | +7 / -9 |

The residual oscillation of the contrast about its own mean is about 6 percent
of R, so the mean is reproduced roughly two orders of magnitude below the scale
on which the instantaneous contrast moves. Neither the deviation nor the drift
grows along any axis.

## 6. Validation of the solver (`05_solver_validation.py`)

Three independent references, all in the repository.

- **Degenerate line**, exact in the maximal spin manifold at any N. Cumulant
  error falls as 1/N: 1.9e-1, 1.1e-2, 8.7e-4, 1.3e-4 at N = 50, 200, 1000, 5000.
- **Full Hilbert space**, exact diagonalisation for eight and ten emitters with
  four distinct detunings. Agreement 2.7e-8 in the contrast and 1.1e-5 in the
  squeezing parameter at short time, growing with time as a truncation must.
- **Collective spin space**, exact for a few tens of emitters on a genuinely
  inhomogeneous line, dimension prod(n_j+1) instead of 2^N. Verified against the
  full Hilbert space result to 10 decimal places.

Two errors were caught this way and both are recorded because they changed
results:

1. A **transposed index** in the sigma-z sigma-z block of the cumulant
   equations. Found by comparing the right hand side term by term against an
   independent implementation; the exact diagonalisation test at short time did
   NOT catch it. A structural test was added that does: the hierarchy must
   preserve the Hermiticity of one correlation block and the symmetry of two
   others, and a transposition breaks it.
2. **Rebuilding the ensemble** from its detunings and populations silently
   discarded the far detuned classes, truncating the line and renormalising it,
   which put a constant +0.033 offset on every contrast. Fixed by never
   rebuilding; the population is now conserved by construction and a test checks
   that it sums to one.

**A limitation, recorded rather than hidden.** In a model with no damping the
truncated hierarchy is secularly unstable: connected correlations, which should
stay of order 1/N, eventually grow and the reconstructed collective covariance
loses positivity. Near the physical optimum the cumulant is accurate to parts in
1e3, but over long windows it is not reliable. The mean field limit used in the
paper carries no connected correlations and is unaffected; the point of the
check is to establish that, and to establish the equations from which the limit
is taken. Physicality diagnostics are provided in `lockkernel.cumulant`.

## 7. A claim that was withdrawn

An earlier version of this work claimed a second, higher threshold at which the
synchronized state acquires metrological entanglement, at r* about 1.7. **It was
an artifact and it is withdrawn.** Two independent problems: the time grid used
to locate the minimum of the squeezing parameter was too coarse to resolve the
short time behaviour, and the cumulant expansion is not trustworthy over the
long windows the large N cases require, as Sec. 6 above now documents. Exact
dynamics in the collective spin space at N = 36 shows squeezing already at
r = 1 on a Gaussian line. Whether an entanglement threshold exists at all
depends on the line shape, through whether the contrast decays linearly or
quadratically at short times, and settling it quantitatively needs a solver that
stays physical over long windows in an undamped model. No number is claimed.

## 8. Prior art, delimited

A dedicated search established the following, with sources retrieved rather
than recalled.

- beta = 1/(gamma-3) for 3 < gamma < 5, 1/2 above: **confirmed**, primary source
  Lee, PRE 72, 026208 (2005), verified through four independent citing sources
  including Hong, Park & Tang, PRE 76, 066104 (2007) and the reviews.
- Ichinomiya, PRE 70, 026116 (2004) gives the **threshold only**, not the
  exponent. Do not cite it for beta.
- The kernel tail formulation beta = 1/(s-1) with a crossover at s = 3: **no
  prior art found**. Novelty cannot be certified, only searched for.
- Daido, PRL 73, 760 (1994) reports beta = 1 for generic coupling; it
  **requires a non vanishing second harmonic**, which the present model does not
  have, so the mechanisms are distinct.
- Virkar, Restrepo & Meiss, PRE 92, 052802 (2015) treat the Hamiltonian mean
  field model on a network and derive beta = 1/2. **That is a different model**:
  its members carry inertia, and a drifting member retains an average projection
  of order Omega/omega^2, which is not a function of omega/Omega, so no kernel
  of the present type exists for it. They also state that the leading
  coefficient vanishes in the heavy tailed regime and leave that regime open.
- A power law distribution of **coupling strengths** independent of the degrees:
  no prior art found. The nearest is Oh et al. (2007), where the coupling
  heterogeneity is inherited from the degrees.

## 9. Reproducing

```
python scripts/01_parametric.py          # minutes
python scripts/02_universality_line.py   # minutes
python scripts/03_order_of_transition.py # minutes
python scripts/04_meanfield_check.py     # tens of minutes, parallel
python scripts/05_solver_validation.py   # minutes
python scripts/06_coupling_disorder.py   # tens of minutes
python scripts/07_figures.py
python scripts/08_tables.py
pytest
```
