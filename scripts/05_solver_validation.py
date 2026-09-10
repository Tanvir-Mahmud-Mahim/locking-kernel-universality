"""Step 5.  How far the class resolved solver can be trusted.

Section VI of the paper tests the time averaged locking kernel against the
conservative MEAN FIELD dynamics.  Those mean field equations are the limit of
a second order cumulant hierarchy at large emitter number, and this script
establishes that the hierarchy, and therefore the limit taken from it, is
correctly implemented and correctly used.

Three checks, each against something the solver does not know about.

  (a) DEGENERATE LINE.  All detunings equal.  The state stays in the maximal
      spin manifold, where the Hamiltonian is diagonal, so the exact answer is
      available at any emitter number.  The cumulant error is shown falling
      like 1/N.

  (b) INHOMOGENEOUS LINE, FULL HILBERT SPACE.  Exact diagonalisation of the
      full Hamiltonian for a small ensemble with several distinct detunings.
      This tests every term of the hierarchy: a transposed index in any one of
      the partner sums shows up at leading order in time.

  (c) INHOMOGENEOUS LINE, COLLECTIVE SPIN SPACE.  Emitters within a class are
      exchange symmetric, so each class stays in its own maximal spin
      manifold, and the exact dynamics costs prod(n_j + 1) rather than 2^N.
      This reaches a few tens of emitters and shows where the truncation stops
      being reliable.

The outcome of (c) is a limitation, and it is recorded here rather than
elsewhere: in a model with no damping the truncated hierarchy is secularly
unstable.  Connected correlations, which should stay of order 1/N, eventually
grow, and the reconstructed collective covariance loses positivity.  The mean
field limit used in the paper does not carry those correlations at all and is
unaffected, but any use of the hierarchy itself at long times must be checked
against the physicality diagnostics in `lockkernel.cumulant.physicality`.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel.cumulant import (System, coherence, coherent_state_x,  # noqa: E402
                                 evolve, physicality, wineland_xi2)
from lockkernel.exact import class_exact, full_exact, symmetric_exact  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "05_solver_validation.json")
FW = 2 * np.pi


def cumulant(sysm, times, rtol=1e-10):
    states = evolve(coherent_state_x(sysm), sysm, times[-1], t_eval=times,
                    rtol=rtol, atol=1e-14)
    return (states,
            np.array([coherence(s, sysm) for s in states]),
            np.array([wineland_xi2(s, sysm) for s in states]))


def main():
    res = {"degenerate": [], "full_hilbert": [], "collective": []}

    print("(a) degenerate line, against the exact maximal spin manifold")
    print(f"    {'N':>6} {'max |d xi2|':>13} {'max |dR|':>12}")
    times = np.linspace(0.0, 3.0, 7)[1:]
    for N in (50, 200, 1000, 5000):
        sysm = System(delta=np.zeros(1), n=np.array([float(N)]), chiN=2.0)
        _, co_c, xi_c = cumulant(sysm, times)
        co_e, xi_e = symmetric_exact(N, 2.0, times)
        row = dict(N=N, d_xi2=float(np.max(np.abs(xi_c - xi_e))),
                   d_R=float(np.max(np.abs(co_c - co_e))))
        print(f"    {N:>6} {row['d_xi2']:>13.2e} {row['d_R']:>12.2e}")
        res["degenerate"].append(row)

    print()
    print("(b) inhomogeneous line, against the full Hilbert space")
    print(f"    {'t':>6} {'|dR|':>12} {'|d xi2|':>12}")
    deltas = np.array([-1.0, -1.0, -0.3, -0.3, 0.3, 0.3, 1.0, 1.0])
    times = np.array([0.05, 0.1, 0.2, 0.4])
    co_e, xi_e = full_exact(deltas, 1.5, times)
    u, cnt = np.unique(deltas, return_counts=True)
    sysm = System(delta=u, n=cnt.astype(float), chiN=1.5)
    _, co_c, xi_c = cumulant(sysm, times, rtol=1e-11)
    for k, t in enumerate(times):
        row = dict(t=float(t), d_R=float(abs(co_c[k] - co_e[k])),
                   d_xi2=float(abs(xi_c[k] - xi_e[k])))
        print(f"    {t:>6.2f} {row['d_R']:>12.2e} {row['d_xi2']:>12.2e}")
        res["full_hilbert"].append(row)

    print()
    print("(c) inhomogeneous line, against the collective spin space")
    print(f"    {'r':>5} {'N':>4} {'xi2 exact':>11} {'xi2 cumulant':>13} "
          f"{'rel dev':>9} {'physical':>9}")
    nodes = stats.cauchy(scale=FW / 2).ppf((np.arange(4) + 0.5) / 4)
    chiNc = 0.5 * FW
    for r in (1.6, 2.0, 3.0):
        for nper in (3, 6, 12):
            pops = np.full(4, nper)
            N = int(pops.sum())
            chiN = r * chiNc
            T = max(6.0, 2.5 * N ** (1 / 3) / chiN)
            times = np.linspace(0.0, T, 120)[1:]
            co_e, xi_e = class_exact(nodes, pops, chiN, times)
            sysm = System(delta=nodes, n=pops.astype(float), chiN=chiN)
            states, _, xi_c = cumulant(sysm, times, rtol=1e-9)
            ok = [physicality(s, sysm) for s in states]
            good = [k for k, (mev, vmax) in enumerate(ok)
                    if mev >= 0.0 and vmax <= 1.0 + 1e-6]
            cut = 0
            for k in range(len(states)):
                if k not in good:
                    break
                cut = k
            xe, xc = float(np.nanmin(xi_e)), float(np.nanmin(xi_c[:cut + 1]))
            row = dict(r=r, N=N, xi2_exact=xe, xi2_cumulant=xc,
                       rel_dev=abs(xc - xe) / xe,
                       physical_fraction=(cut + 1) / len(states))
            print(f"    {r:>5.1f} {N:>4} {xe:>11.5f} {xc:>13.5f} "
                  f"{row['rel_dev']:>9.2e} {row['physical_fraction']:>9.2f}")
            res["collective"].append(row)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
