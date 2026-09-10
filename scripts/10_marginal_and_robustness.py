"""Step 10.  The two loose ends: the marginal case, and initial conditions.

Two limitations of the argument are addressed here rather than left as caveats.

(a) THE MARGINAL CASE s = 3.  Everywhere else the exponent is read off a
converged local slope.  At s = 3 the slope does not converge, and the paper
would otherwise have to say only that it approaches 1/2 slowly.  The reason is
known from the derivation: the integral I_s diverges logarithmically at the
origin when s = 3, so the deficit is

    G(0) - G(Omega) ~ const * Omega^2 ln(1/Omega),

hence eps ~ const * Omega^2 ln(1/Omega) while R ~ const * Omega, and therefore

    R^2 ln(1/eps) / eps  ->  constant.

That is a sharp, falsifiable statement, and it is what is tested: the combination
above is measured along the branch and checked for convergence, while the
uncorrected combination R^2/eps is shown to drift.  A local slope with the
logarithm removed is also reported, and it should approach 1/2 much faster than
the raw slope does.

(b) INITIAL CONDITIONS, AND WHAT THE LAW ACTUALLY DESCRIBES.  The locking
kernel of a precessing oscillator,

    W(u) = 1 / (1 + u^2),

is the time averaged projection of a unit vector that STARTS ALIGNED WITH THE
COLLECTIVE FIELD and then precesses about the tilted axis (Omega, 0, delta).
An oscillator that starts somewhere else keeps a different average, because a
conservative dynamics never forgets where it began.  The self consistency built
from that kernel therefore describes an ensemble prepared coherently, which is
exactly the state a pi/2 pulse produces in the cavity realisation, and not an
arbitrary preparation.

This is a scope condition, not a defect, and the point of the second part of
this script is to state it quantitatively rather than leave it implicit.  Four
preparations are run at the same couplings.  The coherent one reproduces the
locking law.  The others do not, and they should not: a conservative mean field
has a continuum of stationary states and no mechanism that selects one of them.
In particular the incoherent state on the equator is itself stationary and is
NOT linearly unstable, because the torque that would build coherence is
proportional to the population inversion, which vanishes there; R = 0 solves
the self consistency for every coupling.  The overdamped problem is different
precisely because it relaxes, so the same attractor is reached from almost any
start.
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402
from lockkernel.cumulant import State, evolve_meanfield, _rhs_meanfield  # noqa: E402
from lockkernel.ensemble import build_system  # noqa: E402

mp.mp.dps = 25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "10_marginal_and_robustness.json")
FW = 2 * np.pi
CHINC = 0.5 * FW


# ---------------------------------------------------------------- part (a) --

def marginal(line, s=3.0, exps=(-3, -4, -5, -6, -7, -8)):
    ker = K.power_tail(s)
    rows = []
    for e in exps:
        Om = mp.mpf(10) ** mp.mpf(e)
        _, R, eps = P.branch_point(line, ker, Om)
        rows.append(dict(exp=float(e), Omega=float(Om), R=float(R),
                         eps=float(eps),
                         raw=float(R ** 2 / eps),
                         corrected=float(R ** 2 * mp.log(1 / eps) / eps)))
    for k in range(1, len(rows)):
        a, b = rows[k - 1], rows[k]
        rows[k]["slope_raw"] = float(
            mp.log(mp.mpf(b["R"]) / a["R"]) / mp.log(mp.mpf(b["eps"]) / a["eps"]))
        # slope with the logarithm divided out of eps
        ea = mp.mpf(a["eps"]) / mp.log(1 / mp.mpf(a["eps"]))
        eb = mp.mpf(b["eps"]) / mp.log(1 / mp.mpf(b["eps"]))
        rows[k]["slope_corrected"] = float(
            mp.log(mp.mpf(b["R"]) / a["R"]) / mp.log(eb / ea))
    return rows


# ---------------------------------------------------------------- part (b) --

def initial_states(M, kind, rng):
    """Four different starting states, all of physical single spin states."""
    if kind == "coherent":
        return np.full(M, 0.5, complex), np.zeros(M, complex)
    if kind == "random_phase":
        ph = rng.uniform(0, 2 * np.pi, M)
        return 0.5 * np.exp(1j * ph), np.zeros(M, complex)
    if kind == "partially_dephased":
        ph = rng.normal(scale=1.2, size=M)
        return 0.5 * np.exp(1j * ph), np.zeros(M, complex)
    if kind == "tilted":
        z = -0.5 + rng.uniform(-0.1, 0.1, M)
        amp = 0.5 * np.sqrt(np.clip(1 - z ** 2, 0, 1))
        return amp.astype(complex), z.astype(complex)
    raise ValueError(kind)


def contrast_from(kind, r, seed, M_core=600, T=300.0, npts=800):
    chiN = r * CHINC
    sysm = build_system(L.lorentzian(FW), 1e10, chiN, M_core=M_core, M_tail=24,
                        M_spec=8, core_edge=3 * FW, free_factor=8.0,
                        delta_max=2000.0 * FW)
    rng = np.random.default_rng(seed)
    s0, z0 = initial_states(sysm.M, kind, rng)
    vs0 = np.zeros((sysm.K, 3))
    vs0[:, 0] = 1.0
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0.0, T, npts)
    sol = solve_ivp(_rhs_meanfield, (0.0, T),
                    np.concatenate([s0, z0]),
                    args=(sysm.delta, sysm.n, sysm.chi1), t_eval=t_eval,
                    rtol=1e-8, atol=1e-10, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    s = sol.y[:sysm.M, :]
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    half = t_eval > 0.5 * T
    return float(np.mean(coh[half])), float(np.std(coh[half]))


def main():
    res = {}

    print("(a) the marginal case s = 3")
    print(f"    {'Omega':>8} {'eps':>12} {'R^2/eps':>12} "
          f"{'R^2 ln(1/eps)/eps':>18} {'slope':>9} {'slope corr':>11}")
    rows = marginal(L.gaussian(1.0))
    for q in rows:
        print(f"    1e{q['exp']:>4.0f} {q['eps']:>12.4e} {q['raw']:>12.6f} "
              f"{q['corrected']:>18.6f} "
              f"{q.get('slope_raw', float('nan')):>9.5f} "
              f"{q.get('slope_corrected', float('nan')):>11.5f}")
    raw = [q["raw"] for q in rows]
    cor = [q["corrected"] for q in rows]
    print(f"    raw combination drifts by a factor "
          f"{max(raw) / min(raw):.3f} over five decades")
    print(f"    corrected combination drifts by a factor "
          f"{max(cor) / min(cor):.3f}")
    res["marginal"] = dict(rows=rows, raw_drift=max(raw) / min(raw),
                           corrected_drift=max(cor) / min(cor))

    print()
    print("(b) what the locking law describes: the coherent preparation")
    print(f"    {'r':>5} {'preparation':>20} {'contrast':>10} "
          f"{'oscillation':>12} {'law':>9} {'dev':>9}")
    res["initial"] = []
    for r in (1.5, 2.0, 3.0):
        law = 1.0 - 1.0 / r
        vals = {}
        for kind in ("coherent", "random_phase", "partially_dephased", "tilted"):
            R, sd = contrast_from(kind, r, seed=11)
            vals[kind] = R
            dev = abs(R - law) / law
            print(f"    {r:>5.1f} {kind:>20} {R:>10.5f} {sd:>12.2e} "
                  f"{law:>9.5f} {dev:>9.2e}", flush=True)
            res["initial"].append(dict(r=r, kind=kind, R=R, osc_sd=sd,
                                       R_law=law, rel_dev=dev))
        print(f"    {'':>5} {'coherent deviation':>20} "
              f"{abs(vals['coherent'] - law) / law:>10.2e}")
    json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
