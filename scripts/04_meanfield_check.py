"""Step 4.  Does the dynamics actually reach the state the locking law describes?

This is the one assumption in the whole construction that is not a theorem.
For a conservative system the mean field kinetics is Vlasov like, so writing a
self consistency for a TIME AVERAGED locking kernel is an ansatz: the standard
objections are Landau damping and long lived quasi stationary states that
never settle on the assumed profile.  The ansatz is therefore tested directly
rather than asserted.

PROTOCOL.  Integrate the class resolved mean field equations, with no closure
and no quantum corrections, from the coherent spin state on a Lorentzian line,
and compare the long time contrast with the exact parametric prediction

    R = 1 - 1/r,        r = chiN / chiN_c,

along three independent axes of convergence:

  (a) the detuning grid, M_core = 200 to 1600 classes, at fixed long time;
  (b) the run length, T = 50 to 400 inverse linewidths, at the finest grid;
  (c) the boundary of the free region, at 4 to 32 times the larger of the
      coupling and twice the line width.

The contrast is averaged over the second half of each run.  The DRIFT between
the third and fourth quarters is recorded as the quasi stationarity
diagnostic: a state that is slowly decaying shows a systematic drift of one
sign, a genuinely stationary state does not.  The residual oscillation of the
contrast about its mean is recorded as well, since it sets the floor below
which agreement cannot be resolved.
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import lineshapes as L  # noqa: E402
from lockkernel.cumulant import evolve_meanfield  # noqa: E402
from lockkernel.ensemble import build_system  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "04_meanfield_check.json")

FW = 2 * np.pi                 # line full width at half maximum
CHINC = 0.5 * FW               # threshold coupling for the conservative kernel
N = 1e10                       # mean field limit: correlations play no role here
RS = [1.2, 1.5, 2.0, 3.0]


def run(r, M_core, T, freefac=8.0, M_tail=24, npts=600):
    chiN = r * CHINC
    sysm = build_system(L.lorentzian(FW), N, chiN, M_core=M_core, M_tail=M_tail,
                        M_spec=8, core_edge=3 * FW, free_factor=freefac,
                        delta_max=2000.0 * FW)
    t_eval = np.linspace(0.0, T, npts)
    t0 = time.time()
    s, _ = evolve_meanfield(sysm, T, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    free = 0.5 * np.exp(1j * np.outer(sysm.spec_delta, t_eval))
    coh = 2.0 * np.abs((sysm.n @ s) + (sysm.spec_n @ free)) / sysm.N
    half = t_eval > 0.5 * T
    q3 = (t_eval > 0.50 * T) & (t_eval <= 0.75 * T)
    q4 = t_eval > 0.75 * T
    R_law = max(0.0, 1.0 - 1.0 / r)
    R_dyn = float(np.mean(coh[half]))
    return dict(r=r, M_core=M_core, T=T, freefac=freefac, R_dyn=R_dyn,
                R_law=R_law, rel_dev=abs(R_dyn - R_law) / R_law,
                drift=float(np.mean(coh[q4]) - np.mean(coh[q3])),
                osc_sd=float(np.std(coh[half])), secs=time.time() - t0)


def _run(a):
    return run(*a)


HEAD = f"{'r':>5} {'x':>7} {'R dyn':>9} {'R law':>9} {'rel dev':>9} {'drift':>10} {'osc sd':>9} {'s':>5}"


def show(row, key):
    print(f"{row['r']:>5.1f} {row[key]:>7.0f} {row['R_dyn']:>9.5f} "
          f"{row['R_law']:>9.5f} {row['rel_dev']:>9.2e} {row['drift']:>10.2e} "
          f"{row['osc_sd']:>9.2e} {row['secs']:>5.0f}", flush=True)


def main():
    from concurrent.futures import ProcessPoolExecutor
    res = {"grid": [], "time": [], "cutoff": []}
    nproc = max(1, (os.cpu_count() or 2))

    blocks = [
        ("grid", "M_core", "(a) detuning grid, T = 200",
         [(r, M, 200.0) for r in RS for M in (200, 400, 800, 1600)]),
        ("time", "T", "(b) run length, M_core = 1600",
         [(r, 1600, T) for r in RS for T in (50.0, 100.0, 200.0, 400.0)]),
        ("cutoff", "freefac", "(c) boundary of the free region, M_core = 800, T = 200",
         [(r, 800, 200.0, ff) for r in RS for ff in (4.0, 8.0, 16.0, 32.0)]),
    ]
    for key, col, title, jobs in blocks:
        print()
        print(title)
        print(HEAD.replace("x", col[:7]))
        with ProcessPoolExecutor(nproc) as ex:
            for row in ex.map(_run, jobs):
                show(row, col)
                res[key].append(row)
                json.dump(res, open(OUT, "w"), indent=1)

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
