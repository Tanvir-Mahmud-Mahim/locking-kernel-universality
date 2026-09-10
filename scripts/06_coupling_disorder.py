"""Step 6.  Coupling disorder, and where the known network exponents come from.

The tail of the locking kernel is not a property of the single oscillator
alone.  If the oscillators couple with different strengths, the kernel that
enters the self consistency is an AVERAGE of the single oscillator response
over the coupling distribution,

    Wt(v) = (1 / <k>) int P(k) k W(v / k^eta) dk,

for weights k drawn from P, a field on each oscillator proportional to k^eta,
and a contribution to the collective field weighted by k.  This is again a
function of v alone, so the parametric solution applies unchanged.

An oscillator whose weight is far above the typical one has a locking
bandwidth far above the typical one, and stays locked at detunings where a
typical oscillator has long been released.  A heavy tail in P is therefore
inherited by the kernel, and for P(k) ~ k^-gamma,

    s = min(s0, (gamma - 2) / eta),

with s0 the tail of the base kernel, infinite for compact support.  Combined
with beta = 1/(s-1) this gives three statements, two of which can be checked
against the published literature and one of which is a prediction.

  (a) OVERDAMPED, uniform coupling exponent eta = 1.  The base kernel is
      compact, so s = gamma - 2 and

          beta = 1/(gamma - 3)   for 3 < gamma < 5,     beta = 1/2 above,

      which is the result of Lee, Phys. Rev. E 72, 026208 (2005).  The
      framework also says what the crossover at gamma = 5 IS: the point at
      which the second moment of the kernel stops converging.

  (b) OVERDAMPED, degree dependent coupling J k^(eta-1).  Then s =
      (gamma-2)/eta and

          beta = eta / (gamma - 2 - eta),

      which is the result of Oh, Lee, Kahng and Kim, Phys. Rev. E 75, 011104
      (2007) in their regime II.  Their other regime, where the exponent
      changes form, is where s < 1 and the kernel mass diverges.

  (c) CONSERVATIVE.  The base kernel already has an algebraic tail, s0 = 2, so
      disorder can only soften it so far:

          beta = 1/(gamma - 3)   for 3 < gamma < 4,     beta = 1   above.

      The crossover moves from gamma = 5 to gamma = 4.  Strong coupling
      disorder erases the difference between conservative and overdamped
      dynamics, and the two separate again only once the disorder is mild.
      This is a prediction.

The script measures the kernel tail directly, then measures the exponent from
the exact branch, and compares both with the formulas above.
"""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402

mp.mp.dps = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "06_coupling_disorder.json")
EXPS = (-3, -4, -5, -6)


def measured_tail(kernel, v1=mp.mpf(5000), v2=mp.mpf(50000)):
    """Local log-log slope of the kernel far out in the tail."""
    return -mp.log(kernel.W(v2) / kernel.W(v1)) / mp.log(v2 / v1)


def published(base_name, g, eta):
    """What the literature says, where it says anything.

    Both published families have a regime boundary, and in both cases it is
    the same boundary as the crossover of the exponent formula.  Lee's result
    changes over at gamma = 5, which is where s = (gamma-2) reaches three.
    The result of Oh, Lee, Kahng and Kim changes over from their regime II to
    their regime I at gamma = 2 + 3 eta, which is where s = (gamma-2)/eta
    reaches three.  The two statements are the same statement.
    """
    if base_name != "kuramoto":
        return None
    return eta / (g - 2 - eta) if g < 2 + 3 * eta else mp.mpf(0.5)


def main():
    line = L.gaussian(1.0)
    out = {"line": line.name, "rows": []}
    cases = (
        [("kuramoto", K.kuramoto(), g, 1.0) for g in ("3.4", "3.8", "4.4", "5.5")]
        + [("conservative", K.conservative(), g, 1.0)
           for g in ("3.4", "3.8", "4.4", "5.5")]
        + [("kuramoto", K.kuramoto(), "4.5", e) for e in (0.7, 1.3)]
    )

    print(f"{'base':>13} {'gamma':>6} {'eta':>5} {'s pred':>8} {'s meas':>10} "
          f"{'beta pred':>10} {'beta meas':>12} {'published':>11}")
    for name, base, gm, eta in cases:
        g = mp.mpf(gm)
        eta = mp.mpf(eta)
        ker = K.heterogeneous(base, g, eta)
        s_pred = mp.mpf(ker.tail)
        s_meas = measured_tail(ker)
        b_pred = 1 / (s_pred - 1) if s_pred < 3 else mp.mpf(0.5)
        b_meas = P.extract_beta(line, ker, EXPS)[-1]
        pub = published(name, g, eta)
        print(f"{name:>13} {gm:>6} {float(eta):>5.1f} {mp.nstr(s_pred, 5):>8} "
              f"{mp.nstr(s_meas, 6):>10} {mp.nstr(b_pred, 6):>10} "
              f"{mp.nstr(b_meas, 8):>12} "
              f"{(mp.nstr(pub, 6) if pub is not None else 'none'):>11}", flush=True)
        out["rows"].append(dict(
            base=name, gamma=float(g), eta=float(eta),
            s_predicted=float(s_pred), s_measured=float(s_meas),
            beta_predicted=float(b_pred), beta_measured=float(b_meas),
            beta_published=(float(pub) if pub is not None else None)))
        json.dump(out, open(OUT, "w"), indent=1)

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
