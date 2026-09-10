"""Step 2.  The line of universality classes.

The mechanism predicts that the order parameter exponent is set by the TAIL of
the locking kernel and by nothing else,

    beta(s) = 1/(s-1)   for 1 < s < 3,        beta(s) = 1/2   for s >= 3,

with the crossover at s = 3 where the second moment of the kernel stops
converging.  This is tested against the family W_s(u) = 1/(1+|u|^s) at fixed
line shape, and against the two kernels that occur physically, the
conservative kernel (s = 2) and the Kuramoto kernel (compact support).

The exponent is read as the local slope of log R against log eps along the
exact branch, at Omega down to 1e-6, so that no fitting is involved.
"""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402
from lockkernel.kernels import predicted_beta  # noqa: E402

mp.mp.dps = 25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "02_universality_line.json")

EXPS = (-3, -4, -5, -6)
S_TABLE = [1.4, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 6.0]
S_CURVE = [1.2 + 0.1 * i for i in range(49)]     # 1.2 to 6.0


def measure(line, kernel):
    betas = P.extract_beta(line, kernel, EXPS)
    return betas[-1], [mp.nstr(b, 12) for b in betas]


def main():
    line = L.gaussian(1.0)
    out = {"line": line.name, "omega_exponents": list(EXPS), "table": [], "curve": [],
           "reference": {}}

    print("=" * 72)
    print("The exponent against the kernel tail, on a Gaussian line")
    print(f"    {'s':>5} {'beta measured':>16} {'predicted':>12} {'rel dev':>10}")
    for s in S_TABLE:
        b, hist = measure(line, K.power_tail(s))
        pr = predicted_beta(s)
        rel = abs(float(b) - pr) / pr
        print(f"    {s:>5.1f} {mp.nstr(b, 10):>16} {pr:>12.6f} {rel:>10.1e}")
        out["table"].append(dict(s=s, beta=mp.nstr(b, 12), predicted=pr,
                                 rel_dev=rel, history=hist))

    print("=" * 72)
    print("Reference kernels")
    for name, kern, note in (
            ("conservative", K.conservative(), "algebraic tail s = 2"),
            ("kuramoto", K.kuramoto(), "compact support"),
            ("gaussian", K.gaussian_kernel(), "faster than algebraic")):
        b, hist = measure(line, kern)
        print(f"    {name:14s} beta = {mp.nstr(b, 10):>16}   ({note})")
        out["reference"][name] = dict(beta=mp.nstr(b, 12), note=note, history=hist)

    print("=" * 72)
    print("The amplitude along the line of classes")
    print("    R = A_s eps^(1/(s-1)),  A_s = p(0) m [p(0) m / (C I_s)]^(1/(s-1))")
    print(f"    {'line':13s} {'s':>5} {'A predicted':>16} {'A measured':>16} {'rel dev':>9}")
    out["amplitude"] = []
    for other in (L.gaussian(1.0), L.lorentzian(1.0), L.student_t(3.0, 0.5)):
        for s in (1.5, 1.8, 2.0, 2.2, 2.5):
            kern = K.power_tail(s)
            A = P.amplitude_general(other, kern)
            _, R, eps = P.branch_point(other, kern, mp.mpf(10) ** -7)
            Am = R / eps ** (1 / (mp.mpf(s) - 1))
            rel = abs(A - Am) / A
            print(f"    {other.name:13s} {s:>5.1f} {mp.nstr(A, 10):>16} "
                  f"{mp.nstr(Am, 10):>16} {float(rel):>9.1e}", flush=True)
            out["amplitude"].append(dict(line=other.name, s=s,
                                         A_pred=mp.nstr(A, 12),
                                         A_meas=mp.nstr(Am, 12),
                                         rel_dev=float(rel)))

    print("=" * 72)
    print("Fine curve for the figure")
    for s in S_CURVE:
        b, _ = measure(line, K.power_tail(s))
        out["curve"].append(dict(s=round(s, 4), beta=float(b),
                                 predicted=predicted_beta(s)))
        print(f"    s = {s:5.2f}   beta = {float(b):.8f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
