"""Step 3.  The order of the transition, and the boundary of the class.

The same coefficient that fixes the amplitude of the continuous branch,

    c = -(1/pi) int [p(delta) - p(0)] / delta^2 d delta,

also fixes the ORDER of the transition through its sign.  When the centre of
the line is a maximum the smoothing lowers it, c > 0, the branch chiN(Omega)
is monotonic, every coupling above threshold has a single solution, and the
onset is continuous with R = A eps.  When the centre is a sufficiently deep
local minimum the smoothing RAISES it, c < 0, the branch folds back, three
solutions coexist over an interval of coupling, and the onset is first order
with hysteresis and a jump in R.

The criterion is tested on a line made of two equal Gaussians of separation a
and component width s.  Nothing about the order is assumed: it is read off the
shape of the exactly computed branch, and compared with the sign of c.
"""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402

mp.mp.dps = 25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "03_order_of_transition.json")
WIDTH = mp.mpf(1) / 4


def line_at(ratio):
    return L.bimodal_gaussian(float(ratio * WIDTH * 2), float(WIDTH))


def monotonic(line, kernel, n=60):
    Om = [mp.mpf("0.001") * mp.mpf("1.25") ** k for k in range(n)]
    chi = [P.branch_point(line, kernel, o)[0] for o in Om]
    return all(chi[i + 1] > chi[i] for i in range(len(chi) - 1))


def main():
    cons = K.conservative()
    out = {"component_width": mp.nstr(WIDTH, 6), "scan": [], "order": [],
           "sharpness": []}

    print("=" * 74)
    print("Sign of c against the separation of a two Gaussian line")
    print(f"    {'a/s':>6} {'p(0)':>15} {'c':>15} {'class':>22}")
    for i in range(0, 31, 2):
        ratio = mp.mpf(i) / 10
        line = line_at(ratio)
        c = P.c_coefficient(line)
        cls = "continuous, beta = 1" if c > 0 else "first order"
        print(f"    {mp.nstr(ratio, 3):>6} {mp.nstr(line.p0(), 8):>15} "
              f"{mp.nstr(c, 8):>15} {cls:>22}")
        out["scan"].append(dict(a_over_s=mp.nstr(ratio, 6), p0=mp.nstr(line.p0(), 10),
                                c=mp.nstr(c, 10), positive=bool(c > 0)))

    f = lambda x: P.c_coefficient(line_at(x))
    xc = mp.findroot(f, (mp.mpf(1), mp.mpf(2)), solver="bisect", tol=mp.mpf("1e-14"))
    print(f"\n    tricritical point, sign change of c at a/s = {mp.nstr(xc, 12)}")
    out["tricritical_a_over_s"] = mp.nstr(xc, 12)

    print("=" * 74)
    print("Order read from the shape of the exact branch")
    print(f"    {'a/s':>6} {'monotonic':>10}   {'fold and jump'}")
    for r in ("0.5", "1.0", "1.5", "2.0", "2.5"):
        ratio = mp.mpf(r)
        line = line_at(ratio)
        mono = monotonic(line, cons)
        note = ""
        if not mono:
            fold = P.fold_interval(line, cons)
            note = (f"chiN in [{mp.nstr(fold['chiN_lo'], 5)}, "
                    f"{mp.nstr(fold['chiN_hi'], 5)}], "
                    f"R jumps {mp.nstr(fold['R_lower'], 4)} -> "
                    f"{mp.nstr(fold['R_upper'], 4)}")
        print(f"    {r:>6} {str(mono):>10}   {note}")
        out["order"].append(dict(a_over_s=r, monotonic=bool(mono), note=note))

    print("=" * 74)
    print("Sharpness of the criterion across the tricritical point")
    for r in ("1.20", "1.28", "1.32", "1.40"):
        ratio = mp.mpf(r)
        line = line_at(ratio)
        c = P.c_coefficient(line)
        mono = monotonic(line, cons)
        verdict = "continuous" if mono else "fold, first order"
        print(f"    a/s = {r:>5}  c = {mp.nstr(c, 6):>12}  monotonic = {str(mono):>5}"
              f"  -> {verdict}")
        out["sharpness"].append(dict(a_over_s=r, c=mp.nstr(c, 10),
                                     monotonic=bool(mono), verdict=verdict))

    with open(OUT, "w") as f2:
        json.dump(out, f2, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
