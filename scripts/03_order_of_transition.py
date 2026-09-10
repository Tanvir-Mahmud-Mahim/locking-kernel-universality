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

The criterion is tested on a line made of two equal Gaussians of component
width s whose peaks sit at +/- a, so that a/s is the offset of each peak from
the origin in units of the width.  Nothing about the order is assumed: it is read off the
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
    """Two equal Gaussians of standard deviation WIDTH centred at +/- a.

    `ratio` is a / WIDTH, the offset of each peak from the origin in units of
    the component width, which is the variable reported as a_over_s.  Note
    that lineshapes.bimodal_gaussian takes the peak to peak separation, which
    is twice the offset.
    """
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
        entry = dict(a_over_s=r, monotonic=bool(mono))
        if not mono:
            # The grid reaches down to Omega = 1e-8 so that the upper end of
            # the loop is resolved as the Omega -> 0 limit of the branch,
            # which for a subcritical onset is the threshold itself.
            fold = P.fold_interval(line, cons, lo=-8.0)
            chiN_c = P.threshold(line, cons)
            note = (f"hysteretic over chiN in [{mp.nstr(fold['chiN_lo'], 6)}, "
                    f"{mp.nstr(fold['chiN_hi'], 6)}]; the upper end is the "
                    f"threshold chiN_c = {mp.nstr(chiN_c, 6)}, where the "
                    f"low branch ceases to exist and R jumps "
                    f"{mp.nstr(fold['R_low'], 4)} -> "
                    f"{mp.nstr(fold['R_jump'], 4)}; on the way down the high "
                    f"branch ends at R = {mp.nstr(fold['R_high_at_lo'], 4)}")
            entry["chiN_lo"] = mp.nstr(fold["chiN_lo"], 10)
            entry["chiN_hi"] = mp.nstr(fold["chiN_hi"], 10)
            entry["chiN_c"] = mp.nstr(chiN_c, 10)
            entry["R_jump"] = mp.nstr(fold["R_jump"], 6)
            entry["R_high_at_lo"] = mp.nstr(fold["R_high_at_lo"], 6)
            entry["hysteresis_factor"] = mp.nstr(chiN_c / fold["chiN_lo"], 6)
        print(f"    {r:>6} {str(mono):>10}   {note}")
        entry["note"] = note
        out["order"].append(entry)

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
