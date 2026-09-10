"""Step 1.  The exact parametric solution, the closed forms, and the mechanism.

Verifies, at thirty significant digits:

  (a) the parametric solution against the two cases where the branch is known
      in closed form, the conservative Lorentzian and the Kuramoto Lorentzian;
  (b) the linear law G(Omega)/pi = p(0) - c Omega, comparing c obtained from
      its defining integral with the extrapolated slope of G;
  (c) the resulting amplitude A = pi p(0)^2 / c against the amplitude measured
      on the branch itself;
  (d) the Kuramoto exponent 1/2 on the same lines, for contrast.
"""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402

mp.mp.dps = 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "01_parametric.json")

LINES = [
    ("lorentzian", L.lorentzian(1.0), True),
    ("gaussian", L.gaussian(1.0), True),
    ("student_t_nu3", L.student_t(3.0, 0.5), True),
    ("uniform_box", L.box(0.5), True),
    ("bimodal", L.bimodal_gaussian(2.0, 0.25), False),
]


def main():
    out = {}
    cons, kur = K.conservative(), K.kuramoto()

    print("=" * 76)
    print("(a) closed forms")
    lor = L.lorentzian(1.0)
    a = mp.mpf(1) / 2
    worst = mp.mpf(0)
    for k in range(1, 7):
        Om = mp.mpf(10) ** -k
        chiN, R, _ = P.branch_point(lor, cons, Om)
        worst = max(worst, abs(chiN - (a + Om)), abs(R - (1 - a / chiN)))
    print(f"    conservative Lorentzian, chiN = a + Omega and R = 1 - a/chiN")
    print(f"    worst absolute error over six decades of Omega: {mp.nstr(worst, 5)}")
    out["lorentzian_closed_form_err"] = mp.nstr(worst, 8)

    Kc = P.threshold(lor, kur)
    worstk = mp.mpf(0)
    for k in range(1, 5):
        Om = mp.mpf(10) ** -k
        KK, r, _ = P.branch_point(lor, kur, Om)
        worstk = max(worstk, abs(r - mp.sqrt(1 - Kc / KK)))
    print(f"    Kuramoto Lorentzian, r = sqrt(1 - Kc/K)")
    print(f"    worst absolute error: {mp.nstr(worstk, 5)}")
    out["kuramoto_closed_form_err"] = mp.nstr(worstk, 8)

    print(f"    kernel masses: conservative {mp.nstr(cons.mass(), 12)}"
          f"  Kuramoto {mp.nstr(kur.mass(), 12)}")
    print(f"    threshold ratio K_c / chiN_c = "
          f"{mp.nstr(P.threshold(lor, kur) / P.threshold(lor, cons), 12)}  (exactly 2)")
    out["threshold_ratio"] = mp.nstr(P.threshold(lor, kur) / P.threshold(lor, cons), 12)

    print("=" * 76)
    print("(b) the linear law and the coefficient c")
    print(f"    {'line':16s} {'p(0)':>14} {'c integral':>14} {'c slope':>14} {'rel dev':>10}")
    out["c_check"] = {}
    for name, line, _ in LINES:
        p0 = line.p0()
        c_int = P.c_coefficient(line)
        O1, O2 = mp.mpf("1e-6"), mp.mpf("2e-6")
        g = lambda o: P.G_of_Omega(line, cons, o) / mp.pi
        slope = 2 * (p0 - g(O1)) / O1 - (p0 - g(O2)) / O2
        rel = abs(slope - c_int) / abs(c_int)
        print(f"    {name:16s} {mp.nstr(p0, 8):>14} {mp.nstr(c_int, 8):>14} "
              f"{mp.nstr(slope, 8):>14} {mp.nstr(rel, 3):>10}")
        out["c_check"][name] = dict(p0=mp.nstr(p0, 12), c_integral=mp.nstr(c_int, 12),
                                    c_slope=mp.nstr(slope, 12), rel_dev=mp.nstr(rel, 6))

    print("=" * 76)
    print("(c) the exponent and the amplitude on the conservative kernel")
    out["conservative"] = {}
    for name, line, unimodal in LINES:
        betas = P.extract_beta(line, cons, (-3, -4, -5, -6))
        A_pred = P.amplitude(line)
        _, R, eps = P.branch_point(line, cons, mp.mpf(10) ** -6)
        A_num = R / eps
        print(f"    {name:16s} beta -> {mp.nstr(betas[-1], 10):>14}   "
              f"A predicted {mp.nstr(A_pred, 10):>12}   measured {mp.nstr(A_num, 10):>12}")
        out["conservative"][name] = dict(beta=[mp.nstr(b, 12) for b in betas],
                                         A_pred=mp.nstr(A_pred, 12),
                                         A_num=mp.nstr(A_num, 12),
                                         unimodal=unimodal)

    print("=" * 76)
    print("(d) the Kuramoto kernel on the same lines")
    out["kuramoto"] = {}
    for name, line, _ in LINES[:4]:
        pts = P.sweep(line, kur, (-2, -3, -4))
        if any(p[2] == 0 for p in pts):
            print(f"    {name:16s} degenerate: a box line makes G constant for "
                  f"Omega below its half width, so the coupling never leaves "
                  f"threshold and the onset is not a power law")
            out["kuramoto"][name] = "degenerate"
            continue
        betas = P.extract_beta(line, kur, (-2, -3, -4))
        print(f"    {name:16s} beta -> {mp.nstr(betas[-1], 10)}")
        out["kuramoto"][name] = [mp.nstr(b, 12) for b in betas]

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
