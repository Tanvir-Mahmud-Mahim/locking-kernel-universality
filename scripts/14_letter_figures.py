"""Figures for the Letter version.

Two files are produced.

  figL1  the three panels the Letter needs in one place: the two locking
         kernels, the branches they give on two different lines, and the
         exponent along the whole family.  Panels (a) and (b) are computed
         here from the package; panel (c) reads the measured exponents from
         data/02_universality_line.json.

  figL3  the order of the transition, for the End Matter, in a single column.
         It reads data/03_order_of_transition.json.

  figL4  the one physical setting the Letter names, a spin ensemble coupled to
         a detuned cavity mode, in a single column.  The drawing is the one
         already used as panel (a) of the graphical abstract: this script
         imports draw_cavity from scripts/13_graphical_abstract.py rather than
         redrawing it, so the two cannot drift apart.  It carries no numbers
         and is not a device drawing, because the paper contains no device.

The two remaining Letter figures are the ones the long version already uses,
fig2_disorder and fig3_convergence, written by scripts/11_figures.py.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
D = os.path.join(ROOT, "data")
F = os.path.join(ROOT, "figures")

for fam in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "mathtext.fontset": "stix",
                     "xtick.direction": "in", "ytick.direction": "in"})

BLUE, RED, GREY, GREEN = "#1f6feb", "#c1121f", "#666666", "#2a7f4f"


def save(fig, stem):
    out = os.path.join(F, stem)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", out + ".pdf and .png")


def letter_fig1():
    import mpmath as mp
    from lockkernel import kernels as K, lineshapes as L, parametric as P
    mp.mp.dps = 20

    uni = json.load(open(os.path.join(D, "02_universality_line.json")))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # ---- (a) the two kernels ------------------------------------------
    # The compact kernel is sampled on a grid that closes on u = 1 from below,
    # so the curve falls continuously through the bottom of the frame; it is
    # identically zero beyond u = 1 and cannot be drawn there on a log axis.
    u = np.logspace(-1.5, 1.5, 500)
    cons = np.array([float(K.conservative().W(x)) for x in u])
    uk = np.concatenate([np.logspace(-1.5, np.log10(0.98), 300),
                         1.0 - np.logspace(-2, -10, 200)])
    kur = np.array([float(K.kuramoto().W(x)) for x in uk])
    ax[0].loglog(u, cons, color=RED, lw=1.4)
    ax[0].loglog(uk, kur, color=BLUE, lw=1.4)
    ax[0].loglog(u[u > 2], 1.0 / u[u > 2] ** 2, color=GREY, lw=0.7,
                 ls=(0, (4, 3)))
    ax[0].set_xlim(4e-2, 30)
    ax[0].set_ylim(3e-4, 4.0)
    ax[0].set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax[0].set_xlabel(r"detuning $u$, in locked widths")
    ax[0].set_ylabel(r"locking kernel $W(u)$")
    ax[0].text(2.3, 0.60, "precessing", color=RED, fontsize=7.5)
    ax[0].text(2.3, 0.26, r"$|u|^{-2}$", color=RED, fontsize=8)
    ax[0].text(0.055, 0.006, "overdamped:\ncompact support", color=BLUE,
               fontsize=7.5)
    ax[0].set_title("(a)  the two kernels", fontsize=8.5, loc="left")

    # ---- (b) the branches they give -----------------------------------
    consk, kur_k = K.conservative(), K.kuramoto()
    eps = np.logspace(-6.6, -1.0, 60)
    for line, mk, dash in ((L.lorentzian(1.0), "o", "-"),
                           (L.gaussian(1.0), "s", "--")):
        E, R = zip(*[(float(q[2]), float(q[1]))
                     for q in P.sweep(line, consk, (-2, -3, -4, -5, -6))])
        ax[1].loglog(E, R, mk, ms=3.2, color=RED, mfc="none", mew=0.9, zorder=3)
        ax[1].loglog(eps, (R[-1] / E[-1]) * eps, dash, color=RED, lw=1.0,
                     zorder=1)
        Ek, Rk = zip(*[(float(q[2]), float(q[1]))
                       for q in P.sweep(line, kur_k, (-1, -1.5, -2, -2.5, -3))])
        ax[1].loglog(Ek, Rk, mk, ms=3.2, color=BLUE, mfc="none", mew=0.9,
                     zorder=3)
        ax[1].loglog(eps, (Rk[-1] / np.sqrt(Ek[-1])) * np.sqrt(eps), dash,
                     color=BLUE, lw=1.0, zorder=1)
    ax[1].loglog([], [], "-", color=RED, lw=1.2, label=r"precessing, $\beta=1$")
    ax[1].loglog([], [], "-", color=BLUE, lw=1.2,
                 label=r"overdamped, $\beta=1/2$")
    ax[1].set_xlim(3e-7, 2e-1)
    ax[1].set_ylim(1e-7, 3.0)
    ax[1].set_xlabel(r"$\varepsilon=\chi N/\chi N_c-1$")
    ax[1].set_ylabel(r"order parameter $R$")
    ax[1].legend(frameon=False, fontsize=6.4, loc="upper left",
                 handlelength=1.4, borderaxespad=0.3)
    ax[1].text(0.97, 0.05, "circles: Lorentzian line\nsquares: Gaussian line",
               transform=ax[1].transAxes, fontsize=5.9, color=GREY,
               ha="right", va="bottom")
    ax[1].set_title("(b)  same lines, two kernels", fontsize=8.5, loc="left")

    # ---- (c) the exponent along the family ----------------------------
    s = np.array([r["s"] for r in uni["curve"]])
    b = np.array([r["beta"] for r in uni["curve"]])
    ss = np.linspace(s.min(), s.max(), 600)
    ax[2].plot(ss, np.where(ss < 3, 1.0 / (ss - 1), 0.5), "-", color=GREY,
               lw=1.2, zorder=1, label=r"$1/(s-1)$ and $1/2$")
    ax[2].plot(s, b, "o", ms=3.0, mfc="none", mec=RED, mew=0.8, zorder=3,
               label="exact branch")
    ax[2].plot([2.0], [1.0], "*", ms=10, color=RED, zorder=4,
               label="precessing spins")
    ax[2].axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax[2].plot([3.0, 3.0], [0.35, 2.1], color=GREY, lw=0.6, ls=":", zorder=0)
    ax[2].text(4.3, 0.57, "Kuramoto class", fontsize=6.5, color=GREY,
               ha="center")
    ax[2].set_xlabel(r"kernel tail exponent $s$")
    ax[2].set_ylabel(r"order-parameter exponent $\beta$")
    ax[2].set_xlim(s.min(), s.max())
    ax[2].set_ylim(0.35, 5.2)
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.5, 1, 2, 5])
    ax[2].set_yticklabels(["0.5", "1", "2", "5"])
    ax[2].legend(frameon=False, fontsize=6.3, loc="upper right",
                 handlelength=1.4, borderaxespad=0.3)
    ax[2].set_title("(c)  one line of classes", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.5)
    save(fig, "figL1_kernel_and_exponent")


def letter_fig2():
    """Coupling disorder, in the colour convention of the Letter.

    Red is the precessing population throughout the Letter and blue the
    overdamped one, which is the reverse of the long version, so the disorder
    figure is redrawn here rather than reused.
    """
    d = json.load(open(os.path.join(D, "06_coupling_disorder.json")))["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

    g = np.linspace(3.05, 6.5, 700)
    over = np.where(g < 5, 1.0 / np.maximum(g - 3, 1e-9), 0.5)
    prec = np.where(g < 4, 1.0 / np.maximum(g - 3, 1e-9), 1.0)

    ax[0].fill_between([3.0, 4.0], 0.3, 12, color=GREY, alpha=0.10, lw=0)
    ax[0].plot(g, prec, "-", color=RED, lw=3.0, alpha=0.55,
               solid_capstyle="butt", label=r"precessing, $s_0=2$")
    ax[0].plot(g, over, "-", color=BLUE, lw=1.4,
               label=r"overdamped, $s_0=\infty$")
    for base, col, mk in (("kuramoto", BLUE, "o"), ("conservative", RED, "s")):
        rows = [r for r in d if r["base"] == base and r["eta"] == 1.0]
        if rows:
            ax[0].plot([r["gamma"] for r in rows],
                       [r["beta_measured"] for r in rows],
                       mk, ms=4.2, color=col, mfc="none", mew=1.1, zorder=4)
    ax[0].axvline(4.0, color=RED, lw=0.7, ls=":")
    ax[0].axvline(5.0, color=BLUE, lw=0.7, ls=":")
    ax[0].text(3.50, 0.44, "same exponent:\ndisorder dominates", fontsize=6.2,
               ha="center", color="#333333")
    ax[0].text(4.05, 1.13, r"$\gamma=4$", fontsize=6.2, color=RED)
    ax[0].text(5.05, 0.56, r"$\gamma=5$", fontsize=6.2, color=BLUE)
    ax[0].set_yscale("log")
    ax[0].set_xlim(3.05, 6.5)
    ax[0].set_ylim(0.35, 12)
    ax[0].set_yticks([0.5, 1, 2, 5, 10])
    ax[0].set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax[0].set_xlabel(r"coupling-disorder exponent $\gamma$")
    ax[0].set_ylabel(r"order-parameter exponent $\beta$")
    ax[0].legend(frameon=False, fontsize=6.4, loc="upper right",
                 handlelength=1.5, borderaxespad=0.3)
    ax[0].set_title("(a)  two dynamics, one criterion", fontsize=8.5,
                    loc="left")

    pub, meas, lab = [], [], []
    for r in d:
        if r["beta_published"] is not None:
            pub.append(r["beta_published"])
            meas.append(r["beta_measured"])
            lab.append(r["eta"])
    lo, hi = 0.4, max(pub + meas) * 1.4
    ax[1].plot([lo, hi], [lo, hi], "-", color=GREY, lw=1.0, zorder=1)
    e1 = [k for k, e in enumerate(lab) if e == 1.0]
    e2 = [k for k, e in enumerate(lab) if e != 1.0]
    ax[1].plot([pub[k] for k in e1], [meas[k] for k in e1], "o", ms=5,
               color=BLUE, mfc="none", mew=1.2, zorder=3,
               label=r"$\eta=1$: Lee (2005)")
    if e2:
        ax[1].plot([pub[k] for k in e2], [meas[k] for k in e2], "^", ms=5,
                   color=GREEN, mfc="none", mew=1.2, zorder=3,
                   label=r"$\eta\neq1$: Oh et al. (2007)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ticks = [0.5, 1, 2]
    for a in (ax[1].xaxis, ax[1].yaxis):
        a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        a.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_major_formatter(matplotlib.ticker.FixedFormatter(
            [str(v) for v in ticks]))
    ax[1].set_xlabel(r"published $\beta$")
    ax[1].set_ylabel(r"$\beta$ from the kernel tail")
    ax[1].legend(frameon=False, fontsize=6.4, loc="upper left",
                 handlelength=1.5, borderaxespad=0.3)
    ax[1].set_title("(b)  both published families recovered", fontsize=8.5,
                    loc="left")

    fig.tight_layout(w_pad=1.8)
    save(fig, "figL2_disorder")


def letter_fig3():
    ordr = json.load(open(os.path.join(D, "03_order_of_transition.json")))
    fig, ax = plt.subplots(figsize=(3.35, 2.5))

    a_s = np.array([float(r["a_over_s"]) for r in ordr["scan"]])
    cc = np.array([float(r["c"]) for r in ordr["scan"]])
    xc = float(ordr["tricritical_a_over_s"])
    lo, hi = -2.0, 6.0
    ax.fill_between([a_s.min(), xc], lo, hi, color=BLUE, alpha=0.06, lw=0)
    ax.fill_between([xc, a_s.max()], lo, hi, color=RED, alpha=0.06, lw=0)
    ax.plot(a_s, cc, "o-", ms=3.0, lw=1.1, color=BLUE, zorder=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(xc, color=RED, lw=1.0, ls="--")
    ax.text(0.10, -1.55, "continuous\n" + r"$\beta=1$", fontsize=7, color=BLUE)
    ax.text(2.05, 3.2, "first order\n(fold)", fontsize=7, color=RED)
    ax.annotate(f"tricritical\n$a/\\sigma={xc:.4f}$", xy=(xc, 0),
                xytext=(1.62, 1.35), fontsize=6.4, color=RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7,
                                shrinkA=1, shrinkB=2))
    ax.set_xlabel(r"bimodal separation $a/\sigma$")
    ax.set_ylabel(r"$c=-\pi^{-1}\!\int[p(\delta)-p(0)]\,\delta^{-2}\,d\delta$")
    ax.set_ylim(lo, hi)
    ax.set_xlim(a_s.min(), a_s.max())
    fig.tight_layout()
    save(fig, "figL3_order")


def letter_fig4():
    """The cavity realization, single column.

    The geometry is imported from the graphical abstract script so that the
    two drawings are the same object seen at two sizes.  Only the size and
    the placement of the four labels change: at single column width the
    labels of the three panel version would collide with the box, so each is
    positioned against the empty region it sits in and checked against the
    rendered figure rather than assumed.
    """
    import importlib.util
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "13_graphical_abstract.py")
    spec = importlib.util.spec_from_file_location("ga13", path)
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)

    # A 3-D axes reserves a square region whatever the box aspect, so most of
    # it is empty here.  The figure is therefore sized and the axes placed by
    # hand and saved without a tight bounding box, which would otherwise keep
    # the empty region and cost length in the journal's figure count for
    # nothing.  The four labels are placed in figure coordinates against the
    # white areas the drawing leaves.
    fig = plt.figure(figsize=(3.35, 1.90))
    ax = fig.add_axes([-0.085, -0.20, 1.17, 1.48], projection="3d")
    ga.draw_cavity(ax)

    # The labels sit in bands above and below the drawing, not on it, so that
    # none of them can touch an edge of the box or an arrow.
    fig.text(0.50, 0.985, r"spins detuned by $\delta$", fontsize=7.5,
             color=RED, ha="center", va="top")
    fig.text(0.015, 0.035, r"cavity mode $\omega_c$", fontsize=7.5,
             color=BLUE, ha="left", va="bottom")
    fig.text(0.66, 0.035, "locked", fontsize=7.5, color=RED,
             ha="left", va="bottom")
    fig.text(0.845, 0.035, "drifting", fontsize=7.5, color="#8a8a8a",
             ha="left", va="bottom")

    out = os.path.join(F, "figL4_realization")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print("saved", out + ".pdf and .png")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in want:
        letter_fig1()
    if "2" in want:
        letter_fig2()
    if "3" in want:
        letter_fig3()
    if "4" in want:
        letter_fig4()
