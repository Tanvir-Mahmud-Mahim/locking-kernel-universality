"""Step 6.  Figures.

Reads only the JSON records written by the earlier scripts, so that no figure
can disagree with the numbers that produced it, except for the two reference
branches in figure 1(b), which are recomputed here from the exact parametric
solution because they are curves rather than table entries.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def load(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(F, exist_ok=True)
    out = os.path.join(F, stem)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", out + ".pdf and .png")


# ---------------------------------------------------------------- figure 1 --
def figure_universality():
    import mpmath as mp
    from lockkernel import kernels as K, lineshapes as L, parametric as P
    mp.mp.dps = 20

    uni = load("02_universality_line.json")
    ordr = load("03_order_of_transition.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) the line of classes
    s = np.array([r["s"] for r in uni["curve"]])
    b = np.array([r["beta"] for r in uni["curve"]])
    ss = np.linspace(s.min(), s.max(), 600)
    ax[0].plot(ss, np.where(ss < 3, 1.0 / (ss - 1), 0.5), "-", color=GREY, lw=1.2,
               zorder=1, label=r"$1/(s-1)$ and $1/2$")
    ax[0].plot(s, b, "o", ms=3.4, color=BLUE, zorder=3, label="exact branch")
    ax[0].plot([2.0], [1.0], "*", ms=11, color=RED, zorder=4,
               label="conservative spins")
    ax[0].axhline(0.5, color=GREY, lw=0.6, ls=":")
    # the crossover marker is cut short of the legend rather than run through it
    ax[0].plot([3.0, 3.0], [0.35, 2.1], color=GREY, lw=0.6, ls=":", zorder=0)
    ax[0].text(4.3, 0.57, "Kuramoto class", fontsize=6.5, color=GREY, ha="center")
    ax[0].set_xlabel(r"kernel tail exponent $s$")
    ax[0].set_ylabel(r"order-parameter exponent $\beta$")
    ax[0].set_xlim(s.min(), s.max())
    ax[0].set_ylim(0.35, 5.2)
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.5, 1, 2, 5])
    ax[0].set_yticklabels(["0.5", "1", "2", "5"])
    ax[0].legend(frameon=False, fontsize=6.3, loc="upper right",
                 handlelength=1.4, borderaxespad=0.3)
    ax[0].set_title("(a)  a line of universality classes", fontsize=8.5, loc="left")

    # (b) the two physical kernels on the same lines.  The guide lines are
    # anchored on the measured amplitude of each branch, so that a point off
    # the line means an exponent that differs, not an amplitude that differs.
    cons, kur = K.conservative(), K.kuramoto()
    eps = np.logspace(-6.6, -1.0, 80)
    for line, mk, dash in ((L.lorentzian(1.0), "o", "-"),
                           (L.gaussian(1.0), "s", "--")):
        E, R = zip(*[(float(q[2]), float(q[1]))
                     for q in P.sweep(line, cons, (-2, -3, -4, -5, -6))])
        ax[1].loglog(E, R, mk, ms=3.4, color=BLUE, mfc="none", mew=0.9, zorder=3)
        ax[1].loglog(eps, (R[-1] / E[-1]) * eps, dash, color=BLUE, lw=1.0, zorder=1)
        Ek, Rk = zip(*[(float(q[2]), float(q[1]))
                       for q in P.sweep(line, kur, (-1, -1.5, -2, -2.5, -3))])
        ax[1].loglog(Ek, Rk, mk, ms=3.4, color=RED, mfc="none", mew=0.9, zorder=3)
        ax[1].loglog(eps, (Rk[-1] / np.sqrt(Ek[-1])) * np.sqrt(eps), dash,
                     color=RED, lw=1.0, zorder=1)
    ax[1].loglog([], [], "-", color=BLUE, lw=1.2, label=r"conservative, $\beta=1$")
    ax[1].loglog([], [], "-", color=RED, lw=1.2, label=r"Kuramoto, $\beta=1/2$")
    ax[1].set_xlabel(r"$\varepsilon=\chi N/\chi N_c-1$")
    ax[1].set_ylabel(r"order parameter $R$")
    ax[1].legend(frameon=False, fontsize=6.3, loc="upper left",
                 handlelength=1.4, borderaxespad=0.3)
    ax[1].set_xlim(3e-7, 2e-1)
    ax[1].set_ylim(1e-7, 3.0)
    ax[1].text(0.97, 0.06, "circles: Lorentzian line\nsquares: Gaussian line",
               transform=ax[1].transAxes, fontsize=5.9, color=GREY,
               ha="right", va="bottom")
    ax[1].set_title("(b)  same lines, two kernels", fontsize=8.5, loc="left")

    # (c) the boundary of the class
    a_s = np.array([float(r["a_over_s"]) for r in ordr["scan"]])
    cc = np.array([float(r["c"]) for r in ordr["scan"]])
    xc = float(ordr["tricritical_a_over_s"])
    lo, hi = -2.0, 6.0
    ax[2].fill_between([a_s.min(), xc], lo, hi, color=BLUE, alpha=0.06, lw=0)
    ax[2].fill_between([xc, a_s.max()], lo, hi, color=RED, alpha=0.06, lw=0)
    ax[2].plot(a_s, cc, "o-", ms=3.2, lw=1.1, color=BLUE, zorder=3)
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].axvline(xc, color=RED, lw=1.0, ls="--")
    ax[2].text(0.10, -1.55, "continuous\n" + r"$\beta=1$", fontsize=7,
               color=BLUE)
    ax[2].text(2.05, 3.2, "first order\n(fold)", fontsize=7, color=RED)
    ax[2].annotate(f"tricritical\n$a/\\sigma={xc:.4f}$", xy=(xc, 0),
                   xytext=(1.62, 1.35), fontsize=6.4, color=RED,
                   ha="left", va="center",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=0.7,
                                   shrinkA=1, shrinkB=2))
    ax[2].set_xlabel(r"bimodal separation $a/\sigma$")
    ax[2].set_ylabel(r"$c=-\pi^{-1}\!\int[p(\delta)-p(0)]\,\delta^{-2}\,d\delta$")
    ax[2].set_ylim(lo, hi)
    ax[2].set_xlim(a_s.min(), a_s.max())
    ax[2].set_title("(c)  boundary of the class", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_universality")


# ---------------------------------------------------------------- figure 2 --
def figure_disorder():
    """Coupling disorder: one formula, two published families, one prediction."""
    d = load("06_coupling_disorder.json")["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))

    g = np.linspace(3.05, 6.5, 700)
    over = np.where(g < 5, 1.0 / np.maximum(g - 3, 1e-9), 0.5)
    cons = np.where(g < 4, 1.0 / np.maximum(g - 3, 1e-9), 1.0)

    # (a) the exponent against the coupling-disorder exponent
    ax[0].fill_between([3.0, 4.0], 0.3, 12, color=GREY, alpha=0.10, lw=0)
    ax[0].plot(g, cons, "-", color=BLUE, lw=3.0, alpha=0.55, solid_capstyle="butt",
               label="conservative, $s_0=2$")
    ax[0].plot(g, over, "-", color=RED, lw=1.4, label="overdamped, $s_0=\\infty$")
    for base, col, mk in (("kuramoto", RED, "o"), ("conservative", BLUE, "s")):
        rows = [r for r in d if r["base"] == base and r["eta"] == 1.0]
        if rows:
            ax[0].plot([r["gamma"] for r in rows], [r["beta_measured"] for r in rows],
                       mk, ms=4.2, color=col, mfc="none", mew=1.1, zorder=4)
    ax[0].axvline(4.0, color=BLUE, lw=0.7, ls=":")
    ax[0].axvline(5.0, color=RED, lw=0.7, ls=":")
    ax[0].text(3.44, 0.44, "same exponent:\ndisorder dominates", fontsize=6.2,
               ha="center", color="#333333")
    ax[0].text(4.05, 1.13, "$\\gamma=4$", fontsize=6.2, color=BLUE)
    ax[0].text(5.05, 0.56, "$\\gamma=5$", fontsize=6.2, color=RED)
    ax[0].set_yscale("log")
    ax[0].set_xlim(3.05, 6.5)
    ax[0].set_ylim(0.35, 12)
    ax[0].set_yticks([0.5, 1, 2, 5, 10])
    ax[0].set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax[0].set_xlabel(r"coupling-disorder exponent $\gamma$")
    ax[0].set_ylabel(r"order-parameter exponent $\beta$")
    ax[0].legend(frameon=False, fontsize=6.4, loc="upper right",
                 handlelength=1.5, borderaxespad=0.3)
    ax[0].set_title("(a)  two dynamics, one criterion", fontsize=8.5, loc="left")

    # (b) the measured exponent against the published one
    pub, meas, lab = [], [], []
    for r in d:
        if r["beta_published"] is not None:
            pub.append(r["beta_published"])
            meas.append(r["beta_measured"])
            lab.append(r["eta"])
    if pub:
        lo, hi = 0.4, max(pub + meas) * 1.4
        ax[1].plot([lo, hi], [lo, hi], "-", color=GREY, lw=1.0, zorder=1)
        e1 = [k for k, e in enumerate(lab) if e == 1.0]
        e2 = [k for k, e in enumerate(lab) if e != 1.0]
        ax[1].plot([pub[k] for k in e1], [meas[k] for k in e1], "o", ms=5,
                   color=RED, mfc="none", mew=1.2, zorder=3,
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
    ax[1].set_title("(b)  the framework reproduces both families",
                    fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig2_disorder")


# ---------------------------------------------------------------- figure 3 --
# ---------------------------------------------------------------- figure 3 --
def figure_convergence():
    mf = load("04_meanfield_check.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    panels = [("grid", "M_core", r"classes $M$", "(a)  detuning grid",
               [200, 400, 800, 1600]),
              ("time", "T", r"run length $T\gamma$", "(b)  run length",
               [50, 100, 200, 400]),
              ("cutoff", "freefac", "free-region factor", "(c)  free region",
               [4, 8, 16, 32])]
    rs = sorted({row["r"] for row in mf["grid"]})
    cols = {r: c for r, c in zip(rs, [BLUE, RED, GREEN, "#8a5cf0"])}
    for k, (key, col, xlab, title, ticks) in enumerate(panels):
        for r in rs:
            rows = sorted([q for q in mf[key] if q["r"] == r], key=lambda q: q[col])
            x = [q[col] for q in rows]
            y = [q["rel_dev"] for q in rows]
            ax[k].plot(x, y, "o-", ms=3.4, lw=0.9, color=cols[r], label=f"$r={r}$")
            sd = [q["osc_sd"] / q["R_law"] for q in rows]
            ax[k].plot(x, sd, ":", lw=0.8, color=cols[r])
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")
        ax[k].set_xlabel(xlab)
        ax[k].set_ylim(1e-6, 1.5)
        ax[k].set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
        ax[k].set_xticks(ticks)
        ax[k].set_xticklabels([str(v) for v in ticks])
        ax[k].minorticks_off()
        ax[k].set_title(title, fontsize=8.5, loc="left")
    ax[0].set_ylabel(r"$|R_{\rm dyn}-R_{\rm law}|/R_{\rm law}$")

    # One legend for the whole figure, placed above the panels, so that no
    # entry can sit on top of a curve in any of the three.
    handles = [matplotlib.lines.Line2D([], [], color=cols[r], marker="o",
                                       ms=3.4, lw=0.9, label=f"$r={r}$")
               for r in rs]
    handles.append(matplotlib.lines.Line2D([], [], color=GREY, ls=":", lw=0.9,
                                           label="dotted: residual oscillation"))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.90))
    fig.legend(handles=handles, frameon=False, fontsize=6.6, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.005),
               handlelength=1.5, columnspacing=1.5)
    save(fig, "fig3_convergence")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3"]
    if "1" in want:
        figure_universality()
    if "2" in want:
        figure_disorder()
    if "3" in want:
        figure_convergence()
