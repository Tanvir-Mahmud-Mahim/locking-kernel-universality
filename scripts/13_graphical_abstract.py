"""Graphical abstract.

Three panels.  (a) is a schematic of the one physical setting the paper names,
a spin ensemble coupled to a detuned cavity mode; it carries no numbers and is
not a device drawing, because the paper contains no device.  (b) and (c) are
computed: (b) evaluates the two locking kernels directly from the package, and
(c) reads the measured exponents from data/02_universality_line.json.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

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


# ----------------------------------------------------------------- panel (a)

def box_edges(lx, ly, lz):
    """The twelve edges of a box centred on the origin."""
    x, y, z = lx / 2, ly / 2, lz / 2
    c = np.array([[sx * x, sy * y, sz * z]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(c[i] != c[j]) == 1:
                edges.append([c[i], c[j]])
    return edges


def draw_cavity(ax):
    LX, LY, LZ = 4.4, 2.0, 2.0
    hx, hy, hz = LX / 2, LY / 2, LZ / 2

    # the back and bottom walls, shaded, so the box reads as a closed volume
    yy, zz = np.meshgrid([-hy, hy], [-hz, hz])
    ax.plot_surface(np.full_like(yy, -hx), yy, zz, color=GREY, alpha=0.20,
                    shade=False, edgecolor="none", zorder=0)
    ax.plot_surface(np.full_like(yy, hx), yy, zz, color=GREY, alpha=0.20,
                    shade=False, edgecolor="none", zorder=0)
    xx, yy2 = np.meshgrid([-hx, hx], [-hy, hy])
    ax.plot_surface(xx, yy2, np.full_like(xx, -hz), color=GREY, alpha=0.10,
                    shade=False, edgecolor="none", zorder=0)
    ax.add_collection3d(Line3DCollection(box_edges(LX, LY, LZ),
                                         colors="#4d4d4d", linewidths=0.9))

    # the mode: the field of a standing wave with one half wavelength between
    # the end walls, drawn as vertical field arrows whose length follows
    # cos(pi x / L) and vanishes at the walls
    xs = np.linspace(-hx, hx, 200)
    env = 0.88 * np.cos(np.pi * xs / LX)
    poly = np.concatenate([np.stack([xs, env], 1),
                           np.stack([xs[::-1], -env[::-1]], 1)])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [[(p[0], -hy + 0.02, p[1]) for p in poly]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor=BLUE, alpha=0.16,
                                         edgecolor=BLUE, linewidths=0.8))
    for x0 in np.linspace(-hx + 0.55, hx - 0.55, 7):
        a = 0.88 * np.cos(np.pi * x0 / LX)
        ax.quiver(x0, -hy + 0.02, -a, 0, 0, 2 * a, color=BLUE, lw=0.8,
                  alpha=0.55, arrow_length_ratio=0.11, zorder=0)

    # the ensemble: each spin precesses about the cavity axis at its own
    # detuning, so the tilts differ; the ones that lock stay near the axis
    rng = np.random.default_rng(11)
    xg = np.linspace(-1.55, 1.15, 4)
    yg = np.array([0.18, 0.72])
    zg = np.array([-0.55, 0.20])
    pts = [(x + rng.uniform(-0.14, 0.14), y + rng.uniform(-0.10, 0.10),
            z + rng.uniform(-0.12, 0.12))
           for x in xg for y in yg for z in zg]
    L = 0.58
    for k, (x0, y0, z0) in enumerate(pts):
        tilt = rng.uniform(0.18, 1.35)
        phase = rng.uniform(0, 2 * np.pi)
        locked = tilt < 0.80
        ax.quiver(x0, y0, z0,
                  L * np.cos(tilt),
                  L * np.sin(tilt) * np.cos(phase),
                  L * np.sin(tilt) * np.sin(phase),
                  color=RED if locked else "#9a9a9a",
                  alpha=0.95 if locked else 0.75,
                  lw=1.15 if locked else 0.9,
                  arrow_length_ratio=0.36, zorder=4)

    # the collective spin, along the cavity axis
    ax.quiver(-1.45, 0.50, 0.80, 2.90, 0.0, 0.0, color=GREEN, lw=2.4,
              arrow_length_ratio=0.13, zorder=6)
    ax.text(1.58, 0.50, 0.96, r"$\mathbf{S}$", color=GREEN, fontsize=11,
            zorder=7)

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((4.6, 2.3, 2.3))
    ax.view_init(elev=17, azim=-62)
    ax.set_axis_off()


# ----------------------------------------------------------------- the figure

def main():
    from lockkernel import kernels as K

    rec = json.load(open(os.path.join(D, "02_universality_line.json")))
    curve = rec["curve"]

    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.52, 1.0, 1.0],
                          left=0.0, right=0.985, bottom=0.17, top=0.98,
                          wspace=0.42)

    axa = fig.add_subplot(gs[0, 0], projection="3d")
    draw_cavity(axa)
    axa.set_position([-0.015, -0.01, 0.415, 1.00])
    axa.text2D(0.03, 0.93, "(a)", transform=axa.transAxes, fontsize=10)
    axa.text2D(0.05, 0.235, r"cavity mode $\omega_c$",
               transform=axa.transAxes, fontsize=8.5, color=BLUE)
    axa.text2D(0.46, 0.845, r"spins detuned by $\delta$",
               transform=axa.transAxes, fontsize=8.5, color=RED)
    axa.text2D(0.55, 0.135, "locked", transform=axa.transAxes, fontsize=8,
               color=RED)
    axa.text2D(0.74, 0.135, "drifting", transform=axa.transAxes, fontsize=8,
               color="#8a8a8a")

    # ---- (b) the two kernels
    axb = fig.add_subplot(gs[0, 1])
    u = np.logspace(-1.6, 1.6, 700)
    cons = np.array([float(K.conservative().W(x)) for x in u])
    kur = np.array([float(K.kuramoto().W(x)) for x in u])
    axb.loglog(u, cons, color=RED, lw=1.4)
    kur_masked = np.where(kur > 0, kur, np.nan)
    axb.loglog(u, kur_masked, color=BLUE, lw=1.4)
    axb.loglog([1.0, 1.0], [3e-4, float(K.kuramoto().W(0.999))], color=BLUE,
               lw=1.4)
    axb.loglog(u[u > 2], 1.0 / u[u > 2] ** 2, color=GREY, lw=0.7,
               ls=(0, (4, 3)))
    axb.set_xlim(3e-2, 40)
    axb.set_ylim(3e-4, 4.0)
    axb.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    axb.set_xlabel(r"detuning $u$, in locked widths")
    axb.set_ylabel(r"locking kernel $W(u)$")
    axb.text(0.03, 0.93, "(b)", transform=axb.transAxes, fontsize=10)
    axb.text(3.0, 0.22, r"$|u|^{-s}$", color=RED, fontsize=9)
    axb.text(0.038, 0.0055, "compact\nsupport", color=BLUE, fontsize=8.5)
    axb.text(0.30, 0.030, "precessing", color=RED, fontsize=8.5)

    # ---- (c) the exponent
    axc = fig.add_subplot(gs[0, 2])
    s = np.array([p["s"] for p in curve])
    beta = np.array([p["beta"] for p in curve])
    ss = np.linspace(1.25, 6.0, 500)
    pred = np.where(ss < 3.0, 1.0 / (ss - 1.0), 0.5)
    axc.plot(ss, pred, color=GREY, lw=1.0)
    axc.plot(s, beta, ls="none", marker="o", ms=2.6, mfc="none",
             mec=RED, mew=0.8)
    axc.plot([3.0, 3.0], [0.0, 1.72], color=GREY, lw=0.6, ls=(0, (3, 3)))
    axc.set_xlim(1.3, 6.2)
    axc.set_ylim(0.0, 3.0)
    axc.set_xlabel(r"kernel tail exponent, $s$")
    axc.set_ylabel(r"order parameter exponent, $\beta$")
    axc.text(0.03, 0.93, "(c)", transform=axc.transAxes, fontsize=10)
    axc.text(1.95, 2.20, r"$\beta=\dfrac{1}{s-1}$", color="#333333",
             fontsize=9.5)
    axc.text(4.15, 0.80, r"$\beta=1/2$", color="#333333", fontsize=9.5)
    axc.text(3.10, 1.78, r"$s=3$", color=GREY, fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, "fig0_graphical_abstract." + ext),
                    dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote figures/fig0_graphical_abstract.pdf and .png")


if __name__ == "__main__":
    main()
