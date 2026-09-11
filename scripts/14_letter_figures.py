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
         a detuned cavity mode, in a single column and in two panels.  It is
         not a sketch.  Panel (a) draws every spin at the tilt the model gives
         for its own detuning, arctan(u), for detunings at evenly spaced
         quantiles of the Gaussian line, and the collective spin at the length
         the self consistency returns.  Panel (b) is the construction that
         produces the kernel: the effective field (Om, 0, delta), the cone the
         oscillator precesses on, the time average along that field, and its
         projection cos^2(alpha) on the drive axis.  Every angle and length in
         panel (b) is checked against the kernel the package computes, in
         tests/test_figure_geometry.py.  It is still not a device drawing,
         because the paper contains no device.

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

# The layout of figL4, kept here rather than inside the drawing code so that
# tests/test_figure_geometry.py checks the same numbers the figure is drawn
# from.  The two panel rectangles overlap: each is wider and taller than the
# drawing it holds, which is what lets a long flat cavity and a cone be drawn
# large on one narrow column.  Overlapping rectangles are only safe while
# neither panel carries a background, because an axes background is opaque
# white and the panel added second would paint over the first.  That is not a
# stylistic point: it took the near lower corner off the cavity in v1.0.6.
FIG4_SIZE = (3.35, 3.02)
FIG4_AXA = [-0.085, 0.452, 1.175, 0.598]
FIG4_AXB = [-0.030, -0.150, 1.060, 0.740]


def fig4_panels(fig):
    """Add the two panels of figL4 to `fig`, with no background on either."""
    axa = fig.add_axes(FIG4_AXA, projection="3d")
    axb = fig.add_axes(FIG4_AXB, projection="3d")
    for ax in (axa, axb):
        ax.patch.set_visible(False)
    return axa, axb


def _fig_xy(fig, ax, xyz):
    """Where a point of a 3-D axes lands, in figure coordinates."""
    from mpl_toolkits.mplot3d import proj3d
    x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], ax.get_proj())
    disp = ax.transData.transform((x2, y2))
    return fig.transFigure.inverted().transform(disp)


def label_at(fig, ax, xyz, text, offset=(0.030, 0.0), **kw):
    """Anchor a label to a point of a 3-D axes, in figure coordinates.

    Nothing is decided about overlap here.  `resolve_label_clashes` measures
    that afterwards, from the glyphs themselves, and moves whatever needs
    moving; keeping a second, weaker opinion in this function only produced
    two answers that disagreed.
    """
    x0, y0 = _fig_xy(fig, ax, xyz)
    return fig.text(x0 + offset[0], y0 + offset[1], text, **kw)


def resolve_label_clashes(fig, dpi=300, max_r=48, step=4, verbose=True):
    """Slide every label off whatever it is sitting on, by measurement.

    Bounding boxes are not trusted anywhere here.  A text artist on a 3-D axes
    reports a stale window extent, so a box based check silently passes labels
    that plainly overlap.  Instead each label's true glyph mask is captured
    once, by rendering the figure with only that label shown and differencing
    against the figure drawn with no labels at all.  A label keeps its exact
    shape when it moves, so a candidate position is then tested by shifting
    that mask in pixels and intersecting it with the ink of the drawing.  One
    render per label buys a search over a whole neighbourhood.

    The search spirals outward from where the label already is, so a label
    that is already clear does not move and one that is not moves as little as
    it can.  Anything still overlapping after `max_r` pixels is reported
    rather than left to be found by a reader.
    """
    import io

    import numpy as _np
    from matplotlib import image as _mimg

    fig.set_dpi(dpi)
    fig.canvas.draw()

    texts = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        texts.extend(t for t in ax.texts
                     if t.get_visible() and t.get_text().strip())
    if not texts:
        return []

    def shot():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        return _mimg.imread(buf)[..., :3]

    for t in texts:
        t.set_visible(False)
    bare = shot()
    ink = bare.min(axis=2) < 0.97
    H, W = ink.shape

    def grow(mask, n):
        out = mask.copy()
        for _ in range(n):
            g = out.copy()
            g[1:, :] |= out[:-1, :]
            g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]
            g[:, :-1] |= out[:, 1:]
            out = g
        return out

    # The glyph mask is widened before searching.  Converting the chosen pixel
    # shift back into a figure fraction rounds, so a position that only just
    # cleared would come back touching; two pixels of margin absorb that.
    masks = []
    for t in texts:
        t.set_visible(True)
        img = shot()
        t.set_visible(False)
        masks.append(grow(_np.abs(img - bare).max(axis=2) > 0.02, 2))
    for t in texts:
        t.set_visible(True)

    def hits(mask, dx, dy):
        """Overlap of the mask, shifted by (dx, dy) pixels, with the ink."""
        sx0, sx1 = max(0, dx), min(W, W + dx)
        mx0, mx1 = max(0, -dx), min(W, W - dx)
        sy0, sy1 = max(0, dy), min(H, H + dy)
        my0, my1 = max(0, -dy), min(H, H - dy)
        if sx1 <= sx0 or sy1 <= sy0:
            return 10 ** 9
        return int((mask[my0:my1, mx0:mx1] & ink[sy0:sy1, sx0:sx1]).sum())

    order = [(0, 0)]
    for r in range(step, max_r + 1, step):
        for k in range(16):
            a = 2 * _np.pi * k / 16
            order.append((int(round(r * _np.cos(a))),
                          int(round(-r * _np.sin(a)))))

    moved, stuck = [], []
    for t, mask in zip(texts, masks):
        if not mask.any():
            continue
        best = None
        for dx, dy in order:
            n = hits(mask, dx, dy)
            if n == 0:
                best = (dx, dy, 0)
                break
            if best is None or n < best[2]:
                best = (dx, dy, n)
        dx, dy, n = best
        if dx or dy:
            x, y = t.get_position()
            # image rows run downward while figure y runs upward, so the
            # vertical shift changes sign on the way back
            t.set_position((x + dx / (fig.get_figwidth() * dpi),
                            y - dy / (fig.get_figheight() * dpi)))
            moved.append((t.get_text(), dx, dy))
        if n:
            stuck.append((t.get_text(), n))

    if verbose:
        for label, dx, dy in moved:
            print("    moved %-26s by (%+d, %+d) px to clear the drawing"
                  % (repr(label), dx, dy))
        for label, n in stuck:
            print("    STILL OVERLAPPING %r, %d px" % (label, n))
    fig.canvas.draw()
    return stuck


def _report_clipping(fig, stem, dpi=300):
    """Say whether anything drawn runs off the edge of the canvas.

    The figure is saved without a tight bounding box, so a label placed too
    near an edge is quietly cut in half rather than making the file larger.
    """
    import io

    import numpy as _np
    from matplotlib import image as _mimg
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    ink = _mimg.imread(buf)[..., :3].min(axis=2) < 0.97
    sides = [("top", ink[0]), ("bottom", ink[-1]),
             ("left", ink[:, 0]), ("right", ink[:, -1])]
    hit = [nm for nm, row in sides if row.any()]
    if hit:
        print("  %s: CLIPPED at the %s edge" % (stem, " and ".join(hit)))
    else:
        ys, xs = _np.nonzero(ink)
        print("  %s: nothing clipped, %d px of margin at the tightest edge"
              % (stem, min(ys.min(), xs.min(),
                           ink.shape[0] - 1 - ys.max(),
                           ink.shape[1] - 1 - xs.max())))
    return hit


def _report_box_edges(fig, ax, stem, hx, hy, hz, dpi=300, tol=0.995):
    """Say whether every edge of the drawn cavity survives to the page.

    A three dimensional panel is a rectangle on the canvas, and the rectangle
    of one panel can reach over the drawing of another.  When it does, the
    axes background of the panel on top paints white over whatever was under
    it, and the result is a box with a corner quietly missing.  Nothing in
    matplotlib reports that, and the label check does not either, because it
    looks for text printed on drawing and not for drawing wiped out.

    So the twelve edges are checked directly: each is projected to canvas
    pixels with the same transform that drew it, and the render is sampled
    along it for the near neutral dark grey the edges are drawn in.  An edge
    that is painted over its whole length is present; anything less is named,
    with the stretch that is missing.
    """
    import io

    import numpy as _np
    from matplotlib import image as _mimg
    from mpl_toolkits.mplot3d import proj3d

    # The data transform reports pixels at the figure's current dpi, and the
    # render below is made at `dpi`.  If the two differ every sample lands in
    # the wrong place and the check reports nothing drawn at all, so the dpi
    # is set here rather than assumed.
    fig.set_dpi(dpi)
    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = _mimg.imread(buf)[..., :3]
    H, W, _ = img.shape

    def to_px(p):
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        col, row = ax.transData.transform((x, y))
        return col, H - row

    corners = {(sx, sy, sz): (sx * hx, sy * hy, sz * hz)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
    edges = [(a, b) for a in corners for b in corners
             if a < b and sum(u != v for u, v in zip(a, b)) == 1]

    def painted(row, col, rad=2):
        r0, r1 = max(0, row - rad), min(H, row + rad + 1)
        c0, c1 = max(0, col - rad), min(W, col + rad + 1)
        if r1 <= r0 or c1 <= c0:
            return False
        patch = img[r0:r1, c0:c1].reshape(-1, 3)
        dark = patch.sum(axis=1) < 2.2            # darker than a light grey
        flat = (patch.max(axis=1) - patch.min(axis=1)) < 0.10   # near neutral
        return bool((dark & flat).any())

    worst, bad = 1.0, []
    for a, b in edges:
        pa, pb = to_px(corners[a]), to_px(corners[b])
        ts = _np.linspace(0.0, 1.0, 201)
        hit = [painted(int(round(pa[1] + t * (pb[1] - pa[1]))),
                       int(round(pa[0] + t * (pb[0] - pa[0])))) for t in ts]
        frac = sum(hit) / len(hit)
        worst = min(worst, frac)
        if frac < tol:
            miss = [round(float(t), 2) for t, h in zip(ts, hit) if not h]
            bad.append((a, b, frac, miss[0], miss[-1]))

    if bad:
        for a, b, frac, t0, t1 in bad:
            print("  %s: EDGE %s to %s only %.0f%% drawn, missing between "
                  "%.2f and %.2f of its length" % (stem, a, b, 100 * frac,
                                                   t0, t1))
    else:
        print("  %s: all %d edges of the cavity drawn in full, %.1f%% at the "
              "thinnest" % (stem, len(edges), 100 * worst))
    return bad


def _report_title_clearance(fig, stem, titles, dpi=300):
    """Measure the white space under each panel title, in pixels.

    A title that clears the drawing by one pixel passes an overlap check and
    still reads as crowded.  This measures the real gap: the title's own glyph
    pixels are recovered by differencing, and the drawing below is searched,
    column by column under those glyphs, for its first ink.
    """
    import io

    import numpy as _np
    from matplotlib import image as _mimg

    fig.set_dpi(dpi)
    fig.canvas.draw()

    def shot():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        return _mimg.imread(buf)[..., :3]

    keep = [(t, t.get_visible()) for t in titles]
    for t, _ in keep:
        t.set_visible(False)
    bare = shot()
    for t, v in keep:
        t.set_visible(v)
    rest = bare.min(axis=2) < 0.97
    H, W = rest.shape

    gaps = []
    for t, _ in keep:
        t.set_visible(False)
        without = shot()
        t.set_visible(True)
        img = shot()
        glyph = _np.abs(img - without).max(axis=2) > 0.02
        if not glyph.any():
            continue
        rows, cols = _np.nonzero(glyph)
        bottom = rows.max()
        under = rest[bottom + 1:, cols.min():cols.max() + 1]
        if under.any():
            gap = int(_np.nonzero(under.any(axis=1))[0][0])
        else:
            gap = H - bottom - 1
        gaps.append((t.get_text(), gap))

    for label, gap in gaps:
        print("  %s: %r clears the drawing below it by %d px"
              % (stem, label, gap))
    return gaps


def report_label_clashes(fig, stem, dpi=300, grow=1, tol_px=0):
    """Say whether any text in the figure is printed on top of drawn content.

    Bounding boxes are not used.  A text artist on a 3-D axes reports a stale
    window extent, so a box based check silently passes labels that plainly
    overlap; this compares pixels instead, which cannot be fooled.

    The figure is rendered once as it is, once with every text hidden, and
    once per text with only that text hidden.  Differencing the last against
    the first gives the exact pixels that text paints.  If any of those
    pixels carry ink in the image with no text at all, the label is sitting on
    something.  `grow` dilates the label mask by that many pixels so that a
    glyph merely touching a line is also caught.
    """
    import io

    import numpy as _np
    from matplotlib import image as _mimg

    fig.set_dpi(dpi)
    fig.canvas.draw()

    texts = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        texts.extend(t for t in ax.texts
                     if t.get_visible() and t.get_text().strip())

    def shot():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        return _mimg.imread(buf)[..., :3]

    full = shot()
    for t in texts:
        t.set_visible(False)
    bare = shot()
    for t in texts:
        t.set_visible(True)

    ink = bare.min(axis=2) < 0.97          # anything drawn that is not paper

    def dilate(mask, n):
        out = mask.copy()
        for _ in range(n):
            g = out.copy()
            g[1:, :] |= out[:-1, :]
            g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]
            g[:, :-1] |= out[:, 1:]
            out = g
        return out

    bad = []
    for t in texts:
        t.set_visible(False)
        without = shot()
        t.set_visible(True)
        mine = _np.abs(without - full).max(axis=2) > 0.02
        if not mine.any():
            continue
        hit = dilate(mine, grow) & ink
        n = int(hit.sum())
        if n > tol_px:
            ys, xs = _np.nonzero(hit)
            bad.append((t.get_text(), n, int(mine.sum()),
                        (int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max()))))

    if bad:
        print("  LABEL CLASHES in %s:" % stem)
        for label, n, area, box in sorted(bad, key=lambda r: -r[1]):
            print("    %-26s %5d px on ink, of %5d px of glyph,  at %s"
                  % (repr(label), n, area, box))
    else:
        print("  %s: %d labels checked, none printed on drawn content"
              % (stem, len(texts)))
    return bad


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


def _precession_frame(u):
    """The exact single oscillator geometry, taken straight from the paper.

    The supplement states it in one sentence: a unit vector precesses about
    the effective field (Om, 0, delta); the projection of the field direction
    on the drive axis is Om/sqrt(Om^2 + delta^2), and the time averaged
    projection of the oscillator on the drive axis is the square of that.

    Work in units of Om, so that delta/Om = u.  Put the drive axis, which is
    the direction of the collective field, along x, and the detuning axis
    along z.  The effective field is then (1, 0, u), its tilt from the drive
    axis is alpha = arctan(u), and an oscillator prepared along the drive axis
    lies on a cone of half angle alpha about it.  Returns the unit field
    direction, alpha, the oscillator at precession phase phi, and the
    component along the field, which is the time average.

    Every relation used here is checked against the kernel the package
    computes, in tests/test_figure_geometry.py.
    """
    B = np.array([1.0, 0.0, u])
    n = B / np.linalg.norm(B)
    alpha = np.arctan2(u, 1.0)
    x = np.array([1.0, 0.0, 0.0])
    par = np.dot(x, n) * n            # the component along the field
    perp = x - par                    # the radius of the cone
    w = np.cross(n, perp)             # completes the right handed pair

    def spin(phi):
        return par + perp * np.cos(phi) + w * np.sin(phi)

    return n, alpha, spin, par


def _cavity_box(ax, LX=4.4, LY=2.0, LZ=2.0):
    """The cavity: a closed volume carrying one standing wave mode.

    The mode drawn is the fundamental, one half wavelength between the end
    walls, so the envelope is cos(pi x / LX) and vanishes on both walls.  The
    exchange in the model is uniform and all to all, so nothing is meant by
    where an individual spin sits inside the box.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    hx, hy, hz = LX / 2, LY / 2, LZ / 2

    yy, zz = np.meshgrid([-hy, hy], [-hz, hz])
    for sx in (-hx, hx):
        ax.plot_surface(np.full_like(yy, sx), yy, zz, color=GREY, alpha=0.20,
                        shade=False, edgecolor="none", zorder=0)
    xx, yy2 = np.meshgrid([-hx, hx], [-hy, hy])
    ax.plot_surface(xx, yy2, np.full_like(xx, -hz), color=GREY, alpha=0.10,
                    shade=False, edgecolor="none", zorder=0)

    c = np.array([[sx * hx, sy * hy, sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [[c[i], c[j]] for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(c[i] != c[j]) == 1]
    ax.add_collection3d(Line3DCollection(edges, colors="#4d4d4d",
                                         linewidths=0.9))

    xs = np.linspace(-hx, hx, 240)
    env = 0.86 * np.cos(np.pi * xs / LX)
    poly = np.concatenate([np.stack([xs, env], 1),
                           np.stack([xs[::-1], -env[::-1]], 1)])
    ax.add_collection3d(Poly3DCollection(
        [[(q[0], -hy + 0.02, q[1]) for q in poly]],
        facecolor=BLUE, alpha=0.15, edgecolor=BLUE, linewidths=0.7))
    for x0 in np.linspace(-hx + 0.55, hx - 0.55, 7):
        a = 0.86 * np.cos(np.pi * x0 / LX)
        ax.quiver(x0, -hy + 0.02, -a, 0, 0, 2 * a, color=BLUE, lw=0.7,
                  alpha=0.5, arrow_length_ratio=0.10, zorder=0)
    return hx, hy, hz


def letter_fig4():
    """The cavity realization, and the geometry that makes the kernel algebraic.

    Panel (a) is the ensemble in the cavity.  Each spin is drawn at the tilt
    the model gives for its own detuning, alpha = arctan(u), and not at a
    random one; the detunings are the evenly spaced quantiles of the Gaussian
    line, so the spread drawn is the spread the calculation uses.  The
    collective spin carries the length the self consistency returns for that
    line at that locking bandwidth.

    Panel (b) is one oscillator, and is the reason the kernel is algebraic.
    It shows the effective field (Om, 0, delta), the cone the oscillator
    precesses on, the time average along that field, and the projection of
    that average on the drive axis, which is cos^2 alpha and therefore the
    kernel.  The panel is drawn at u = 1, where the tilt is exactly 45 degrees
    and the projection exactly one half.

    There is no locked and drifting distinction in this panel.  That dichotomy
    belongs to the overdamped kernel, which is identically zero beyond u = 1.
    For this case the paper says the opposite, that no detuning releases such
    an oscillator completely, so the spins are shaded continuously by detuning
    instead of being split into two classes.
    """
    import mpmath as mp
    from matplotlib import colors as mcolors
    from scipy.special import erfinv
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from lockkernel import kernels as K, lineshapes as L, parametric as P
    mp.mp.dps = 20

    cons = K.conservative()
    line = L.gaussian(1.0)
    Om = mp.mpf(1)
    _, R, _ = P.branch_point(line, cons, Om)
    R = float(R)

    fig = plt.figure(figsize=FIG4_SIZE)
    axa, axb = fig4_panels(fig)

    # ------------------------------------------------------------ panel (a)
    hx, hy, hz = _cavity_box(axa)

    nspin = 15
    qs = (np.arange(nspin) + 0.5) / nspin
    deltas = np.sqrt(2.0) * erfinv(2 * qs - 1)        # Gaussian line, sigma = 1
    us = deltas / float(Om)

    # Shaded by |detuning| on a single red scale.  A diverging scale would put
    # blue on the spins, which is the colour of the mode, so the two would be
    # read as the same thing.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tilt", ["#f4a0a0", "#d94a4a", RED, "#8d1616", "#5d0f0f"])
    umax = float(np.max(np.abs(us)))

    xg = np.linspace(-1.58, 1.22, 5)
    yg = np.array([0.22, 0.92])
    zg = np.array([-0.54, 0.40])
    sites = [(x, y, z) for z in zg for y in yg for x in xg][:nspin]
    Ls = 0.58
    for k in np.argsort(np.abs(us))[::-1]:
        x0, y0, z0 = sites[k]
        _, _, spin, _ = _precession_frame(us[k])
        v = spin(0.62 * np.pi * (k % 3) + 0.30)
        col = cmap(abs(us[k]) / umax)
        axa.quiver(x0, y0, z0, Ls * v[0], Ls * v[1], Ls * v[2],
                   color=col, lw=1.2, arrow_length_ratio=0.34, zorder=4)

    Sx0, Sy0, Sz0 = -1.70, 0.54, 0.90
    axa.quiver(Sx0, Sy0, Sz0, 4.10 * R, 0.0, 0.0, color=GREEN, lw=2.4,
               arrow_length_ratio=0.15, zorder=6)
    axa.set_xlim(-2.3, 2.3)
    axa.set_ylim(-1.15, 1.15)
    axa.set_zlim(-1.15, 1.15)
    axa.set_box_aspect((4.6, 2.3, 2.3))
    axa.view_init(elev=17, azim=-62)
    axa.set_axis_off()

    # ------------------------------------------------------------ panel (b)
    u0 = 1.0
    n, alpha, spin, _ = _precession_frame(u0)
    cosa = float(np.cos(alpha))
    Wc = float(cons.W(mp.mpf(u0)))

    GOLD, DARK = "#b8860b", "#3a3a3a"

    # A fan of the same field at other detunings was tried here and removed.
    # It filled the wedge between the drive axis and the field, which is the
    # one place the tilt can be labelled, and the tilt family is in any case
    # already on view in panel (a), where each spin carries its own detuning.

    # the drive axis, and the field at u = 1 drawn unnormalised so that its
    # two legs are exactly Om along the drive axis and delta along the other
    axb.quiver(0, 0, 0, 1.34, 0, 0, color=GREEN, lw=1.8,
               arrow_length_ratio=0.12, zorder=3)
    axb.quiver(0, 0, 0, 1.0, 0, u0, color=GOLD, lw=1.5,
               arrow_length_ratio=0.13, zorder=5)
    axb.plot([0, 1.0], [0, 0], [u0, u0], color=GOLD, lw=0.7,
             ls=(0, (2.2, 1.8)), alpha=0.75, zorder=4)
    axb.plot([1.0, 1.0], [0, 0], [0, u0], color=GOLD, lw=0.7,
             ls=(0, (2.2, 1.8)), alpha=0.75, zorder=4)

    # the cone the oscillator precesses on.  Only the locus and two extreme
    # generators are drawn: a full fan of generators fills the wedge between
    # the drive axis and the field, which is where the tilt has to be labelled.
    phi = np.linspace(0, 2 * np.pi, 400)
    cone = np.array([spin(p) for p in phi])
    axb.plot(cone[:, 0], cone[:, 1], cone[:, 2], color=RED, lw=1.0,
             alpha=0.9, zorder=4)
    for p in (0.5 * np.pi, 1.5 * np.pi):
        v = spin(p)
        axb.plot([0, v[0]], [0, v[1]], [0, v[2]], color=RED, lw=0.5,
                 alpha=0.30, zorder=3)
    v0 = spin(0.86 * np.pi)
    axb.quiver(0, 0, 0, v0[0], v0[1], v0[2], color=RED, lw=1.7,
               arrow_length_ratio=0.16, zorder=6)

    # the time average: the component along the field, of length cos alpha
    avg = cosa * n
    axb.quiver(0, 0, 0, avg[0], avg[1], avg[2], color="#1a7f37", lw=2.4,
               arrow_length_ratio=0.22, zorder=7)
    axb.plot([avg[0], avg[0]], [0, 0], [0, avg[2]], color=DARK,
             lw=0.9, ls=(0, (2.4, 2.0)), zorder=5)
    axb.plot([avg[0]], [0], [0], marker="o", ms=3.6, color=DARK, zorder=8)

    arc = np.linspace(0, alpha, 60)
    r = 0.42
    axb.plot(r * np.cos(arc), np.zeros_like(arc), r * np.sin(arc),
             color=GOLD, lw=1.0, zorder=5)

    axb.set_xlim(-0.18, 1.40)
    axb.set_ylim(-0.82, 0.82)
    axb.set_zlim(-0.42, 1.24)
    axb.set_box_aspect((1.58, 1.64, 1.66))
    axb.view_init(elev=15, azim=-68)
    axb.set_axis_off()

    # ------------------------------------------------------------ labels
    ta = fig.text(0.012, 0.962, "(a)", fontsize=8.5)
    ta2 = fig.text(0.082, 0.962, "one mode, one collective field",
                   fontsize=7.2, color="#333333")
    tb = fig.text(0.012, 0.452, "(b)", fontsize=8.5)
    tb2 = fig.text(0.082, 0.452, "why the kernel is algebraic",
                   fontsize=7.2, color="#333333")

    fig.text(0.012, 0.512, r"cavity mode $\omega_c$", fontsize=7.0,
             color=BLUE, ha="left")
    fig.text(0.988, 0.512, r"spins shaded by $|\delta|$", fontsize=7.0,
             color=RED, ha="right")
    fig.text(0.012, 0.010, "drive axis", fontsize=7.0, color=GREEN,
             ha="left")
    fig.text(0.988, 0.010, r"$\cos^{2}\alpha=W_{\rm c}(u)$", fontsize=7.2,
             color=DARK, ha="right")

    # ---- the labels that sit on the drawing --------------------------------
    # Each is anchored to the point of the construction it names.  Overlap is
    # not judged here: resolve_label_clashes measures it from the rendered
    # glyphs afterwards and moves whatever needs moving.
    fig.canvas.draw()
    label_at(fig, axa, (Sx0 + 4.10 * R, Sy0, Sz0), r"$\mathbf{S}$",
             color=GREEN, fontsize=9.5, ha="center", va="center")
    label_at(fig, axb, tuple(1.04 * np.array([1.0, 0.0, u0])),
                r"$(\Omega,0,\delta)$", color=GOLD, fontsize=7.8,
                ha="center", va="center")
    label_at(fig, axb, (0.5, 0.0, u0), r"$\Omega$",
                color=GOLD, fontsize=8.5, ha="center", va="center")
    label_at(fig, axb, (1.0, 0.0, 0.5 * u0), r"$\delta$",
                color=GOLD, fontsize=8.5, ha="center", va="center")
    label_at(fig, axb,
             (0.56 * np.cos(alpha / 2), 0.0, 0.56 * np.sin(alpha / 2)),
             r"$\alpha$", offset=(0.0, 0.0), color=GOLD, fontsize=8.8,
             ha="center", va="center")
    label_at(fig, axb, tuple(v0), r"$\mathbf{s}$",
                color=RED, fontsize=8.8, ha="center", va="center")
    label_at(fig, axb, tuple(avg), r"$\langle\mathbf{s}\rangle$",
                color="#1a7f37", fontsize=8.4, ha="center", va="center")
    resolve_label_clashes(fig)
    report_label_clashes(fig, "figL4_realization")
    _report_box_edges(fig, axa, "figL4_realization", hx, hy, hz)
    _report_title_clearance(fig, "figL4_realization", [ta2, tb2])
    _report_clipping(fig, "figL4_realization")
    out = os.path.join(F, "figL4_realization")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print("saved", out + ".pdf and .png")
    print("  panel (b) at u = %.1f: alpha = %.6f deg, cos^2 alpha = %.12f, "
          "W_c from package = %.12f, deviation %.2e"
          % (u0, np.degrees(alpha), cosa ** 2, Wc, abs(cosa ** 2 - Wc)))
    print("  panel (a): R from the self consistency = %.8f" % R)
    print("  panel (a): detunings span u = %.4f to %.4f" % (us.min(), us.max()))


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
