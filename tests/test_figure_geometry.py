"""The geometry drawn in figL4 is the geometry the package computes.

Figure figL4 of the Letter is not a sketch: panel (b) is a construction whose
every length and angle is fixed by the model.  The paper states the geometry
in one sentence, in the supplement: a unit vector precesses about the
effective field (Om, 0, delta); the projection of the field direction on the
drive axis is Om/sqrt(Om^2 + delta^2), and the time averaged projection of the
oscillator on the drive axis is the square of that.

These tests close the loop between that sentence, the drawing, and the kernel
`lockkernel.kernels.conservative`, so that the figure cannot drift away from
the physics it claims to show.  They import the drawing helper itself rather
than a copy of it.
"""
import importlib.util
import os

import mpmath as mp
import numpy as np
import pytest

from lockkernel import kernels as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

US = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.7, 10.0, 100.0]


@pytest.fixture(scope="module")
def frame_mod():
    """The figure script itself, so the tests read the drawing code."""
    path = os.path.join(ROOT, "scripts", "14_letter_figures.py")
    spec = importlib.util.spec_from_file_location("figL", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def frame(frame_mod):
    """The helper the figure script actually uses."""
    return frame_mod._precession_frame


@pytest.mark.parametrize("u", US)
def test_tilt_is_arctan_u(frame, u):
    """The tilt of the precession axis away from the drive axis is arctan(u)."""
    _, alpha, _, _ = frame(u)
    assert alpha == pytest.approx(np.arctan(u), abs=1e-14)
    assert np.tan(alpha) == pytest.approx(u, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("u", US)
def test_cosine_matches_the_papers_formula(frame, u):
    """cos(alpha) is Om/sqrt(Om^2 + delta^2), in units where Om = 1."""
    _, alpha, _, _ = frame(u)
    assert np.cos(alpha) == pytest.approx(1.0 / np.hypot(1.0, u), abs=1e-14)


@pytest.mark.parametrize("u", US)
def test_square_of_the_cosine_is_the_kernel(frame, u):
    """The drawn projection cos^2(alpha) is the kernel the package returns."""
    _, alpha, _, _ = frame(u)
    assert np.cos(alpha) ** 2 == pytest.approx(
        float(K.conservative().W(mp.mpf(u))), abs=1e-14)


@pytest.mark.parametrize("u", US)
def test_time_average_over_the_cone_is_the_kernel(frame, u):
    """Average the drawn oscillator over a full precession, do not assume it.

    This is the check that matters: it takes the cone the figure draws, walks
    a whole turn around it, and averages the projection on the drive axis.  It
    never uses the closed form, so agreement with the kernel is a statement
    about the drawing and not an identity.
    """
    _, _, spin, _ = frame(u)
    phi = np.linspace(0.0, 2.0 * np.pi, 20001)[:-1]
    proj = np.array([spin(p)[0] for p in phi])
    assert proj.mean() == pytest.approx(
        float(K.conservative().W(mp.mpf(u))), abs=2e-12)


@pytest.mark.parametrize("u", US)
def test_the_drawn_oscillator_stays_a_unit_vector(frame, u):
    """A spin one half points somewhere on the sphere at every instant."""
    _, _, spin, _ = frame(u)
    for p in np.linspace(0.0, 2.0 * np.pi, 37):
        assert np.linalg.norm(spin(p)) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("u", US)
def test_cone_half_angle_equals_the_tilt(frame, u):
    """The oscillator starts on the drive axis, so the cone opens by alpha."""
    n, alpha, spin, par = frame(u)
    for p in np.linspace(0.0, 2.0 * np.pi, 25):
        assert float(np.dot(spin(p), n)) == pytest.approx(
            np.cos(alpha), abs=1e-13)
    assert np.linalg.norm(par) == pytest.approx(np.cos(alpha), abs=1e-13)


@pytest.mark.parametrize("u", US)
def test_time_average_lies_along_the_field(frame, u):
    """The average of the drawn cone is the component along the field.

    The figure draws <s> as an arrow along the field of length cos(alpha).
    This checks that the cone really averages to that vector, so the arrow is
    not merely plausible.
    """
    n, alpha, spin, _ = frame(u)
    phi = np.linspace(0.0, 2.0 * np.pi, 20001)[:-1]
    avg = np.array([spin(p) for p in phi]).mean(axis=0)
    assert np.linalg.norm(avg - np.cos(alpha) * n) < 2e-12


def test_no_detuning_releases_a_precessing_oscillator():
    """The claim the figure is built on, and why it has no drifting class.

    The Letter says of the precessing kernel that no detuning releases such an
    oscillator completely, in contrast to the overdamped one, which is
    identically zero beyond u = 1.  A figure of the cavity setting must
    therefore not split the ensemble into locked and drifting.
    """
    cons, kur = K.conservative(), K.kuramoto()
    for u in (1.0, 1.5, 5.0, 50.0, 500.0):
        assert float(cons.W(mp.mpf(u))) > 0.0
    for u in (1.0000001, 1.5, 5.0):
        assert float(kur.W(mp.mpf(u))) == 0.0


def _cavity_panel_render(frame_mod, opaque_backgrounds=False, dpi=200):
    """Draw the cavity panel of figL4 in its real layout and render it.

    The second panel is added as well, with the rectangle the figure gives it,
    because the defect this guards against is not in either panel on its own.
    It is in how the two overlap.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mimg

    fig = plt.figure(figsize=frame_mod.FIG4_SIZE)
    axa, axb = frame_mod.fig4_panels(fig)
    if opaque_backgrounds:
        for ax in (axa, axb):
            ax.patch.set_visible(True)
            ax.patch.set_facecolor("white")
            ax.patch.set_alpha(1.0)
    hx, hy, hz = frame_mod._cavity_box(axa)
    axa.set_xlim(-2.3, 2.3)
    axa.set_ylim(-1.15, 1.15)
    axa.set_zlim(-1.15, 1.15)
    axa.set_box_aspect((4.6, 2.3, 2.3))
    axa.view_init(elev=17, azim=-62)
    axa.set_axis_off()
    axb.set_axis_off()
    fig.set_dpi(dpi)
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = mimg.imread(buf)[..., :3]
    return fig, axa, (hx, hy, hz), img


def _edge_coverage(fig, ax, half, img):
    """Fraction of each of the twelve cavity edges that reached the page."""
    from mpl_toolkits.mplot3d import proj3d
    hx, hy, hz = half
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
        dark = patch.sum(axis=1) < 2.2
        flat = (patch.max(axis=1) - patch.min(axis=1)) < 0.10
        return bool((dark & flat).any())

    out = {}
    for a, b in edges:
        pa, pb = to_px(corners[a]), to_px(corners[b])
        ts = np.linspace(0.0, 1.0, 161)
        hit = [painted(int(round(pa[1] + t * (pb[1] - pa[1]))),
                       int(round(pa[0] + t * (pb[0] - pa[0])))) for t in ts]
        out[(a, b)] = sum(hit) / len(hit)
    return out


def test_every_edge_of_the_cavity_reaches_the_page(frame_mod):
    """The cavity is drawn closed, and stays closed once rendered.

    Figure 3(a) is a box, and a box with a corner missing is wrong on the
    page whatever the code intended.  The twelve edges are projected with the
    transform that drew them and the render is sampled along each one.
    """
    import matplotlib.pyplot as plt
    fig, axa, half, img = _cavity_panel_render(frame_mod)
    try:
        cover = _edge_coverage(fig, axa, half, img)
    finally:
        plt.close(fig)
    assert len(cover) == 12
    missing = {e: f for e, f in cover.items() if f < 0.995}
    assert not missing, "edges not fully drawn: %s" % missing


def test_an_opaque_panel_background_would_remove_a_corner(frame_mod):
    """The defect this guards against, shown to be real and not theoretical.

    The two panel rectangles of the figure overlap. With backgrounds turned
    on, the panel added second paints white over the bottom of the first and
    the near lower corner of the cavity disappears. The check above is
    therefore worth running, and the backgrounds must stay off.
    """
    import matplotlib.pyplot as plt
    fig, axa, half, img = _cavity_panel_render(frame_mod,
                                               opaque_backgrounds=True)
    try:
        cover = _edge_coverage(fig, axa, half, img)
    finally:
        plt.close(fig)
    assert any(f < 0.995 for f in cover.values()), (
        "an opaque background no longer hides anything, so this test and the "
        "note in the figure script are both out of date")


def test_the_panels_carry_no_background(frame_mod):
    """The invariant behind the two tests above, stated directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=frame_mod.FIG4_SIZE)
    try:
        for ax in frame_mod.fig4_panels(fig):
            assert not ax.patch.get_visible()
    finally:
        plt.close(fig)


def test_the_panel_rectangles_really_do_overlap(frame_mod):
    """If they ever stop overlapping, the tests above stop meaning anything."""
    ax, bx = frame_mod.FIG4_AXA, frame_mod.FIG4_AXB
    a = (ax[0], ax[1], ax[0] + ax[2], ax[1] + ax[3])
    b = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
    assert a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
