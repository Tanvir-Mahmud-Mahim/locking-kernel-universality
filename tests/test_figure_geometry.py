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
def frame():
    """The helper the figure script actually uses."""
    path = os.path.join(ROOT, "scripts", "14_letter_figures.py")
    spec = importlib.util.spec_from_file_location("figL", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._precession_frame


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
