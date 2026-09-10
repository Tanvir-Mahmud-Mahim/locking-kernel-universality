"""Step 9.  The averaged kernel against a quenched simulation.

The treatment of coupling disorder in Sec. V of the paper replaces the
population by a distribution: each oscillator's weight enters only through
P(k), and no correlation between the weights of interacting oscillators is
kept.  That is an annealed treatment, and it is the same approximation under
which the results it reproduces were derived, so agreement with them is a
consistency check within a common approximation and not a test of the
approximation itself.

This script supplies the missing test.  It simulates the actual dynamics of a
finite population with QUENCHED weights and QUENCHED detunings, drawn once and
held fixed, and measures the locking kernel directly rather than assuming it:
for each oscillator it records the time averaged projection on the collective
field, then bins those projections by detuning.  The prediction is

    Wt(v) = (1/<k>) int P(k) k W(v/k^eta) dk,      v = delta / F,

with F the amplitude of the collective field, which is also measured from the
simulation rather than imposed.  Nothing in the comparison is fitted.

Both dynamics are simulated.

  OVERDAMPED.  d theta_i / dt = delta_i + k_i F sin(psi - theta_i), with the
  field amplitude and phase recomputed from the population at every step.

  PRECESSING.  Each oscillator is a unit vector obeying dv_i/dt = B_i x v_i
  with B_i = (k_i F_x, k_i F_y, delta_i), again with the field recomputed at
  every step.  This is the conservative case, and it is the one for which the
  prediction of the paper is new.

The order parameter of the simulation is compared with the exact parametric
branch at the same coupling as well, which tests the self consistency itself
and not only the kernel inside it.

DISORDER AVERAGING.  The weights are drawn from a distribution whose third
moment diverges for the exponents of interest, so one finite sample is a poor
estimate of the population it came from: the tail of the averaged kernel is
carried by the largest few weights, and a different draw moves the answer by
several percent.  The deviation of a single draw therefore does not shrink as
the sample is enlarged, which is what the runs show directly.  Every quantity
below is accordingly averaged over independent draws of the disorder, and the
spread across draws is reported next to the mean; that spread, and not the
deviation of any one draw, is what the comparison must be judged against.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import mpmath as mp  # noqa: E402

from lockkernel import kernels as K, lineshapes as L, parametric as P  # noqa: E402

mp.mp.dps = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "09_quenched_simulation.json")


def draw(N, gamma, seed, sigma=1.0, k_min=1.0):
    """Quenched detunings (Gaussian line) and weights (power law)."""
    rng = np.random.default_rng(seed)
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    delta = rng.normal(scale=sigma, size=N)
    # inverse transform for P(k) = (g-1) k_min^(g-1) k^-g on k > k_min
    k = k_min * rng.uniform(size=N) ** (-1.0 / (gamma - 1.0))
    return delta, k, fwhm


def run_overdamped(delta, k, Kc_mult, T=400.0, dt=0.01, seed=0):
    """Overdamped phase oscillators with quenched weights."""
    N = len(delta)
    kbar = k.mean()
    rng = np.random.default_rng(seed + 7)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    nsteps = int(T / dt)
    half = nsteps // 2
    acc = np.zeros(N)
    Facc = 0.0
    nacc = 0
    for n in range(nsteps):
        z = (k * np.exp(1j * theta)).sum() / (N * kbar)
        F = Kc_mult * abs(z)
        psi = np.angle(z)
        # midpoint step
        d1 = delta + k * F * np.sin(psi - theta)
        th2 = theta + 0.5 * dt * d1
        z2 = (k * np.exp(1j * th2)).sum() / (N * kbar)
        F2, psi2 = Kc_mult * abs(z2), np.angle(z2)
        theta = theta + dt * (delta + k * F2 * np.sin(psi2 - th2))
        if n >= half:
            acc += np.cos(theta - psi2)
            Facc += F2
            nacc += 1
    return acc / nacc, Facc / nacc


def run_precessing(delta, k, chi, T=400.0, dt=0.005, seed=0):
    """Unit vectors precessing in the field they generate."""
    N = len(delta)
    kbar = k.mean()
    v = np.zeros((N, 3))
    v[:, 0] = 1.0                      # fully coherent start
    nsteps = int(T / dt)
    half = nsteps // 2
    acc = np.zeros(N)
    Facc = 0.0
    nacc = 0

    def field(v):
        M = (k[:, None] * v).sum(axis=0) / (N * kbar)
        return chi * M[0], chi * M[1]

    def deriv(v):
        Fx, Fy = field(v)
        B = np.empty_like(v)
        B[:, 0] = k * Fx
        B[:, 1] = k * Fy
        B[:, 2] = delta
        return np.cross(B, v)

    for n in range(nsteps):
        k1 = deriv(v)
        k2 = deriv(v + 0.5 * dt * k1)
        k3 = deriv(v + 0.5 * dt * k2)
        k4 = deriv(v + dt * k3)
        v = v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        v /= np.linalg.norm(v, axis=1)[:, None]
        if n >= half:
            Fx, Fy = field(v)
            amp = np.hypot(Fx, Fy)
            if amp > 0:
                acc += (v[:, 0] * Fx + v[:, 1] * Fy) / amp
                Facc += amp
                nacc += 1
    return acc / max(nacc, 1), Facc / max(nacc, 1)


def compare(name, delta, k, proj, F, gamma, base, line, nbins=22,
            quiet=False):
    """Bin the measured projections by detuning and compare with the kernel.

    The kernel that enters the self consistency is defined so that
    R = int p(delta) Wt(delta/F) d delta with R the WEIGHTED order parameter,
    so the quantity to compare against Wt is the mean of k_i cos_i over the
    bin, divided by the mean weight of the whole population.  Averaging the
    projections without their weights is a different average and does not
    converge to Wt.
    """
    ker = K.heterogeneous(base, gamma)
    kbar = k.mean()
    edges = np.linspace(0.0, 6.0 * F, nbins + 1)
    a = np.abs(delta)
    rows = []
    for j in range(nbins):
        m = (a >= edges[j]) & (a < edges[j + 1])
        if m.sum() < 200:
            continue
        v = 0.5 * (edges[j] + edges[j + 1]) / F
        w = k[m] * proj[m] / kbar
        rows.append(dict(v=float(v), n=int(m.sum()),
                         measured=float(w.mean()),
                         stderr=float(w.std(ddof=1) / np.sqrt(m.sum())),
                         predicted=float(ker.W(mp.mpf(float(v))))))
    dev = [abs(r["measured"] - r["predicted"]) for r in rows]
    sig = [abs(r["measured"] - r["predicted"]) / max(r["stderr"], 1e-12)
           for r in rows]
    if not quiet:
        print(f"  {name}: {len(rows)} bins, max |measured - predicted| = "
              f"{max(dev):.4f}, mean = {np.mean(dev):.4f}, "
              f"worst bin = {max(sig):.1f} standard errors", flush=True)
    return rows, float(max(dev)), float(np.mean(dev)), float(max(sig))


def one_realisation(label, base, r, gamma, N, seed, T):
    """One draw of the quenched disorder, start to finish."""
    delta, k, fwhm = draw(N, gamma, seed)
    line = L.gaussian(fwhm)
    ker = K.heterogeneous(base, gamma)
    coupling = r * float(P.threshold(line, ker))
    if label == "overdamped":
        proj, F = run_overdamped(delta, k, coupling, T=T, seed=seed)
    else:
        proj, F = run_precessing(delta, k, coupling, T=T, seed=seed)
    rows, dmax, dmean, dsig = compare(label, delta, k, proj, F, gamma, base,
                                      line, quiet=True)
    R_sim = float((k * proj).sum() / k.sum())
    R_law = float(P.G_of_Omega(line, ker, mp.mpf(F)) * F)
    return dict(seed=seed, F=float(F), R_sim=R_sim, R_law=R_law,
                signed_dev=(R_sim - R_law) / R_law,
                kernel_max_dev=dmax, kernel_mean_dev=dmean, bins=rows)


def main():
    N = 20000
    gamma = 3.8
    seeds = [20260910 + 1000 * j for j in range(6)]
    res = {"N": N, "gamma": gamma, "seeds": seeds, "cases": []}
    print(f"{len(seeds)} independent draws of the disorder, N = {N}, "
          f"gamma = {gamma}")

    for label, base, T in (("overdamped", K.kuramoto(), 250.0),
                           ("precessing", K.conservative(), 150.0)):
        for r in (1.5, 2.5):
            runs = [one_realisation(label, base, r, gamma, N, s, T)
                    for s in seeds]
            dev = np.array([q["signed_dev"] for q in runs])
            kmax = np.array([q["kernel_max_dev"] for q in runs])
            mean = float(dev.mean())
            spread = float(dev.std(ddof=1))
            sem = spread / np.sqrt(len(dev))
            print(f"  {label}, r = {r}")
            print(f"      order parameter, relative deviation from the law: "
                  f"mean {mean:+.4f}, spread across draws {spread:.4f}, "
                  f"standard error {sem:.4f}")
            print(f"      the mean is {abs(mean) / sem:.1f} standard errors "
                  f"from zero; draws range {dev.min():+.4f} to {dev.max():+.4f}")
            print(f"      worst kernel bin, averaged over draws: "
                  f"{kmax.mean():.4f}", flush=True)
            res["cases"].append(dict(
                dynamics=label, r=r, T=T, mean_signed_dev=mean, spread=spread,
                sem=float(sem), sigmas_from_zero=float(abs(mean) / sem),
                dev_min=float(dev.min()), dev_max=float(dev.max()),
                kernel_max_dev_mean=float(kmax.mean()), runs=runs))
            json.dump(res, open(OUT, "w"), indent=1)

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
