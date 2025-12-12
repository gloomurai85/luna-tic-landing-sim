"""
verification.py

Simple verification checks for the Luna TIC lander integrator.

This module is not a full unit test suite, but it provides a concrete
pathway to verification by comparing the numerical method in sim.py
against a known analytic solution in a simplified case.

Test problem:

- Pure vertical free fall near the Moon, with:
      dv/dt = g_lunar
      dh/dt = -v
  and initial conditions h(0) = h0, v(0) = 0.

- This has a closed-form solution:
      h_exact(t) = h0 - 0.5 * g_lunar * t^2

Numerical experiment:

- We re-implement the same Euler scheme used in sim.py for this free
  fall, but in a small standalone function (free_fall_numeric). For a
  set of time steps dt, we:

    1. Integrate from t = 0 to t = t_final.
    2. Compute h_exact(t) at the same discrete times.
    3. Measure the maximum absolute difference |h_num - h_exact|.

- The outer function free_fall_convergence prints a small table of
  (dt, max error). For a first-order Euler method, we expect the error
  to shrink roughly linearly as dt is reduced.

- Even though the full lander problem has no simple analytic solution,
  this free-fall test exercises the same integration scheme under
  controlled conditions. Seeing errors decrease with dt gives evidence
  that the implementation is at least consistent with the theory.

In a larger project, these ideas could be extended into proper unit
tests, regression tests, and more sophisticated verification cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FreeFallTestConfig:
    h0: float = 1000.0      # m
    v0: float = 0.0         # m/s downward
    g_lunar: float = 1.62   # m/s^2
    t_final: float = 5.0    # s
    dts: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05)


def free_fall_numeric(h0: float, v0: float, g: float, dt: float, t_final: float):
    """
    Integrate pure free fall with same Euler scheme used in sim.py.

        dv/dt = g
        dh/dt = -v
    """
    steps = int(t_final / dt)
    h = h0
    v = v0
    t = 0.0

    ts = []
    hs = []

    for _ in range(steps):
        ts.append(t)
        hs.append(h)

        a = g
        v = v + a * dt
        h = h - v * dt
        t += dt

    return np.array(ts), np.array(hs)


def free_fall_convergence(test_cfg: FreeFallTestConfig | None = None):
    """
    Print max altitude error for several time steps.

    Analytic solution:
        h_exact(t) = h0 - 0.5 * g * t^2

    For a first-order Euler scheme we expect the error to decrease roughly
    linearly with dt.
    """
    if test_cfg is None:
        test_cfg = FreeFallTestConfig()

    print("Free-fall convergence test")
    print(f"h0 = {test_cfg.h0} m, v0 = {test_cfg.v0} m/s, g = {test_cfg.g_lunar} m/s^2")
    print("dt [s]   max |h_num - h_exact| [m]")

    for dt in test_cfg.dts:
        t, h_num = free_fall_numeric(
            test_cfg.h0, test_cfg.v0, test_cfg.g_lunar, dt, test_cfg.t_final
        )
        h_exact = test_cfg.h0 - 0.5 * test_cfg.g_lunar * t**2
        max_err = float(np.max(np.abs(h_exact - h_num)))
        print(f"{dt:6.3f}   {max_err:10.5f}")
