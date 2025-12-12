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


def free_fall_convergence() -> None:

    """
    Free-fall analytic comparison and diagnostic error-vs-dt plot.

    This function integrates a simple 1D free-fall trajectory with the
    same explicit Euler scheme used in the main lander model and compares
    it to the analytic solution h_exact(t) = h0 - 0.5 * g * t^2.

    It prints a small convergence table and also generates a log-log plot
    of max altitude error vs. time step dt. The roughly linear trend in
    log-log space is consistent with the expected first-order accuracy of
    Euler's method and serves as the primary verification/diagnostic plot
    for this mini-project.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    h0 = 1000.0
    v0 = 0.0
    g = 1.62

    # Time steps to test (coarse -> fine)
    dt_values = [0.400, 0.200, 0.100, 0.050]

    print("--- Free-fall convergence test ---")
    print(f"h0 = {h0:.1f} m, v0 = {v0:.1f} m/s, g = {g:.2f} m/s^2")
    print("dt [s]   max |h_num - h_exact| [m]")

    errors = []

    for dt in dt_values:
        # Explicit Euler on the same equations used in the main code:
        # dh/dt = -v, dv/dt = g (downward positive)
        t = [0.0]
        h = [h0]
        v = [v0]

        while h[-1] > 0.0:
            a = g                     # free-fall acceleration
            v_new = v[-1] + a * dt
            h_new = h[-1] - v_new * dt

            t.append(t[-1] + dt)
            v.append(v_new)
            h.append(h_new)

        t_arr = np.array(t)
        h_arr = np.array(h)

        # Analytic solution for comparison
        h_exact = h0 - 0.5 * g * t_arr**2
        max_err = float(np.max(np.abs(h_arr - h_exact)))
        errors.append(max_err)

        print(f"{dt:6.3f}      {max_err:8.5f}")

    # Diagnostic plot: error vs. time step
    plt.figure()
    plt.loglog(dt_values, errors, "o-")
    plt.xlabel("time step dt [s]")
    plt.ylabel("max altitude error [m]")
    plt.title("Free-fall convergence of Euler integrator")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()