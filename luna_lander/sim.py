"""
sim.py

Core 1D vertical dynamics and time-integration for a human-rated lunar lander.

This module answers the single focused question:
    "Given a mass, an initial altitude and vertical speed, and a simple
     braking burn strategy, what touchdown speed and g-load will a lunar
     lander experience on its way to the surface?"

The model is intentionally minimal but physically interpretable as:

- The motion is purely vertical, representing the final descent segment
  directly above the landing site. Horizontal motion, attitude dynamics,
  and slope effects are neglected so that we can focus on terminal
  velocity and crew g-load without overcomplicating the problem.

- The coordinate system is:
      h(t)  = altitude above lunar surface [m], positive upward
      v(t)  = vertical velocity [m/s], positive downward

  With this convention:
      dh/dt = -v
      dv/dt = g_lunar - T/m

  where g_lunar is constant lunar gravity and T is the engine thrust
  (modeled as a scalar magnitude acting upward).

- The descent is split into two phases:
    1) Powered braking with constant thrust, used to reduce the downward
       velocity from an initial value to a small target value v_target.
    2) Engine-off free fall from that point down to the surface.

  This simple guidance profile is not meant to be optimal, but it is
  representative of a last-ditch terminal braking strategy and is
  straightforward to simulate and analyze.

Numerical method employed:

- The equations are advanced in time with a first-order explicit Euler
  method on velocity and a semi-implicit update on altitude:

      v_{n+1} = v_n + a_n * dt
      h_{n+1} = h_n - v_{n+1} * dt

  This scheme is easy to implement and has global error O(dt) and remains stable
  for the gentle deceleration considered here, provided dt is chosen small enough
  to resolve the time scale of the braking burn.

Human factors:

- At each time step in the powered phase the instantaneous
  proper acceleration is computed (the acceleration felt by the crew) and convert
  it to Earth g's. We track the maximum g-load over the trajectory and
  compare it to a human-rating limit.

- At touchdown, we use the final vertical velocity as a proxy for
  impact severity. Combining the g-limit and a maximum allowable
  touchdown speed lets us define a simple Boolean success flag for
  each trajectory.

The main outputs are collected in a SimulationResult dataclass so that
other modules (Monte Carlo, plotting, report scripts) can consume the
results without needing to know any implementation details here.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DescentConfig:
    """Configuration for time integration and human constraints."""

    g_lunar: float = 1.62        # m/s^2, lunar gravity
    g0: float = 9.81             # m/s^2, Earth g for converting to g-load
    dt: float = 0.02             # time step [s]
    max_time: float = 600.0      # safety cap on simulation time [s]
    touchdown_velocity_limit: float = 3.0   # |v_touch| limit for safe landing [m/s]
    max_g_limit: float = 5.0     # max allowed g-load felt by crew


@dataclass
class BurnProfile:
    """Simple model for terminal braking burn."""

    thrust: float                # constant engine thrust [N]
    v_target: float = 0.0        # cut engine when v <= v_target (still downward)


@dataclass
class SimulationResult:
    """Stores history and scalar metrics for one descent simulation."""

    times: np.ndarray
    altitudes: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    g_loads: np.ndarray
    touchdown_velocity: float
    max_g_load: float
    success: bool


def simulate_descent_two_phase(
    mass: float,
    h0: float,
    v0: float,
    burn_profile: BurnProfile,
    config: DescentConfig | None = None,
) -> SimulationResult:
    """
    Simulate a 1D lunar descent with two phases:
      1) Powered braking with constant thrust until v <= v_target
      2) Engine-off free fall to the surface

    Numerical method:
      - First-order explicit Euler on velocity
      - Semi-implicit update on altitude:
            v_{n+1} = v_n + a_n * dt
            h_{n+1} = h_n - v_{n+1} * dt
    """
    if config is None:
        config = DescentConfig()

    dt = config.dt

    t = 0.0
    h = h0
    v = v0

    times: List[float] = []
    hs: List[float] = []
    vs: List[float] = []
    accels: List[float] = []
    g_loads: List[float] = []

    max_g = 0.0

    # Phase 1: powered braking
    while t < config.max_time and h > 0.0 and v > burn_profile.v_target:
        # dv/dt = g_lunar - T/m    (downward positive)
        a = config.g_lunar - burn_profile.thrust / mass

        # Proper g-load felt by crew in Earth g's
        g_load = abs(a) / config.g0
        max_g = max(max_g, g_load)

        times.append(t)
        hs.append(h)
        vs.append(v)
        accels.append(a)
        g_loads.append(g_load)

        # Euler update. velocity then altitude
        v = v + a * dt
        h = h - v * dt
        t += dt

    # Phase 2: free fall (engine off) -> crew are effectively weightless
    while t < config.max_time and h > 0.0:
        a = config.g_lunar
        g_load = 0.0  # proper acceleration ~0 g in free fall

        times.append(t)
        hs.append(h)
        vs.append(v)
        accels.append(a)
        g_loads.append(g_load)

        v = v + a * dt
        h = h - v * dt
        t += dt

    touchdown_velocity = v
    success = (
        abs(touchdown_velocity) <= config.touchdown_velocity_limit
        and max_g <= config.max_g_limit
    )

    return SimulationResult(
        times=np.array(times),
        altitudes=np.array(hs),
        velocities=np.array(vs),
        accelerations=np.array(accels),
        g_loads=np.array(g_loads),
        touchdown_velocity=touchdown_velocity,
        max_g_load=max_g,
        success=success,
    )
