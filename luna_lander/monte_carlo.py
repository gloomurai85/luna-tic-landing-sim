"""
monte_carlo.py

Monte Carlo driver for the LUNA-TIC lunar landing simulations.

This module wraps the deterministic 1D lander model in a probabilistic
framework. Instead of running one "ideal" descent, we treat several
key quantities as uncertain and sample many possible realizations:

- Lander mass (ex. due to payload variations or propellant margins)
- Altitude at which the terminal braking burn begins
- Vertical speed at burn start
- Engine thrust level (ex. performance variation or throttling error)

For each Monte Carlo sample, we:

    1. Draw a specific set of parameters from Gaussian distributions
       defined in MonteCarloConfig.
    2. Call simulate_descent_two_phase from sim.py to propagate the
       lander from burn start to the surface.
    3. Record scalar outputs:
         - touchdown vertical speed
         - maximum g-load experienced
         - success flag (based on human safety thresholds)
         - the sampled parameters themselves (mass, h0, v0, thrust)

After N samples, we aggregate these into arrays stored in a
MonteCarloResults dataclass. This makes it easy to:

- Estimate a safe landing fraction under the assumed uncertainties.
- Examine the distributions of touchdown speeds and g-loads.
- Explore trends such as how mass or thrust variations correlate with
  hard landings.

Numerical method employed:

- The underlying numerical solver for each trajectory is the Euler-based
  ODE integrator in sim.py. This module does not solve new equations but
  calls that solver many times with different inputs.

- The statistical technique is basic Monte Carlo estimation:
    - Each draw is independent.
    - The estimated probabilities converge as O(1/sqrt(N)).
    - Increasing the number of samples reduces statistical noise but
      increases runtime.

This file demonstrates how to structure
a loop around a numerical core, manage configuration data cleanly, and
separate "physics" (sim.py) from "statistics" (this module).
"""

from dataclasses import dataclass

import numpy as np

from .sim import DescentConfig, BurnProfile, SimulationResult, simulate_descent_two_phase


@dataclass
class MonteCarloConfig:
    """Statistical assumptions and number of runs."""

    n_samples: int = 1000

    # Nominal design values
    mass_mean: float = 15000.0          # kg
    mass_sigma: float = 500.0           # kg

    h0_mean: float = 1000.0             # m, altitude at start of terminal burn
    h0_sigma: float = 50.0              # m

    v0_mean: float = 40.0               # m/s downward at burn start
    v0_sigma: float = 5.0               # m/s

    # Thrust sized so nominal case reaches ~2 m/s at the surface
    thrust_nominal: float = 1.5 * mass_mean * 1.62  # ~1.5x lunar weight
    thrust_sigma_fraction: float = 0.10             # 10% variation

    seed: int = 1                       # RNG seed


@dataclass
class MonteCarloResults:
    """Aggregated arrays from a Monte Carlo batch."""

    touchdown_vs: np.ndarray
    max_gs: np.ndarray
    successes: np.ndarray
    masses: np.ndarray
    h0s: np.ndarray
    v0s: np.ndarray
    thrusts: np.ndarray
    config: MonteCarloConfig
    descent_config: DescentConfig
    burn_profile: BurnProfile


def run_monte_carlo(
    mc_cfg: MonteCarloConfig | None = None,
    descent_cfg: DescentConfig | None = None,
    burn_profile: BurnProfile | None = None,
) -> MonteCarloResults:
    """
    Run many independent landing simulations with random parameters.

    This is a basic Monte Carlo estimator: we approximate probabilities
    by sampling (convergence ~ O(1/sqrt(N))).
    """
    if mc_cfg is None:
        mc_cfg = MonteCarloConfig()
    if descent_cfg is None:
        descent_cfg = DescentConfig()
    if burn_profile is None:
        burn_profile = BurnProfile(thrust=mc_cfg.thrust_nominal, v_target=2.0)

    rng = np.random.default_rng(mc_cfg.seed)

    n = mc_cfg.n_samples
    touchdown_vs = np.zeros(n)
    max_gs = np.zeros(n)
    successes = np.zeros(n, dtype=bool)
    masses = np.zeros(n)
    h0s = np.zeros(n)
    v0s = np.zeros(n)
    thrusts = np.zeros(n)

    for i in range(n):
        mass = rng.normal(mc_cfg.mass_mean, mc_cfg.mass_sigma)
        h0 = rng.normal(mc_cfg.h0_mean, mc_cfg.h0_sigma)
        v0 = rng.normal(mc_cfg.v0_mean, mc_cfg.v0_sigma)
        thrust = rng.normal(
            mc_cfg.thrust_nominal,
            mc_cfg.thrust_nominal * mc_cfg.thrust_sigma_fraction,
        )

        result: SimulationResult = simulate_descent_two_phase(
            mass=mass,
            h0=h0,
            v0=v0,
            burn_profile=BurnProfile(thrust=thrust, v_target=burn_profile.v_target),
            config=descent_cfg,
        )

        touchdown_vs[i] = result.touchdown_velocity
        max_gs[i] = result.max_g_load
        successes[i] = result.success
        masses[i] = mass
        h0s[i] = h0
        v0s[i] = v0
        thrusts[i] = thrust

    return MonteCarloResults(
        touchdown_vs=touchdown_vs,
        max_gs=max_gs,
        successes=successes,
        masses=masses,
        h0s=h0s,
        v0s=v0s,
        thrusts=thrusts,
        config=mc_cfg,
        descent_config=descent_cfg,
        burn_profile=burn_profile,
    )
