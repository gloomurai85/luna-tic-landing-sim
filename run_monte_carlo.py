"""
run_monte_carlo.py

Command-line entry point for the LUNA-TIC human lunar landing analysis.

This script is the front door for users. It ties together the numerical 
components in the luna_lander package and produces the key outputs.

High-level behavior:

1. Parse command-line arguments (currently just the number of samples).
2. Construct configuration objects:
      - MonteCarloConfig: number of runs and uncertainty assumptions.
      - DescentConfig: numerical time-step and human safety thresholds.
      - BurnProfile: nominal thrust level and target vertical speed.
3. Call run_monte_carlo(...) from luna_lander.monte_carlo to generate a
   batch of landing simulations under uncertainty.
4. Print summary statistics, such as:
      - total number of runs
      - estimated "safe landing" fraction
      - mean touchdown speed
      - mean maximum g-load
5. Create basic matplotlib histograms that visualize the distributions
   of touchdown speeds and max g-loads.

User's perspective:

- This is the only script that needs to run to reproduce the main
  numerical experiments in this mini-project:

      python run_monte_carlo.py --samples 1000

From a design perspective:

- Keeping the CLI logic here and the physics/statistics in the
  luna_lander package makes the structure easier to explain. It also
  means that if a different front end is ever added (such as a small GUI or
  web app), the same core simulation code can be used unmodified.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from luna_lander.sim import DescentConfig, BurnProfile
from luna_lander.monte_carlo import MonteCarloConfig, run_monte_carlo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LUNA-TIC Monte Carlo human lunar landing simulation"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="number of Monte Carlo samples (default: 500)",
    )

    parser.add_argument(
        "--v0-mean",
        type=float,
        default=25.0,
        help="mean downward speed at burn start in m/s (default: 25.0)",
    )

    parser.add_argument(
        "--thrust-mult",
        type=float,
        default=3.0,
        help=(
            "multiplicative factor on lunar weight for nominal thrust "
            "(T_nominal = thrust_mult * m_mean * g_lunar, default: 3.0)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed for Monte Carlo sampling (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Numerical settings for the descent integrator
    descent_cfg = DescentConfig(
        # keep defaults for g_lunar, g0, dt, max_time, limits
    )

    # Nominal mass used to size thrust; matches MonteCarloConfig default
    mass_mean = 15000.0

    # Monte Carlo configuration, partly driven by CLI arguments
    mc_cfg = MonteCarloConfig(
        n_samples=args.samples,
        mass_mean=mass_mean,
        v0_mean=args.v0_mean,
        # Size nominal thrust as (thrust_mult * m_mean * g_lunar)
        thrust_nominal=args.thrust_mult * mass_mean * descent_cfg.g_lunar,
        seed=args.seed,
    )

    # Constant-thrust braking profile; v_target is fixed here but could also
    # eventually be exposed as a CLI parameter or config-file option.
    burn = BurnProfile(
        thrust=mc_cfg.thrust_nominal,
        v_target=2.0,
    )

    results = run_monte_carlo(
        mc_cfg=mc_cfg,
        descent_cfg=descent_cfg,
        burn_profile=burn,
    )

    touchdown_vs = results.touchdown_vs
    max_gs = results.max_gs
    successes = results.successes

    safe_fraction = np.mean(successes.astype(float))
    mean_v = float(np.mean(touchdown_vs))
    mean_g = float(np.mean(max_gs))

    # Neatly formatted summary header
    label_width = 24  # width of the left-hand labels

    print("--- LUNA-TIC Monte Carlo summary ---")
    print(f"{'Total runs':<{label_width}} : {results.config.n_samples:7d}")
    print(f"{'Safe landing fraction':<{label_width}} : {safe_fraction:7.3f}")
    print(f"{'Mean touchdown speed':<{label_width}} : {mean_v:7.2f} m/s")
    print(f"{'Mean max g-load':<{label_width}} : {mean_g:7.2f} g")
    print()
    print("Sampling configuration:")
    print(f"  mass_mean     = {results.config.mass_mean:.1f} kg")
    print(f"  h0_mean       = {results.config.h0_mean:.1f} m")
    print(f"  v0_mean       = {results.config.v0_mean:.1f} m/s downward")
    print(f"  thrust_nominal= {results.config.thrust_nominal:.1f} N")
    print(f"  seed          = {results.config.seed:d}")
    print()

    # Histogram: touchdown vertical speed
    plt.figure()
    plt.hist(touchdown_vs, bins=40, edgecolor="black", linewidth=0.5)
    plt.xlabel("Touchdown vertical speed [m/s]")
    plt.ylabel("Count")
    plt.title("LUNA-TIC Monte Carlo: touchdown speed distribution")

    # Histogram: maximum g-load
    plt.figure()
    plt.hist(max_gs, bins=40, edgecolor="black", linewidth=0.5)
    plt.xlabel("Maximum g-load [Earth g]")
    plt.ylabel("Count")
    plt.title("LUNA-TIC Monte Carlo: maximum g-load distribution")

    plt.show()


if __name__ == "__main__":
    main()
