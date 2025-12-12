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


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of LUNA-TIC human lunar landing."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of Monte Carlo landing simulations.",
    )
    args = parser.parse_args()

    mc_cfg = MonteCarloConfig(n_samples=args.samples)
    descent_cfg = DescentConfig(dt=0.02, touchdown_velocity_limit=3.0, max_g_limit=5.0)
    burn_profile = BurnProfile(thrust=mc_cfg.thrust_nominal, v_target=2.0)

    results = run_monte_carlo(mc_cfg, descent_cfg, burn_profile)

    success_fraction = float(np.mean(results.successes))
    mean_v_touch = float(np.mean(results.touchdown_vs))
    mean_max_g = float(np.mean(results.max_gs))

    print("--- LUNA-TIC Monte Carlo summary ---")
    print(f"Total runs            : {mc_cfg.n_samples}")
    print(f"Safe landing fraction : {success_fraction:.3f}")
    print(f"Mean touchdown speed  : {mean_v_touch:.2f} m/s")
    print(f"Mean max g-load       : {mean_max_g:.2f} g")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(results.touchdown_vs, bins=40, edgecolor="black")
    axes[0].set_xlabel("Touchdown vertical speed [m/s]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Touchdown speed distribution")

    axes[1].hist(results.max_gs, bins=40, edgecolor="black")
    axes[1].set_xlabel("Maximum g-load [g]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Max g-load distribution")

    fig.suptitle("Luna TIC terminal landing Monte Carlo")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
