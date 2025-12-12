# LUNA-TIC Human Lunar Landing Monte Carlo Simulation 

**Brief Description**
LUNA-TIC (Lunar Uncertainty Numerical Analysis – Trajectory Integration & Calculations) is a simple 1D Monte Carlo simulator for human lunar terminal descent. It integrates the vertical trajectory of a lander near the Moon under uncertain mass, thrust, and burn conditions and then evaluates basic human-rated safety metrics such as touchdown speed and crew g-load.

---

## What this project does

This mini-project models a **two-phase** lunar descent, focusing on vertical motion close to the surface:

1. A powered braking phase where the engine delivers constant thrust to reduce the downward speed.
2. An engine-off free-fall phase from that point down to the surface.

The simulator tracks:

- Altitude above the lunar surface  
- Vertical velocity (downward positive)  
- Engine thrust during a terminal braking burn  
- Maximum g-load felt by the crew (in Earth g’s)  
- Touchdown vertical speed and a Boolean safe landing flag  

A landing is labeled safe if both:

- Touchdown vertical speed is below a specified limit  
- Maximum g-load stays below a human-rating limit

In this implementation, the limits are fixed to

- Touchdown vertical speed ≤ **3 m/s**  
- Maximum crew g-load ≤ **5 g**

These values are rough stand-ins for “soft” landings and acceptable transient crew loads based on Apollo lander studies and more recent human-rating work for vehicles like Orion. Real missions would tune these numbers using detailed landing-gear design and occupant-protection analysis, but 3 m/s and 5 g give a reasonable human-rated target for this mini-project.

---

## Numerical methods

The code uses two standard numerical techniques.

### 1. Time integration of ODEs

The 1D vertical dynamics near the surface are

- `dv/dt = g_lunar - T/m`  
- `dh/dt = -v`  

with a two-phase descent profile:

1. **Powered braking**  
   Constant thrust is applied until the downward vertical speed is reduced to a small target value `v_target`.

2. **Engine-off free fall**  
   After the burn is cut, the lander falls under the influence of lunar gravity only. The crew is effectively weightless in this phase; proper g-load is set to zero here, and only the powered phase contributes to the felt g’s.

These equations are integrated in time using a **first-order explicit Euler method** with a semi-implicit altitude update:

- `v_{n+1} = v_n + a_n * dt`  
- `h_{n+1} = h_n - v_{n+1} * dt`  

The scheme is simple to read, which is intentional for this mini-project. It has a global error of order `O(dt)`. A small time step `dt` is chosen so that the terminal descent (tens of seconds) is resolved adequately.

### 2. Monte Carlo estimation

Uncertainty is modeled by treating several inputs as random variables:

- Lander mass  
- Altitude at which the terminal burn begins  
- Vertical speed at burn start  
- Engine thrust level  

Here, the mass is intentionally random. It represents variation in payload and remaining propellant at the start of terminal descent, rather than assuming every landing happens at exactly the same vehicle mass. In the default configuration, the lander mass is modeled as a Gaussian distribution, centered at 15,000 kg with a standard deviation of 500 kg.

A Monte Carlo loop draws many independent samples from Gaussian distributions and runs the deterministic simulator once per sample. After `N` runs, the code estimates:

- The fraction of safe landings  
- The distribution of touchdown speeds  
- The distribution of maximum g-loads

### Why look at distributions, not just one number

A single nominal landing speed is insufficient for design purposes. The histograms of touchdown speeds and g-loads show

- how often the vehicle stays near the desired soft-landing regime  
- how fat the tail of bad outcomes is  
- whether small changes in mass, thrust, or timing push many trajectories over the safety limits  

In other words, the distribution of touchdown conditions is what allows you to discuss risk. The safe-landing fraction is just one summary of that distribution.

Monte Carlo converges statistically with error scaling like `1/sqrt(N)`. It is not the fastest possible uncertainty quantification method, but it is conceptually simple and well-suited to this context.

### Verification pathway

The `verification.py` module runs a **pure free-fall test** with no thrust:

- `dv/dt = g_lunar`, `dh/dt = -v`  
- Analytic solution: `h_exact(t) = h0 - 0.5 * g_lunar * t^2`

The script integrates this with the same Euler scheme for several time steps `dt` and prints the maximum altitude error for each time step. The error decreases as `dt` is reduced, which is consistent with the expected first-order behavior and provides a basic check that the integrator is implemented correctly.

---

## Getting started

### Prerequisites

- Python 3.10 or later  
- `pip` for installing dependencies  
- `git` if you want to clone the repository directly

### Installation

Clone the repository and install dependencies:

```
git clone https://github.com/gloomurai85/luna-tic-landing-sim.git
cd luna-tic-landing-sim
```

# Optional: create and activate a virtual environment here

pip install -r requirements.txt

The `requirements.txt` file lists the only two dependencies:

- `numpy` for numerical work  
- `matplotlib` for plotting histograms  

---

## Quickstart- Running a Monte Carlo experiment

From the repository root:

```
python run_monte_carlo.py --samples 1000
```

If successful, you should see output similar to:

```  
=== LUNA-TIC Monte Carlo summary ===
    Total runs            : 1000
    Safe landing fraction : 0.0xx
    Mean touchdown speed  : 53.xx
    Mean max g-load       : 0.xx

Sampling configuration:
  mass_mean     = 15000.0 kg
  h0_mean       = 1000.0 m
  v0_mean       = 25.0 m/s downward
  thrust_nominal= 72900.0 N
  seed          = 1
```
Exact numbers will vary depending on the configuration; the defaults currently yield essentially zero safe landings due to burn sequencing and setup.

Two histogram plots will also open:

- Touchdown vertical speed distribution  
- Maximum g-load distribution  

You can change the number of samples, the mean burn-start speed, the nominal thrust level, and the random seed through the `--samples`, `--v0-mean`, `--thrust-mult`, and `--seed` command-line arguments.

---

## Running the verification test

To run the free-fall convergence test:

```
python -c "from luna_lander.verification import free_fall_convergence; free_fall_convergence()"
```

This prints a small table with columns:

- Time step `dt`  
- Maximum absolute altitude error compared to the analytic solution  

This table is evidence that the numerical method behaves as expected when `dt` is refined.

---

## Repository layout

```
    luna-tic-landing-sim/          # repo root
    ├── README.md                  # overview and build/run instructions
    ├── requirements.txt           # numpy and matplotlib
    ├── run_monte_carlo.py         # command-line entry point and plotting
    └── luna_lander/
        ├── __init__.py            # package docstring and high-level description
        ├── sim.py                 # 1D vertical dynamics and Euler time integrator
        ├── monte_carlo.py         # sampling of uncertainties and batch execution
        └── verification.py        # free-fall convergence/verification test
```
---

## Assumptions and limitations

To keep the model transparent and manageable for this mini-project, several simplifying assumptions are made:

- Motion is purely vertical near the surface. Horizontal dynamics, attitude control, and terrain slope are neglected.  
- Lunar gravity is constant and uniform.  
- The terminal braking burn uses a fixed thrust magnitude and a simple cut-off rule (`v <= v_target`). No detailed guidance or optimal control is modeled.  
- Human factors are represented by two scalar criteria:
  - Maximum g-load below a fixed limit  
  - Touchdown speed below a fixed limit  
- Time integration uses a first-order Euler method. Accuracy is controlled by the step size and verified through the free-fall test.

These choices are sufficient to illustrate numerical methods, uncertainty propagation, and basic human-rating logic without turning the project into a full flight dynamics code.

---

## Possible extensions

If more time were available, the natural next steps would include:

- Adding horizontal motion and surface slope to study tipping risk and landing stability.  
- Replacing Euler with a higher-order method (for example, RK4) to reduce time-discretization error.  
- Allowing the same code to analyze Mars or other bodies by changing gravity and atmospheric models.  
- Introducing configuration files or a small front end to swap between crewed, cargo, or emergency-abort mission profiles.
