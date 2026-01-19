# Interstellar 2.0

This repository takes from concepts discussed in [this](https://github.com/class-ROB-GY-7863/class_ROB-GY_7863) repository, and attempts to integrate an autonomy solution to a system of multiple nano-satellites pursuing debris in Low Earth Orbit. Control algorithms were validated in MATLAB which are discussed in a [precursor](https://github.com/Officially-Siddhant/interstellar) repository.

- Find the paper here: [Swarm Based Encirclement of Orbital Debris in LEO](https://linktr.ee/siddhant_baroth)
---

# System Requirements

- Python **3.8+**
- Anaconda or Miniconda
- macOS / Linux (Windows supported but not recommended for MuJoCo)

---

# Getting Started

### 1. Create a Conda Environment
```bash
conda create -n space-robotics python=3.8 -y
conda activate space-robotics
```
### 2. Running Simulations 
Several experiments helped gauge an understanding of the complexities of keplerian motion in low earth orbit. It is easy to hardcode 2-body motion, but in this project, we are attempting 3-body dynamics with the Earth as the source of gravitation. 

Therefore, satellites must ensure dynamic convergence to a formation, the target debris, and cyclic pursuit while being subjected to gravitational pulls.

To validate conservation of energy, run:
```bash
mjpython validate_dynamics_triangle.py
```

To simulate cyclic pursuit with a single debris object in Earth's orbit:
```bash
mjpython cw_simulation.py --debris-orbit
```

To simulate 5 nanosats in cyclic pursuit around coasting debris:
```bash
mjpython cyclic_pursuit_mujoco2.py
```

Dynamics and Control files are independent. In the simulations, we rely on Mujoco generated dynamics to be accurate.

### What Next?
MuJoCo is extremely lightweight. It gets the job done and to some degree of satisfaction, is sufficient to establish a POC. However, there is a lot more that can be done with the problem statement alone, and that is the setup for upcoming research. 

Things such as working with J2 perturbations, oblateness, multiple coasting debris and using model predictive control instead of classical control techniques are currently in development.

Using differential drag as a means of mobility in LEO is also an interesting track - it has been tested on CubeSats!
