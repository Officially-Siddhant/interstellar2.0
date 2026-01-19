# validate_dynamics_triangle.py
"""
Triangular validation of spacecraft dynamics.

WHAT THIS TESTS
---------------
1) Cross-model equivalence:
   Compare trajectories from:
     (a) Newton–Euler ODE  (dxdt_newton)
     (b) Euler–Lagrange ODE (dxdt_lagrange)
     (c) MuJoCo (nondimensional XML)        -> all gravity-only, no control.
   We compute state error norms between each pair over 2 orbits.

2) Physical fidelity (invariants):
   Check conservation over time of:
     - Specific mechanical energy: e = 0.5*||v||^2 - mu/||r||
     - Specific angular momentum:  h = ||r x v||
   We expect minimal drift.

3) Period accuracy:
   Measure orbital period from the trajectories and compare to Kepler:
     T_kepler = 2*pi*sqrt(a^3/mu), with a≈||r0|| for near-circular.

Outputs:
 - figures: states grid, invariants, and error/period summary
 - metrics printout
 - HDF5 file with trajectories
"""

import os
import sys
import time
import numpy as np
import h5py
import mujoco

# Make local modules importable (same pattern as your test script)
sys.path.append("../")
from spacecraft_dynamics import *
from spacecraft_controllers import empty_controller
import util

# ---- Matplotlib headless config ----
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def simulate_three_models(param, T):
    """Run Newton–Euler, Euler–Lagrange, and MuJoCo (ND) for gravity-only."""
    controller = empty_controller

    # ---------- Baselines: Newton & Lagrange ----------
    xs_lagrange = [newton_to_lagrange_x(param["x0"])]
    xs_newton   = [param["x0"]]
    us_lagrange, us_newton = [], []
    num_steps = int(T / param["dt"])
    for step in range(num_steps):
        if step % 2000 == 0:
            print(f"[NE/EL] step {step}/{num_steps}")
        uL = controller(lagrange_to_newton_x(xs_lagrange[-1]))
        uN = controller(xs_newton[-1])
        us_lagrange.append(uL)
        us_newton.append(uN)

        xL_next = xs_lagrange[-1] + param["dt"] * dxdt_lagrange(xs_lagrange[-1], uL, param)
        xN_next = xs_newton[-1]   + param["dt"] * dxdt_newton  (xs_newton[-1],   uN, param)

        # wrap Euler angles to (-pi, pi]
        xL_next[3:6] = (xL_next[3:6] + np.pi) % (2*np.pi) - np.pi
        xN_next[3:6] = (xN_next[3:6] + np.pi) % (2*np.pi) - np.pi

        xs_lagrange.append(xL_next)
        xs_newton.append(xN_next)

    xs_lagrange = np.array(xs_lagrange)
    xs_newton   = np.array(xs_newton)
    ts_newton   = np.arange(0, num_steps+1) * param["dt"]

    # ---------- MuJoCo (nondimensional) ----------
    nd = make_nd_scales(param)
    xml_path = os.path.join(os.path.dirname(__file__), "spacecraft_nd.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Integrator/time step in ND units
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    mj_dt_nd = param["mj_dt"] / nd["T0"]
    model.opt.timestep = mj_dt_nd

    # Inertia/mass in ND (assumes single body named "spacecraft")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    I_nd = inertia_phys_to_nd(param["inertia"], nd)
    model.body_mass[bid]     = 1.0
    model.body_inertia[bid]  = np.array([I_nd[0,0], I_nd[1,1], I_nd[2,2]])

    # Initial state in ND
    mujoco.mj_resetData(model, data)
    qpos_nd, qvel_nd = mujoco_nd_from_newton_x(param["x0"], nd)
    data.qpos[:] = qpos_nd
    data.qvel[:] = qvel_nd
    mujoco.mj_forward(model, data)

    xs_mj = [np.concatenate([data.qpos, data.qvel])]
    num_nd_steps = int(T / nd["T0"] / mj_dt_nd)
    print(f"[MJ] ND steps: {num_nd_steps}")


    # Mujoco Simulator
    viewer = mujoco.viewer.launch_passive(model, data)
    with viewer:
        for step in range(num_nd_steps):
            if step % 2000 == 0:
                print(f"[MJ] step {step}/{num_nd_steps}")
            # once, after model/data creation
            model.opt.gravity[:] = 0.0  # ensure no uniform gravity

            # in the sim loop, each step BEFORE mj_step:
            data.xfrc_applied[bid, :] = 0.0
            f_g_nd = force_gravity_world_nd(data.qpos[:3])  # = -r_hat / R^2 in ND units (actually -r/R^3)
            data.xfrc_applied[bid, :3] = f_g_nd  # world-frame force at COM
            # optional: add control forces/torques here too (same world-frame convention)
            mujoco.mj_step(model, data)  # advance simulation
            viewer.sync()  # update viewer
            time.sleep(model.opt.timestep)  # optional: real-time pacing

            xs_mj.append(np.concatenate([data.qpos, data.qvel]))


    # after simulation ends → build corresponding time vector
    ts_mj = np.arange(0, num_nd_steps + 1) * mj_dt_nd * nd["T0"]
    xs_mj = np.array([
        newton_x_from_mujoco_nd(x[0:7], x[7:], nd)
        for x in xs_mj
    ])
    return (xs_newton, xs_lagrange, ts_newton), (xs_mj, ts_mj)


def specific_energy_and_h(x, mu):
    """Per-unit-mass energy and ang. momentum magnitude from Newton state x."""
    r = x[:, 0:3]
    v = x[:, 6:9]
    rnorm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v*v, axis=1)
    e = 0.5 * v2 - mu / rnorm
    h = np.linalg.norm(np.cross(r, v), axis=1)
    return e, h


def period_from_vis_viva(xs, mu):
    r = xs[:, :3]
    v = xs[:, 6:9]
    rnorm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v*v, axis=1)

    # a via vis-viva; guard against division by ~0
    denom = (2*mu - rnorm * v2)
    mask = denom > 1e-12  # bound & well-posed samples
    a = np.empty_like(rnorm)
    a[:] = np.nan
    a[mask] = (mu * rnorm[mask]) / denom[mask]

    # keep bound-orbit samples (a > 0)
    a_valid = a[(a > 0) & np.isfinite(a)]
    if len(a_valid) == 0:
        return np.nan

    a_med = np.median(a_valid)
    return 2*np.pi * np.sqrt(a_med**3 / mu)


def kepler_period(r0_norm, mu):
    return 2.0 * np.pi * np.sqrt((r0_norm**3) / mu)


def plot_states_grid(xs_newton, xs_lagrange, ts_newton, xs_mj, ts_mj, out_path):
    state_labels = ["r_x","r_y","r_z","phi","theta","psi",
                    "v_x","v_y","v_z","omega_x","omega_y","omega_z"]
    nrows, ncols = 3, 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for ii_row in range(nrows):
        for ii_col in range(ncols):
            k = ii_row*ncols + ii_col
            a = ax[ii_row, ii_col]
            a.plot(ts_mj,     xs_mj[:,k],        linestyle=(0, (2, 6)), label="MuJoCo")
            a.plot(ts_newton, xs_lagrange[:,k],  linestyle=(2, (2, 6)), label="Euler-Lagrange")
            a.plot(ts_newton, xs_newton[:,k],    linestyle=(4, (2, 6)), label="Newton-Euler")
            a.set_title(state_labels[k])
            if ii_row == 0 and ii_col == 0:
                a.legend()
            if ii_row == nrows - 1:
                a.set_xlabel("time [s]")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_invariants(ts_newton, e_NE, h_NE, e_EL, h_EL, ts_mj, e_MJ, h_MJ, out_path):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(ts_newton, e_NE, label="NE")
    ax1.plot(ts_newton, e_EL, label="EL")
    ax1.plot(ts_mj,     e_MJ, label="MuJoCo")
    ax1.set_ylabel("specific energy")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(ts_newton, h_NE, label="NE")
    ax2.plot(ts_newton, h_EL, label="EL")
    ax2.plot(ts_mj,     h_MJ, label="MuJoCo")
    ax2.set_ylabel("||h||")
    ax2.set_xlabel("time [s]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_errors_and_period(ts_newton, xs_NE, xs_EL, ts_mj, xs_MJ,
                           T_KEP, T_NE, T_EL, T_MJ, out_path):
    # Interpolate MJ onto NE time grid for error norm (simple nearest)
    xs_MJ_interp = np.empty_like(xs_NE)
    mj_idx = 0
    for i, t in enumerate(ts_newton):
        while mj_idx+1 < len(ts_mj) and abs(ts_mj[mj_idx+1]-t) < abs(ts_mj[mj_idx]-t):
            mj_idx += 1
        xs_MJ_interp[i] = xs_MJ[mj_idx]

    err_NE_MJ = np.linalg.norm(xs_NE - xs_MJ_interp, axis=1)
    err_EL_MJ = np.linalg.norm(xs_EL - xs_MJ_interp, axis=1)
    err_NE_EL = np.linalg.norm(xs_NE - xs_EL, axis=1)

    fig = plt.figure(figsize=(12,6))

    # Left: error norms (log y)
    ax1 = fig.add_subplot(1,2,1)
    ax1.semilogy(ts_newton, err_NE_MJ, label="||NE - MJ||")
    ax1.semilogy(ts_newton, err_EL_MJ, label="||EL - MJ||")
    ax1.semilogy(ts_newton, err_NE_EL, label="||NE - EL||")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("state error norm")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    # Right: period bar chart vs Kepler
    ax2 = fig.add_subplot(1,2,2)
    names = ["Kepler", "NE", "EL", "MJ"]
    vals  = [T_KEP, T_NE, T_EL, T_MJ]
    ax2.bar(names, vals)
    ax2.set_ylabel("period [s]")
    ax2.set_title("Measured orbital period vs Kepler")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print quick metrics
    print("\n=== Cross-model error metrics (RMS / Max) ===")
    for name, e in [("NE-MJ", err_NE_MJ), ("EL-MJ", err_EL_MJ), ("NE-EL", err_NE_EL)]:
        print(f"{name}: RMS={np.sqrt(np.mean(e**2)):.4e}, Max={np.max(e):.4e}")


def main():
    # ----- Parameters (headless; viewer off) -----
    param = {
        "mass" : 1600,
        "inertia" : np.diag([7429.33, 7429.33, 4608.0]),
        "mass_earth" : 5.97219e24,
        "r_earth" : np.array([0,0,0]),
        "G" : 6.67430e-11,
        "x0" : np.array([6.371e6+500.0, 0.0, 0.0,    # r
                         0.0, 0.0, 0.0,              # euler
                         0.0, 7670.0, 1000.0,        # velocity
                         0.001, 0.002, 0.003]),      # omega (body)
        "mujoco_viewer" : False,
        "realtime_rate" : 1000.0,
        "dt" : 0.05,
        "mj_dt" : 0.05,
    }
    T = 128 * 60  # two orbits in your setup

    # ----- Simulate three models -----
    (xs_NE, xs_EL, ts_NE), (xs_MJ, ts_MJ) = simulate_three_models(param, T)


    # ----- Invariants -----
    mu = param["G"] * param["mass_earth"]
    e_NE, h_NE = specific_energy_and_h(xs_NE, mu)
    e_EL, h_EL = specific_energy_and_h(xs_EL, mu)
    e_MJ, h_MJ = specific_energy_and_h(xs_MJ, mu)

    # ----- Period checks -----
    r0 = np.linalg.norm(param["x0"][:3])            # radius taken from the
    T_KEP = kepler_period(r0, mu)
    T_NE  = period_from_vis_viva(xs_NE, mu)
    T_EL  = period_from_vis_viva(xs_EL, mu)
    T_MJ  = period_from_vis_viva(xs_MJ, mu)

    print("\n=== Periods [s] ===")
    print(f"Kepler: {T_KEP:.3f} | NE: {T_NE:.3f} | EL: {T_EL:.3f} | MJ: {T_MJ:.3f}")

    # ----- Save HDF5 (optional) -----
    util.save_h5py(
        "./triangle_validation.h5",
        {
            "xs_newton": xs_NE,
            "xs_lagrange": xs_EL,
            "xs_mujoco": xs_MJ,
            "ts_newton": ts_NE,
            "ts_mujoco": ts_MJ,
            "energy_NE": e_NE,
            "energy_EL": e_EL,
            "energy_MJ": e_MJ,
            "h_NE": h_NE,
            "h_EL": h_EL,
            "h_MJ": h_MJ,
        },
        param=param
    )

    # ----- Figures (3) -----
    plot_states_grid(xs_NE, xs_EL, ts_NE, xs_MJ, ts_MJ, "fig1_states_grid.png")
    plot_invariants(ts_NE, e_NE, h_NE, e_EL, h_EL, ts_MJ, e_MJ, h_MJ, "fig2_invariants.png")
    plot_errors_and_period(ts_NE, xs_NE, xs_EL, ts_MJ, xs_MJ,
                           T_KEP, T_NE, T_EL, T_MJ, "fig3_errors_period.png")

    print("\nSaved: triangle_validation.h5, fig1_states_grid.png, fig2_invariants.png, fig3_errors_period.png")


if __name__ == "__main__":
    main()