# plot_cw_states.py
import os, sys
import numpy as np

os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.append("../")
import util
from spacecraft_dynamics import make_nd_scales

def load_time_vector(np_dict, param, xs_key="xs_mj_1"):
    """
    Try to recover time vector:
      - Prefer 'ts_mj' if present
      - Else reconstruct from param["mj_dt"] and trajectory length
    """
    if "ts_mj" in np_dict:
        return np_dict["ts_mj"]

    # Fall back: use mj_dt and number of samples
    xs1 = np_dict[xs_key]
    T = xs1.shape[0]

    # If ND scaling available, mj_dt is physical
    mj_dt = param.get("mj_dt", 0.05)
    t = np.arange(T) * mj_dt
    return t

def plot_velocities(np_dict, param):
    # Extract states
    xs1 = np_dict["xs_mj_1"]  # (T, 12) [r, euler, v, omega]
    xs2 = np_dict["xs_mj_2"]
    xs3 = np_dict["xs_mj_3"]
    xs4 = np_dict["xs_mj_4"]
    xs5 = np_dict["xs_mj_5"]

    xs_list = [xs1, xs2, xs3, xs4, xs5]
    N_agents = 5

    # Time vector
    t = load_time_vector(np_dict, param, xs_key="xs_mj_1")

    # Velocities are columns 6:9 if you used newton_x_from_mujoco_nd
    vxs = [xs[:, 6] for xs in xs_list]
    vys = [xs[:, 7] for xs in xs_list]
    vzs = [xs[:, 8] for xs in xs_list]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = [f"C{i}" for i in range(N_agents)]

    # vx
    ax = axs[0]
    for i in range(N_agents):
        ax.plot(t, vxs[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("v_x [m/s]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Translational Velocities vs Time")
    ax.legend(loc="best", fontsize=8)

    # vy
    ax = axs[1]
    for i in range(N_agents):
        ax.plot(t, vys[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("v_y [m/s]")
    ax.grid(True, alpha=0.3)

    # vz
    ax = axs[2]
    for i in range(N_agents):
        ax.plot(t, vzs[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("v_z [m/s]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("cw_velocities.png", dpi=300, bbox_inches="tight")
    print("Saved velocity plot to cw_velocities.png")


def plot_controls(np_dict, param):
    """
    Assumes you saved control forces as:
      u_mj_1, ..., u_mj_5  each of shape (T, 3) = [Fx, Fy, Fz]
    """
    u1 = np_dict["u_mj_1"]  # (T, 3)
    u2 = np_dict["u_mj_2"]
    u3 = np_dict["u_mj_3"]
    u4 = np_dict["u_mj_4"]
    u5 = np_dict["u_mj_5"]

    u_list = [u1, u2, u3, u4, u5]
    N_agents = 5

    # Time vector – same length as u_*
    t = load_time_vector(np_dict, param, xs_key="xs_mj_1")

    # Components
    uxs = [u[:, 0] for u in u_list]
    uys = [u[:, 1] for u in u_list]
    uzs = [u[:, 2] for u in u_list]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = [f"C{i}" for i in range(N_agents)]

    # Fx
    ax = axs[0]
    for i in range(N_agents):
        ax.plot(t, uxs[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("F_x [N]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Control Forces vs Time")
    ax.legend(loc="best", fontsize=8)

    # Fy
    ax = axs[1]
    for i in range(N_agents):
        ax.plot(t, uys[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("F_y [N]")
    ax.grid(True, alpha=0.3)

    # Fz
    ax = axs[2]
    for i in range(N_agents):
        ax.plot(t, uzs[i], label=f"SC {i+1}", color=colors[i], linewidth=1.5)
    ax.set_ylabel("F_z [N]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("cw_controls.png", dpi=300, bbox_inches="tight")
    print("Saved control plot to cw_controls.png")


def main():
    fn_h5 = "./five_spacecraft.h5"
    np_dict, param = util.load_h5py(fn_h5)

    plot_velocities(np_dict, param)
    plot_controls(np_dict, param)

    plt.show()


if __name__ == "__main__":
    main()