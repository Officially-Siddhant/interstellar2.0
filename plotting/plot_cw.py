# plot_two_spacecraft_3d.py
import os, sys
import numpy as np
from matplotlib.lines import lineStyles

os.environ.pop("MPLBACKEND", None)

import matplotlib
# Choose one backend you have:
matplotlib.use("TkAgg")     # or "MacOSX" / "Qt5Agg"
import matplotlib.pyplot as plt
print("Backend:", matplotlib.get_backend())

sys.path.append("../")
from spacecraft_dynamics import make_nd_scales
import util

def load_h5py(fn):
    import h5py
    np_dict = {}
    param = {}

    with h5py.File(fn, "r") as h5file:

        # Load datasets
        for key in h5file.keys():
            dset = h5file[key]

            # If dataset is a group named "param"
            if isinstance(dset, h5py.Group) and key == "param":
                # Recursively load param values
                for pkey, pval in dset.items():
                    if pval.shape == () or pval.shape is None:
                        param[pkey] = pval[()]
                    else:
                        param[pkey] = pval[:]
                continue

            # Normal datasets
            if dset.shape == () or dset.shape is None:
                np_dict[key] = dset[()]
            else:
                np_dict[key] = dset[:]

        # File-level attributes also get added
        for k, v in h5file.attrs.items():
            param[k] = v

    return np_dict, param
# =====================================================================
#  Generic plotting helper: load *all* spacecraft trajectories in file
#  The new CW logs may contain different spacecraft labels.
# =====================================================================
def extract_spacecraft_trajectories(np_dict):
    """
    Returns a list of (name, trajectory_array) for all spacecraft entries
    detected in the h5 dictionary.
    """
    sc_entries = []
    for key, val in np_dict.items():
        if key.startswith("xs_mj_") or key.startswith("xs_"):
            # Typical shapes: (N,12) or (N,6)
            sc_entries.append((key, val))

    return sorted(sc_entries, key=lambda x: x[0])  # sort by name


# =====================================================================
#                       3D Orbit Plot
# =====================================================================
def plot_orbits_3d(fn_h5, out_png="orbits.png"):

    np_dict, param = util.load_h5py(fn_h5)

    # Get all spacecraft trajectories
    spacecraft_list = extract_spacecraft_trajectories(np_dict)

    if len(spacecraft_list) == 0:
        raise RuntimeError("No spacecraft trajectories found in the file.")

    # Debris (optional)
    debris = np_dict["debris"] if "debris" in np_dict else None

    # ---- Use MuJoCo / ND scales ----
    nd = make_nd_scales(param)   # gives L0, T0, ...
    L0 = nd["L0"]

    # =============================================================
    # Prepare ND trajectories
    # =============================================================
    sc_nd = []
    for name, xs in spacecraft_list:
        # position is first 3 columns
        r_nd = xs[:, :3] / L0
        sc_nd.append((name, r_nd))

    # Debris ND
    if debris is not None:
        deb_nd = debris[:, :3] / L0
    else:
        deb_nd = None

    # Earth radius
    R_earth_phys = 6.371e6
    R_earth_nd = R_earth_phys / L0

    # =============================================================
    # Build 3D figure
    # =============================================================
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = R_earth_nd * np.outer(np.cos(u), np.sin(v))
    ys = R_earth_nd * np.outer(np.sin(u), np.sin(v))
    zs = R_earth_nd * np.outer(np.ones_like(u), np.cos(v))

    ax.set_box_aspect([1, 1, 1])
    ax.plot_surface(xs, ys, zs, alpha=0.25, color="C0", linewidth=0, shade=True)

    # =============================================================
    # Plot each spacecraft trajectory
    # =============================================================
    for idx, (name, r_nd) in enumerate(sc_nd):
        color = f"C{(idx+1) % 10}"
        ax.plot(r_nd[:, 0], r_nd[:, 1], r_nd[:, 2], lw=1.8,
                color=color, label=name)
        # Start marker
        ax.scatter(r_nd[0, 0], r_nd[0, 1], r_nd[0, 2],
                   s=28, marker="o", color=color)

    # Plot debris trajectory
    if deb_nd is not None:
        ax.plot(deb_nd[:,0], deb_nd[:,1], deb_nd[:,2],
                label="Debris", lw=1.2, color="red", linestyle="--")
        ax.scatter(deb_nd[0,0], deb_nd[0,1], deb_nd[0,2],
                   s=40, marker="^", color="red")

    # =============================================================
    # Set up axes
    # =============================================================
    ax.set_title("Spacecraft Trajectories (ND units)")
    ax.set_xlabel("x~")
    ax.set_ylabel("y~")
    ax.set_zlabel("z~")
    ax.legend(loc="upper left")

    # =============================================================
    # Equal aspect in 3D
    # =============================================================
    stacks = [arr for (_, arr) in sc_nd]
    if deb_nd is not None:
        stacks.append(deb_nd)

    all_pts = np.vstack(stacks)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    mid = all_pts.mean(axis=0)

    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2, m + max_range / 2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved 3D orbit plot to: {out_png}")


# =====================================================================
# Driver
# =====================================================================
if __name__ == "__main__":
    # plot_orbits_3d("./five_spacecraft.h5", out_png="five_spacecrafts.png")
    plot_orbits_3d("./cw_pursuit.h5", out_png="cw_pursuit.png")