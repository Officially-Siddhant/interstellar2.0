# plot_two_spacecraft_3d.py
import os, sys
import numpy as np
from matplotlib.lines import lineStyles

os.environ.pop("MPLBACKEND", None)

import matplotlib
# Choose one you have:
matplotlib.use("TkAgg")     # or "MacOSX" / "Qt5Agg"
import matplotlib.pyplot as plt
print("Backend:", matplotlib.get_backend())

sys.path.append("../")
from spacecraft_dynamics import make_nd_scales
import util

def plot_orbits_3d(fn_h5, out_png="five_spacecrafts.png"):

    np_dict, param = util.load_h5py(fn_h5)

    xs1 = np_dict["xs_mj_1"]   # (N,12): [r, euler, v, omega]
    xs2 = np_dict["xs_mj_2"]
    xs3 = np_dict["xs_mj_3"]
    xs4 = np_dict["xs_mj_4"]
    xs5 = np_dict["xs_mj_5"]
    debris = np_dict["debris"] if "debris" in np_dict.keys() else None


    # ---- Use MuJoCo scales (ND)
    nd = make_nd_scales(param)   # L0, T0, ...
    L0 = nd["L0"]

    # Positions in ND (so plot matches MuJoCo scale)
    r1_nd = xs1[:, :3] / L0
    r2_nd = xs2[:, :3] / L0
    r3_nd = xs3[:, :3] / L0
    r4_nd = xs4[:, :3] / L0
    r5_nd = xs5[:, :3] / L0
    deb_nd = debris[:, :3] / L0

    # Earth sphere radius in ND
    R_earth_phys = 6.371e6
    R_earth_nd = R_earth_phys / L0

    # 3D figure
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)

    earth_scale = 0.5

    xs = earth_scale * R_earth_nd * np.outer(np.cos(u), np.sin(v))
    ys = earth_scale * R_earth_nd * np.outer(np.sin(u), np.sin(v))
    zs = earth_scale * R_earth_nd * np.outer(np.ones_like(u), np.cos(v))

    ax.set_box_aspect([1, 1, 1])
    ax.plot_surface(xs, ys, zs, alpha=0.25, color="C0", linewidth=0, shade=True)

    # Trajectories
    ax.plot(r1_nd[:,0], r1_nd[:,1], r1_nd[:,2], label="Spacecraft 1", lw=1.8, color="C1")
    ax.plot(r2_nd[:,0], r2_nd[:,1], r2_nd[:,2], label="Spacecraft 2", lw=1.8, color="C2")
    ax.plot(r3_nd[:,0], r3_nd[:,1], r3_nd[:,2], label="Spacecraft 3", lw=1.8, color="C3")
    ax.plot(r4_nd[:,0], r4_nd[:,1], r4_nd[:,2], label="Spacecraft 4", lw=1.8, color="C4")
    ax.plot(r5_nd[:,0], r5_nd[:,1], r5_nd[:,2], label="Spacecraft 5", lw=1.8, color="C5")

    if debris is not None:
        ax.plot(deb_nd[:,0], deb_nd[:,1], deb_nd[:,2], label="Debris", lw=1.0, color="red", linestyle="--")
        ax.scatter(deb_nd[0,0], deb_nd[0,1], deb_nd[0,2], s=25, marker="^", color="red")

    # Start markers
    ax.scatter(r1_nd[0, 0], r1_nd[0, 1], r1_nd[0, 2], s=30, marker="o", color="C1")
    ax.scatter(r2_nd[0, 0], r2_nd[0, 1], r2_nd[0, 2], s=30, marker="o", color="C2")
    ax.scatter(r3_nd[0, 0], r3_nd[0, 1], r3_nd[0, 2], s=30, marker="o", color="C3")
    ax.scatter(r4_nd[0, 0], r4_nd[0, 1], r4_nd[0, 2], s=30, marker="o", color="C4")
    ax.scatter(r5_nd[0, 0], r5_nd[0, 1], r5_nd[0, 2], s=30, marker="o", color="C5")

    ax.set_title("Orbits (ND units)")
    ax.set_xlabel("x~")
    ax.set_ylabel("y~")
    ax.set_zlabel("z~")
    ax.legend(loc="upper left")

    # Equal aspect for 3D
    stacks = [r1_nd, r2_nd, r3_nd, r4_nd, r5_nd] + ([deb_nd] if debris is not None else []) # add debris later
    all_pts = np.vstack(stacks)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    mid = all_pts.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range/2, m + max_range/2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved 3D orbit plot to: {out_png}")

if __name__ == "__main__":
    plot_orbits_3d("./five_spacecraft.h5", out_png="five_spacecrafts.png")
    # plot_orbits_3d("./cw_pursuit.h5", out_png="cw_pursuit.png")