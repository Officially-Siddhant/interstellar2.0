# simulate_randomized_debris_comoffset.py
import os, sys, time
import numpy as np
import mujoco

# local imports
sys.path.append("../")
from spacecraft_dynamics import make_nd_scales, mujoco_nd_from_newton_x, newton_x_from_mujoco_nd
from spacecraft_dynamics import force_gravity_world_nd  # ND central gravity
import util

# Matplotlib (save-only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Helpers ----------

R_EARTH = 6.371e6  # meters

def d2r(x): return np.deg2rad(x)

def kepler_to_rv(mu, a, e, inc, raan, argp, nu):
    """Perifocal -> ECI (standard). Angles in radians."""
    p = a * (1 - e**2)
    r_pf = np.array([p*np.cos(nu)/(1+e*np.cos(nu)),
                     p*np.sin(nu)/(1+e*np.cos(nu)),
                     0.0])
    v_pf = np.array([-np.sqrt(mu/p)*np.sin(nu),
                      np.sqrt(mu/p)*(e+np.cos(nu)),
                      0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    R3O = np.array([[ cO,-sO,0],[ sO, cO,0],[0,0,1]])
    R1i = np.array([[1,0,0],[0,ci,-si],[0,si,ci]])
    R3w = np.array([[ cw,-sw,0],[ sw, cw,0],[0,0,1]])
    Q = R3O @ R1i @ R3w
    return Q @ r_pf, Q @ v_pf

def simulate_one_debris(xml_path, param, x0_r, x0_v, omega0, T_phys,
                        com_offset_std_nd=3e-4, rng=None):
    """
    Single debris run -> returns (ts, xs_phys, nd).
      - Inertia fixed to [1,1,1] (ND).
      - COM offset is random ND vector (std = com_offset_std_nd per axis).
      - Initial angular velocity 'omega0' (rad/s) is APPLIED (randomized outside).
      - Gravity applied at world COM (data.xipos[bid]).
    """
    if rng is None:
        rng = np.random.default_rng()

    # For ND scales we need param with x0
    param_local = dict(param)
    param_local["x0"] = np.concatenate([x0_r, [0,0,0], x0_v, [0,0,0]])
    nd = make_nd_scales(param_local)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # RK4, zero uniform gravity (we apply central gravity manually)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.gravity[:] = 0.0

    mj_dt_nd = param["mj_dt"] / nd["T0"]
    model.opt.timestep = mj_dt_nd

    # IDs / addresses
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "debris")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "debris6dof")
    qpos_adr = model.jnt_qposadr[jid]
    qvel_adr = model.jnt_dofadr[jid]

    # Fixed ND mass & inertia
    model.body_mass[bid]    = 1.0
    model.body_inertia[bid] = np.array([1.0, 1.0, 1.0])

    # --- Random COM offset in ND (small) ---
    com_offset_nd = rng.normal(0.0, com_offset_std_nd, size=3)
    model.body_ipos[bid] = com_offset_nd

    # Initial state in ND  (omega0 is applied!)
    mujoco.mj_resetData(model, data)
    x0 = np.concatenate([x0_r, [0,0,0], x0_v, omega0])  # <-- include random omega0
    qpos_nd, qvel_nd = mujoco_nd_from_newton_x(x0, nd)
    data.qpos[qpos_adr:qpos_adr+7] = qpos_nd
    data.qvel[qvel_adr:qvel_adr+6] = qvel_nd
    mujoco.mj_forward(model, data)

    # Simulate
    num_nd_steps = int(T_phys / nd["T0"] / mj_dt_nd)
    xs_mj = [np.concatenate([data.qpos[qpos_adr:qpos_adr+7],
                             data.qvel[qvel_adr:qvel_adr+6]])]

    for _ in range(num_nd_steps):
        # Use world COM position for central gravity
        r_nd = data.xipos[bid].copy()
        data.xfrc_applied[bid, :]  = 0.0
        data.xfrc_applied[bid, :3] = force_gravity_world_nd(r_nd)

        mujoco.mj_step(model, data)
        xs_mj.append(np.concatenate([data.qpos[qpos_adr:qpos_adr+7],
                                     data.qvel[qvel_adr:qvel_adr+6]]))

    ts = np.arange(0, num_nd_steps + 1) * mj_dt_nd * nd["T0"]
    xs_phys = np.array([newton_x_from_mujoco_nd(x[:7], x[7:], nd) for x in xs_mj])
    return ts, xs_phys, nd

def main():
    # ---- Physical params (minimal) ----
    param = {
        "mass"       : 1.0,              # reference mass for ND scaling
        "mass_earth" : 5.97219e24,
        "r_earth"    : np.array([0.0, 0.0, 0.0]),
        "G"          : 6.67430e-11,
        "mj_dt"      : 0.05,             # physical sec per MuJoCo step
    }
    mu = param["G"] * param["mass_earth"]

    # ---- XML with a single 'debris' free body (debris & debris6dof) ----
    xml_path = os.path.join(os.path.dirname(__file__), "debris_nd.xml")

    # ---- Debris batch settings ----
    M = 15                       # number of debris runs to overlay
    alt_range = (500e3, 750e3)   # meters
    inc_range = (0.0, 15.0)      # degrees
    T_phys    = 2 * 60 * 60      # ~2 hours
    com_std_nd = 3e-4            # ND COM offset std (tune 1e-4..5e-4)
    omega_std = 1.0e-3           # rad/s std for random initial angular velocity

    all_trajs = []
    nd_scales = None
    rng = np.random.default_rng(42)

    for _ in range(M):
        # Random near-circular elements (e=0)
        alt = rng.uniform(*alt_range)
        a   = R_EARTH + alt
        e   = 0.0
        inc = d2r(rng.uniform(*inc_range))
        raan = d2r(rng.uniform(0, 360))
        argp = d2r(rng.uniform(0, 360))
        nu   = d2r(rng.uniform(0, 360))

        r0, v0 = kepler_to_rv(mu, a, e, inc, raan, argp, nu)

        # --- RANDOM angular velocity (explicit) ---
        omega0 = rng.normal(0.0, omega_std, size=3)  # rad/s, per-axis

        ts, xs, nd = simulate_one_debris(
            xml_path, param, r0, v0, omega0, T_phys,
            com_offset_std_nd=com_std_nd, rng=rng
        )
        all_trajs.append(xs)
        nd_scales = nd# Random near-circular elements (e=0)

    # ---- Save trajectories
    out_h5 = "./random_debris_wobble_comoffset.h5"
    to_save = {"ts": ts}
    for i, xs in enumerate(all_trajs, start=1):
        to_save[f"xs_debris_{i}"] = xs
    util.save_h5py(out_h5, to_save, param=param)
    print(f"Saved: {out_h5}")

    # ---- Plot 3D over ND Earth
    L0 = nd_scales["L0"]
    R_earth_nd = R_EARTH / L0

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1,1,1])

    # Earth sphere (ND)
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs_E = R_earth_nd * np.outer(np.cos(u), np.sin(v))
    ys_E = R_earth_nd * np.outer(np.sin(u), np.sin(v))
    zs_E = R_earth_nd * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs_E, ys_E, zs_E, alpha=0.25, color="C0", linewidth=0, shade=True)

    # Debris trajectories (ND), thin dashed colored lines
    for i, xs in enumerate(all_trajs, start=1):
        r_nd = xs[:, :3] / L0
        ax.plot(r_nd[:,0], r_nd[:,1], r_nd[:,2],
                lw=0.9, linestyle="--", label=f"Debris {i}")
        ax.scatter(r_nd[0,0], r_nd[0,1], r_nd[0,2], s=18, marker="o")

    ax.set_title("Randomized Debris (COM offset + random ω0) — ND units")
    ax.set_xlabel("x~"); ax.set_ylabel("y~"); ax.set_zlabel("z~")
    ax.legend(loc="upper left", fontsize=8)

    # Equal-ish limits
    all_pts = np.vstack([xs[:, :3] for xs in all_trajs]) / L0
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    mid = all_pts.mean(axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(m - max_range/2, m + max_range/2)

    out_png = "./random_debris_wobble_comoffset_3d.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D plot: {out_png}")

if __name__ == "__main__":
    main()