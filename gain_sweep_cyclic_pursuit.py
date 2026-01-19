# gain_sweep_cyclic_pursuit.py
#
# Sweep gains for cyclic-pursuit velocity controller around an orbiting debris.
# Uses 5_spacecraft_nd.xml and runs ONE debris orbit per gain combo, headless.
#
# Costs:
#   J_spacing(t) = mean_i (d_ij - d_des)^2  over neighbor pairs
#   J_center(t)  = || r_center(t) - r_debris(t) ||^2
#
# Integrated (discrete) over one orbit:
#   J_spacing_int = sum_k J_spacing(t_k) * dt_phys
#   J_center_int  = sum_k J_center(t_k)  * dt_phys

import os, sys
import itertools
import numpy as np
import mujoco

# local imports
sys.path.append("../")
from spacecraft_dynamics import (
    make_nd_scales,
    mujoco_nd_from_newton_x,
    newton_x_from_mujoco_nd,
    inertia_phys_to_nd,
)
from test_controllers import cyclic_pursuit_vel_controller_centered
import util

os.environ["MPLBACKEND"] = "Agg"


def set_velocity(r, vz_des, mu):
    """
    Given physical position r and desired v_z, solve for v_x, v_y such that:
      1. r · v = 0 (circular orbit, perpendicular condition)
      2. |v| = sqrt(mu / |r|) (orbital speed)
    """
    r = np.array(r, dtype=float)
    rnorm = np.linalg.norm(r)
    v_c   = np.sqrt(mu / rnorm)
    vz    = float(vz_des)

    if abs(r[0]) < 1e-8 and abs(r[1]) < 1e-8:
        vx, vy = v_c, 0.0
    else:
        tangent = np.array([-r[1], r[0], 0.0])
        tangent /= np.linalg.norm(tangent)
        v_inplane_mag = np.sqrt(max(v_c**2 - vz**2, 0.0))
        vx, vy = v_inplane_mag * tangent[:2]

    return np.array([vx, vy, vz])


def run_single_sim(kd, kc, ka, kv, cdamp, R_des_nd):
    """
    Run ONE orbit of the debris for a given gain combo and return:
        (J_spacing_int, J_center_int, ts_phys, spacing_traj, center_traj)

    spacing_traj[k] = J_spacing(t_k)
    center_traj[k]  = J_center(t_k)
    """

    # ----- Physical params -----
    param = {
        "mass":       1600.0,
        "inertia":    np.diag([7429.33, 7429.33, 4608.0]),
        "mass_earth": 5.97219e24,
        "r_earth":    np.array([0.0, 0.0, 0.0]),
        "G":          6.67430e-11,
        "dt":         0.05,   # not used directly
        "mj_dt":      0.05,   # physical seconds per MuJoCo step
        "viewer":     False,
    }
    mu = param["G"] * param["mass_earth"]

    # Common circular orbit radius (physical)
    R = 6.371e6 + 500e3   # 500 km LEO
    angles_deg = [0, 10, 20, 30, 40]  # 5 sats, 10° apart
    angles = np.deg2rad(angles_deg)

    # Orbital mean motion and period (physical)
    n = np.sqrt(mu / R**3)
    T_orbit = 2.0 * np.pi * np.sqrt(R**3 / mu)  # one orbit

    # Satellite initial states (physical)
    rs, vs = [], []
    for th in angles:
        r = np.array([R * np.cos(th), R * np.sin(th), 0.0])
        v = set_velocity(r, vz_des=0.0, mu=mu)  # same in-plane orbit speed
        rs.append(r)
        vs.append(v)

    x0_list = []
    for r, v in zip(rs, vs):
        x0 = np.array([
            r[0], r[1], r[2],   # position
            0.0, 0.0, 0.0,      # Euler
            v[0], v[1], v[2],   # linear vel
            0.0, 0.0, 0.0       # omega
        ])
        x0_list.append(x0)

    # Debris: same circular orbit, different phase (0 deg)
    r_deb = np.array([R, 0.0, 0.0])
    v_deb = set_velocity(r_deb, vz_des=0.0, mu=mu)

    deb_x0 = np.array([
        r_deb[0], r_deb[1], r_deb[2],
        0.0, 0.0, 0.0,
        v_deb[0], v_deb[1], v_deb[2],
        0.0, 0.0, 0.0
    ])

    # ----- ND scales -----
    param["x0"] = x0_list[0]
    nd = make_nd_scales(param)
    L0 = nd["L0"]
    T0 = nd["T0"]

    # ----- MuJoCo model -----
    xml_path = os.path.join(os.path.dirname(__file__), "5_spacecraft_nd.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Time step (ND)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.gravity[:] = 0.0  # satellites: no gravity; debris: manual if desired
    mj_dt_nd = param["mj_dt"] / T0
    model.opt.timestep = mj_dt_nd

    # IDs
    body_names  = ["spacecraft", "spacecraft2", "spacecraft3", "spacecraft4", "spacecraft5"]
    joint_names = ["base6dof", "base6dof2", "base6dof3", "base6dof4", "base6dof5"]

    bids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n_) for n_ in body_names]
    jids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n_) for n_ in joint_names]
    deb_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "debris")
    deb_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "debris6dof")

    # ND inertia/mass
    I_nd = inertia_phys_to_nd(param["inertia"], nd)
    for bid in bids:
        model.body_mass[bid] = 1.0
        model.body_inertia[bid] = np.array([I_nd[0, 0], I_nd[1, 1], I_nd[2, 2]])

    mujoco.mj_resetData(model, data)

    # Place satellites (in ND)
    for x0, jid in zip(x0_list, jids):
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        qpos_nd, qvel_nd = mujoco_nd_from_newton_x(x0, nd)
        data.qpos[qpos_adr:qpos_adr + 7] = qpos_nd
        data.qvel[qvel_adr:qvel_adr + 6] = qvel_nd

    # Place debris (in ND)
    qpos_adr_deb = model.jnt_qposadr[deb_jid]
    qvel_adr_deb = model.jnt_dofadr[deb_jid]
    qpos_deb_nd, qvel_deb_nd = mujoco_nd_from_newton_x(deb_x0, nd)
    data.qpos[qpos_adr_deb:qpos_adr_deb + 7] = qpos_deb_nd
    data.qvel[qvel_adr_deb:qvel_adr_deb + 6] = qvel_deb_nd

    mujoco.mj_forward(model, data)

    # ----- Time horizon: 1 orbit -----
    dt_phys = mj_dt_nd * T0
    num_nd_steps = int(T_orbit / dt_phys)

    # ----- Cost logging -----
    N = len(bids)
    spacing_cost_traj = np.zeros(num_nd_steps)
    center_cost_traj  = np.zeros(num_nd_steps)

    # Desired inter-agent distance (chord) in ND
    d_des_phys = 2.0 * (R_des_nd * L0) * np.sin(np.pi / N)   # R_des_nd * L0 = physical radius
    d_des_nd   = d_des_phys / L0                             # back to ND → effectively 2*R_des_nd*sin(pi/N)

    # ----- MAIN LOOP (NO VIEWER) -----
    for k in range(num_nd_steps):

        # 1) Read satellite states in ND
        positions_nd  = np.zeros((N, 3))
        velocities_nd = np.zeros((N, 3))
        for i, jid in enumerate(jids):
            qpos_adr = model.jnt_qposadr[jid]
            qvel_adr = model.jnt_dofadr[jid]
            positions_nd[i]  = data.qpos[qpos_adr:qpos_adr + 3]
            velocities_nd[i] = data.qvel[qvel_adr:qvel_adr + 3]

        # 2) Read debris state (ND)
        debris_qpos_adr = model.jnt_qposadr[deb_jid]
        debris_qvel_adr = model.jnt_dofadr[deb_jid]
        r_debris_nd = data.qpos[debris_qpos_adr:debris_qpos_adr + 3]
        v_debris_nd = data.qvel[debris_qvel_adr:debris_qvel_adr + 3]

        # (Optional) let debris move under central gravity only; satellites ignore gravity
        # If you want that back, uncomment:
        # from spacecraft_dynamics import force_gravity_world_nd
        # Fg_deb_nd = force_gravity_world_nd(r_debris_nd)
        # data.xfrc_applied[deb_bid, :] = 0.0
        # data.xfrc_applied[deb_bid, :3] = Fg_deb_nd

        # 3) Cyclic pursuit velocity-based control (centered on debris)
        a_cyc_nd = cyclic_pursuit_vel_controller_centered(
            positions_nd=positions_nd,
            velocities_nd=velocities_nd,
            r_center_nd=r_debris_nd,
            v_center_nd=v_debris_nd,
            kd=kd,
            kc=kc,
            ka=ka,
            kv=kv,
            cdamp=cdamp,
            R_des_nd=R_des_nd,
        )  # shape: (N,3)

        # 4) Apply control forces to each spacecraft
        for i, bid in enumerate(bids):
            data.xfrc_applied[bid, :] = 0.0
            # mass is 1.0 in ND, but keep it explicit:
            m_nd = model.body_mass[bid]
            data.xfrc_applied[bid, :3] = m_nd * a_cyc_nd[i]

        # 5) Step simulation
        mujoco.mj_step(model, data)

        # 6) Compute costs at time step k
        #    a) Inter-agent spacing cost (neighbor chord length)
        d_err_sq = []
        for i in range(N):
            j = (i + 1) % N
            dx = positions_nd[j] - positions_nd[i]
            d_ij = np.linalg.norm(dx)
            e_ij = d_ij - d_des_nd
            d_err_sq.append(e_ij**2)
        spacing_cost_traj[k] = np.mean(d_err_sq)

        #    b) Center-tracking cost (formation center vs debris)
        r_center_nd = positions_nd.mean(axis=0)
        e_center = np.linalg.norm(r_center_nd - r_debris_nd)
        center_cost_traj[k] = e_center**2

    # Integrated costs over one orbit
    J_spacing_int = np.sum(spacing_cost_traj) * dt_phys
    J_center_int  = np.sum(center_cost_traj)  * dt_phys

    # Time vector (physical)
    ts_phys = np.arange(num_nd_steps) * dt_phys

    return J_spacing_int, J_center_int, ts_phys, spacing_cost_traj, center_cost_traj


def main():
    # ----- Gain grids to sweep -----
    kd_list    = [0.5, 1.0, 2.0]
    kc_list    = [0.2, 0.5]
    ka_list    = [0.0, 0.2]       # bearing adaptation
    kv_list    = [0.5, 1.0, 2.0]
    cdamp_list = [0.5, 1.0]

    # Desired ring radius in ND (in units of L0, but we don't need L0 here
    # because it's converted consistently inside run_single_sim)
    R_des_nd = 1.0   # try ~10% of the orbital radius in ND units

    results = []

    for kd, kc, ka, kv, cdamp in itertools.product(
        kd_list, kc_list, ka_list, kv_list, cdamp_list
    ):
        print(f"\n=== Running combo: kd={kd}, kc={kc}, ka={ka}, kv={kv}, cdamp={cdamp} ===")

        J_spacing, J_center, ts, spacing_traj, center_traj = run_single_sim(
            kd, kc, ka, kv, cdamp, R_des_nd
        )

        print(f"  -> J_spacing_int = {J_spacing:.4e},  J_center_int = {J_center:.4e}")

        results.append({
            "kd": kd, "kc": kc, "ka": ka, "kv": kv, "cdamp": cdamp,
            "J_spacing": J_spacing,
            "J_center": J_center,
        })

        # Optional: save time-series for this combo, or only for the best one later.

    # Sort by combined cost (you can change the weighting)
    results_sorted = sorted(
        results,
        key=lambda d: (d["J_center"] + d["J_spacing"])
    )

    print("\n=== Top 5 gain combinations (by J_center + J_spacing) ===")
    for r in results_sorted[:5]:
        print(
            f"kd={r['kd']}, kc={r['kc']}, ka={r['ka']}, kv={r['kv']}, cdamp={r['cdamp']} | "
            f"J_spacing={r['J_spacing']:.3e}, J_center={r['J_center']:.3e}"
        )


if __name__ == "__main__":
    main()