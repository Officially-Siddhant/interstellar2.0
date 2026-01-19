# ------------------------------------------------------
# We test 5 spacecraft with an orbiting/static/coasting debris.
# Debris movement is specified at runtime using args:
#   --debris-coast, --debris-orbit, default: static
# Input:  uses 5_spacecraft_nd.xml
# Output: ./five_spacecraft.h5
# ------------------------------------------------------

import os, sys, time
import numpy as np
import mujoco
sys.path.append("../")
from spacecraft_dynamics import *
from test_controllers import *
import util

os.environ["MPLBACKEND"] = "Agg"

# Debris mode from CLI
if "--debris-coast" in sys.argv:
    debris_mode = "coast"
elif "--debris-orbit" in sys.argv:
    debris_mode = "orbit"
else:
    debris_mode = "static"   # default
print(f"[INFO] Debris mode: {debris_mode}")


def set_velocity(r, vz_des, mu):
    """
    Given position r and desired v_z, solve for v_x, v_y such that:
      1. r · v = 0 (circular orbit, perpendicular condition)
      2. |v| = sqrt(mu / |r|) (orbital speed)
    """
    r = np.array(r, dtype=float)
    rnorm = np.linalg.norm(r)
    v_c   = np.sqrt(mu / rnorm)
    vz    = float(vz_des)

    if abs(r[0]) < 1e-8 and abs(r[1]) < 1e-8:
        # if r is near z-axis, orbit in x–y plane
        vx, vy = v_c, 0.0
    else:
        # tangent direction in xy-plane
        tangent = np.array([-r[1], r[0], 0.0])
        tangent /= np.linalg.norm(tangent)
        v_inplane_mag = np.sqrt(max(v_c**2 - vz**2, 0.0))
        vx, vy = v_inplane_mag * tangent[:2]

    return np.array([vx, vy, vz])


def period_from_vis_viva(xs, mu):
    r = xs[:, :3]
    v = xs[:, 6:9]
    rnorm = np.linalg.norm(r, axis=1)
    v2    = np.sum(v * v, axis=1)
    denom = (2 * mu - rnorm * v2)
    mask  = denom > 1e-12
    a     = np.full_like(rnorm, np.nan, dtype=float)
    a[mask] = (mu * rnorm[mask]) / denom[mask]
    a_valid = a[(a > 0) & np.isfinite(a)]
    if len(a_valid) == 0:
        return np.nan
    a_med = np.median(a_valid)
    return 2 * np.pi * np.sqrt(a_med**3 / mu)


def inertial_to_hill(r, v, r_ref, v_ref):
    # Convert inertial (r, v) into LVLH/CW frame relative to reference (debris).
    # Output: x, xdot in Hill frame, with X radial-out, Y along-track, Z cross-track.
    
    # relative vectors
    dr = r - r_ref
    dv = v - v_ref

    # build Hill frame basis
    ez = np.cross(r_ref, v_ref)
    ez = ez / np.linalg.norm(ez)
    er = r_ref / np.linalg.norm(r_ref)
    ey = np.cross(ez, er)

    # rotation from inertial → hill
    H = np.vstack([er, ey, ez])  # 3×3
    x = H @ dr
    xdot = H @ dv

    return x, xdot, H


def hill_to_inertial(u_hill, H):
    # Convert CW-frame control acceleration to inertial acceleration.
    return H.T @ u_hill



def cw_dynamics(x_hill_nd: np.ndarray, xd_hill_nd: np.ndarray, u_hill_nd: np.ndarray, n_nd: float) -> np.ndarray:
    """
    CW dynamics in Hill frame:

        xdd = f_CW(x, xd) + u

    x_hill_nd  : (N,3) positions [x,y,z] in Hill frame (ND)
    xd_hill_nd : (N,3) velocities [xd,yd,zd] in Hill frame (ND)
    u_hill_nd  : (N,3) control accelerations in Hill frame (ND)
    n_nd       : scalar mean motion in ND units

    Returns
    -------
    xdd_hill_nd : (N,3) accelerations in Hill frame (ND)
    """
    N = x_hill_nd.shape[0]
    xdd_hill_nd = np.zeros_like(x_hill_nd)
    for i in range(N):
        f_i = hill_drift(x_hill_nd[i], xd_hill_nd[i], n_nd)  # from spacecraft_dynamics
        xdd_hill_nd[i] = f_i + u_hill_nd[i]

    return xdd_hill_nd


def cyclic_pursuit_cw(x_hill_nd: np.ndarray, xd_hill_nd: np.ndarray,n_nd: float,kc: float,kd: float,z0: float,phi_z: float,alpha: float,rot_axis: str = "z") -> np.ndarray:
    
    # Cyclic pursuit control law in the CW / Hill frame.
    x_hill_nd  = np.asarray(x_hill_nd)
    xd_hill_nd = np.asarray(xd_hill_nd)

    N = x_hill_nd.shape[0]
    assert x_hill_nd.shape == xd_hill_nd.shape == (N, 3)

    # Gains taken from Paper
    k_g = n_nd / (2.0 * np.sin(np.pi / float(N)))

    # Axis-selectable Hill-frame rotation R(α)
    c = np.cos(alpha)
    s = np.sin(alpha)

    if rot_axis == "z":
        R_alpha = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    elif rot_axis == "x":
        R_alpha = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  c, -s],
            [0.0,  s,  c]
        ])

    elif rot_axis == "y":
        R_alpha = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c]
        ])

    else:
        raise ValueError("rot_axis must be 'x', 'y', or 'z'")


    # Hill shaping matrices
    T = np.array([
        [0.5,              0.0,              0.0],
        [0.0,              1.0,              0.0],
        [z0*np.cos(phi_z), z0*np.sin(phi_z), 1.0],
    ])

    T_inv = np.linalg.inv(T)
    A = T @ R_alpha @ T_inv
    u_hill_nd = np.zeros_like(x_hill_nd)


    # Hill-frame cyclic pursuit
    for i in range(N):
        j = (i + 1) % N

        xi  = x_hill_nd[i]
        vi  = xd_hill_nd[i]
        xj  = x_hill_nd[j]
        vj  = xd_hill_nd[j]

        dx  = xj - xi
        dv  = vj - vi

        # CW drift term
        f_i = hill_drift(xi, vi, n_nd)  # stays in HILL frame

        pos_term = A @ dx
        vel_term = A @ dv

        radial_PD = -kc * kd * xi
        damping   = -(kc + kd / k_g) * vi

        u_hill_nd[i] = -f_i + k_g * (kd * pos_term + vel_term + radial_PD + damping)

    return u_hill_nd


def main():
    #  Physical params
    param = {
        "mass":       1600.0,
        "inertia":    np.diag([7429.33, 7429.33, 4608.0]),
        "mass_earth": 5.97219e24,
        "r_earth":    np.array([0.0, 0.0, 0.0]),
        "G":          6.67430e-11,
        "dt":         0.05,   # not used here except for symmetry
        "mj_dt":      0.05,   # physical seconds per MuJoCo step
        "viewer":     True,
    }
    mu = param["G"] * param["mass_earth"]

    # Common circular orbit radius
    R = 6.371e6 + 500e3   # 500 km LEO
    angles_deg = [0, 2, 4, 6, 8]  # 5 sats, 10° apart
    angles = np.deg2rad(angles_deg)
    n = np.sqrt(mu / R ** 3)

    # Satellite initial states
    rs, vs = [], []
    for th in angles:
        r = np.array([R * np.cos(th), R * np.sin(th), 0.0])
        v = set_velocity(r, vz_des=0.0, mu=mu)  # slight out-of-plane
        rs.append(r)
        vs.append(v)

    x0_list = []
    for r, v in zip(rs, vs):
        x0 = np.array([
            r[0], r[1], r[2],   # position
            0.0, 0.0, 0.0,      # euler
            v[0], v[1], v[2],   # linear vel
            0.0, 0.0, 0.0       # omega
        ])
        x0_list.append(x0)

    # Debris setup (in physical frame)
    r_deb = np.array([R * np.cos(0.0), R * np.sin(0.0), 0.0])

    if debris_mode == "static":
        v_deb = np.array([0.0, 0.0, 0.0])
    elif debris_mode == "coast":
        # straight line in inertial, no gravity
        v_deb = set_velocity(r_deb, vz_des=1000.0, mu=mu)
    elif debris_mode == "orbit":
        # circular orbit under central gravity
        v_deb = set_velocity(r_deb, vz_des=0.0, mu=mu)
    else:
        v_deb = np.array([0.0, 0.0, 0.0])  # fallback

    deb_x0 = np.array([
        r_deb[0], r_deb[1], r_deb[2],
        0.0, 0.0, 0.0,         # Euler
        v_deb[0], v_deb[1], v_deb[2],
        0.0, 0.0, 0.0          # Omega
    ])

    # ND scales -----
    param["x0"] = x0_list[0]
    nd = make_nd_scales(param)
    # mean motion in ND units (n_phys * T0; for our choice should be roughly 1)
    n_nd = n * nd["T0"]

    # MuJoCo model -----
    xml_path = os.path.join(os.path.dirname(__file__), "5_spacecraft_nd.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Integrator and timestep (ND) -----
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.gravity[:] = 0.0  # we apply central gravity manually (only to debris)
    mj_dt_nd = param["mj_dt"] / nd["T0"]
    realtime_factor = 0.5
    model.opt.timestep = mj_dt_nd

    # IDs -----
    body_names  = ["spacecraft", "spacecraft2", "spacecraft3", "spacecraft4", "spacecraft5"]
    joint_names = ["base6dof", "base6dof2", "base6dof3", "base6dof4", "base6dof5"]

    bids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  for name in body_names]
    jids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    deb_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "debris")
    deb_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "debris6dof")

    # Inertia and mass for sats (ND) -----
    I_nd = inertia_phys_to_nd(param["inertia"], nd)
    for bid in bids:
        model.body_mass[bid] = 1.0
        model.body_inertia[bid] = np.array([I_nd[0, 0], I_nd[1, 1], I_nd[2, 2]])

    mujoco.mj_resetData(model, data)

    # Place satellites (convert physical to ND MuJoCo state) ----
    for x0, jid in zip(x0_list, jids):
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        qpos_nd, qvel_nd = mujoco_nd_from_newton_x(x0, nd)
        data.qpos[qpos_adr:qpos_adr + 7] = qpos_nd
        data.qvel[qvel_adr:qvel_adr + 6] = qvel_nd

    # Place debris ----
    qpos_adr_deb = model.jnt_qposadr[deb_jid]
    qvel_adr_deb = model.jnt_dofadr[deb_jid]
    qpos_deb_nd, qvel_deb_nd = mujoco_nd_from_newton_x(deb_x0, nd)
    data.qpos[qpos_adr_deb:qpos_adr_deb + 7] = qpos_deb_nd
    data.qvel[qvel_adr_deb:qvel_adr_deb + 6] = qvel_deb_nd

    mujoco.mj_forward(model, data)

    # Sim config -----
    T_phys       = 128 * 60.0  # two orbits-ish
    num_nd_steps = int(T_phys / nd["T0"] / mj_dt_nd)

    xs_mj  = [[] for _ in range(5)]
    deb_mj = []

    # Logs for control forces (ND)
    u_mj   = [[] for _ in range(5)]   # each is a list of (3,) control forces applied

    # Integral error per satellite (ND) — currently unused with CW-only control
    e_r_int_nd = np.zeros((len(bids), 3))

    # initial logging
    for i, jid in enumerate(jids):
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        xs_mj[i].append(np.concatenate([data.qpos[qpos_adr:qpos_adr + 7],data.qvel[qvel_adr:qvel_adr + 6],]))
        u_mj[i].append(np.zeros(3))

    deb_mj.append(np.concatenate([data.qpos[qpos_adr_deb:qpos_adr_deb + 7],data.qvel[qvel_adr_deb:qvel_adr_deb + 6],]))

    # ----- MuJoCo Viewer -----
    if param["viewer"]:
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        class _Dummy:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def sync(self): pass
        viewer = _Dummy()

    # Controller Params -----
    ring_offsets_nd = [None] * 5 
    R_nd = R / nd["L0"]

    with viewer:
        for step in range(num_nd_steps):
            if step % 2000 == 0:
                print(f"[MJ] step {step}/{num_nd_steps}")

            # Read satellite states (ND)
            positions_nd  = np.zeros((5, 3))
            velocities_nd = np.zeros((5, 3))
            for i, jid in enumerate(jids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]
                positions_nd[i]  = data.qpos[qpos_adr:qpos_adr + 3]
                velocities_nd[i] = data.qvel[qvel_adr:qvel_adr + 3]

            # Read debris state (ND)
            debris_qpos_adr = model.jnt_qposadr[deb_jid]
            debris_qvel_adr = model.jnt_dofadr[deb_jid]
            r_debris_nd = data.qpos[debris_qpos_adr:debris_qpos_adr + 3]
            v_debris_nd = data.qvel[debris_qvel_adr:debris_qvel_adr + 3]

            # Debris dynamics per mode
            data.xfrc_applied[deb_bid, :] = 0.0
            if debris_mode == "orbit":
                # debris feels central gravity
                Fg_deb_nd = force_gravity_world_nd(r_debris_nd)
                data.xfrc_applied[deb_bid, :3] = Fg_deb_nd
            elif debris_mode == "coast":
                # no forces, straight-line in inertial
                pass
            elif debris_mode == "static":
                # keep debris frozen where it started
                data.qvel[debris_qvel_adr:debris_qvel_adr + 6] = 0.0

            # build Hill states for each satellite, relative to debris
            N_sats = positions_nd.shape[0]
            x_hill_nd = np.zeros((N_sats, 3))
            xd_hill_nd = np.zeros((N_sats, 3))
            H_list = []

            for i in range(N_sats):
                r_i = positions_nd[i]
                v_i = velocities_nd[i]

                # Inertial to Hill (relative to debris)
                x_i, xd_i, H_i = inertial_to_hill(
                    r_i, v_i,
                    r_ref=r_debris_nd,
                    v_ref=v_debris_nd
                )

                x_hill_nd[i]  = x_i
                xd_hill_nd[i] = xd_i
                H_list.append(H_i)

            # Cyclic Pursuit Control in Hill frame
            u_hill_nd = cyclic_pursuit_cw(
                x_hill_nd=x_hill_nd,
                xd_hill_nd=xd_hill_nd,
                n_nd=n_nd,
                kc=0.5,
                kd=1.0,
                z0=0.5,
                phi_z=np.pi/4.0,
                alpha=2.0 * np.pi / float(N_sats), #72 for 5 sats
                rot_axis="z"
            )

            # Dynamics in Hill frame: xdd = f_CW + u
            xdd_hill_nd = cw_dynamics(
                x_hill_nd, xd_hill_nd,
                u_hill_nd=u_hill_nd,
                n_nd=n_nd
            )

            # Map Hill accelerations back to inertial/world frame
            a_cyc_nd = np.zeros((N_sats, 3))
            for i in range(N_sats):
                H_i = H_list[i]
                a_cyc_nd[i] = hill_to_inertial(xdd_hill_nd[i], H_i)

            # Apply forces
            for i, bid in enumerate(bids):
                m_nd = model.body_mass[bid]
                F_i = m_nd * a_cyc_nd[i]

                data.xfrc_applied[bid, :] = 0.0
                data.xfrc_applied[bid, :3] = F_i
                u_mj[i].append(F_i.copy())

            # Step sim
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep / realtime_factor)

            # Log Satellites
            for i, jid in enumerate(jids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]
                xs_mj[i].append(np.concatenate([data.qpos[qpos_adr:qpos_adr + 7],data.qvel[qvel_adr:qvel_adr + 6],]))

            # Log Debris
            deb_mj.append(np.concatenate([data.qpos[qpos_adr_deb:qpos_adr_deb + 7],data.qvel[qvel_adr_deb:qvel_adr_deb + 6],]))

    # ----- Logging -----
    ts_mj = np.arange(0, num_nd_steps + 1) * mj_dt_nd * nd["T0"]

    xs_MJ_all = [np.array([newton_x_from_mujoco_nd(x[:7], x[7:], nd) for x in traj]) for traj in xs_mj]
    xs_debris_phys = np.array([newton_x_from_mujoco_nd(x[:7], x[7:], nd) for x in deb_mj])

    u_mj_np = [np.array(u_list) for u_list in u_mj]

    # Periods
    T1 = period_from_vis_viva(xs_MJ_all[0], mu)
    T2 = period_from_vis_viva(xs_MJ_all[1], mu)
    T3 = period_from_vis_viva(xs_MJ_all[2], mu)
    T4 = period_from_vis_viva(xs_MJ_all[3], mu)
    T5 = period_from_vis_viva(xs_MJ_all[4], mu)

    T1_kepler = 2 * np.pi * np.sqrt((np.linalg.norm(x0_list[0][:3])**3) / mu)
    T2_kepler = 2 * np.pi * np.sqrt((np.linalg.norm(x0_list[1][:3])**3) / mu)
    T3_kepler = 2 * np.pi * np.sqrt((np.linalg.norm(x0_list[2][:3])**3) / mu)
    T4_kepler = 2 * np.pi * np.sqrt((np.linalg.norm(x0_list[3][:3])**3) / mu)
    T5_kepler = 2 * np.pi * np.sqrt((np.linalg.norm(x0_list[4][:3])**3) / mu)

    print("\n=== Periods [s] (vis-viva) ===")
    print(f"Craft 1: {T1:.3f} (Kepler ref: {T1_kepler:.3f})")
    print(f"Craft 2: {T2:.3f} (Kepler ref: {T2_kepler:.3f})")
    print(f"Craft 3: {T3:.3f} (Kepler ref: {T3_kepler:.3f})")
    print(f"Craft 4: {T4:.3f} (Kepler ref: {T4_kepler:.3f})")
    print(f"Craft 5: {T5:.3f} (Kepler ref: {T5_kepler:.3f})")

    util.save_h5py(
        "./five_spacecraft.h5",
        {
            "ts_mj":   ts_mj,
            "xs_mj_1": xs_MJ_all[0],
            "xs_mj_2": xs_MJ_all[1],
            "xs_mj_3": xs_MJ_all[2],
            "xs_mj_4": xs_MJ_all[3],
            "xs_mj_5": xs_MJ_all[4],
            "debris":  xs_debris_phys,

            # NEW: control forces (ND)
            "u_mj_1": u_mj_np[0],
            "u_mj_2": u_mj_np[1],
            "u_mj_3": u_mj_np[2],
            "u_mj_4": u_mj_np[3],
            "u_mj_5": u_mj_np[4],

            "T1":      np.array([T1]),
            "T2":      np.array([T2]),
            "T3":      np.array([T3]),
            "T4":      np.array([T4]),
            "T5":      np.array([T5]),
        },
        param=param
    )
    print("\nSaved: five_spacecraft.h5")


if __name__ == "__main__":
    main()