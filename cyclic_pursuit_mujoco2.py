"""
Objective:
Visualize 5 spacecraft performing cyclic pursuit around a debris object
that coasts in a straight line in an inertial MuJoCo world.

Translation:
    - Controlled via cyclic pursuit -> forces in data.xfrc_applied[bid,:3]

Attitude Control:
    - Controlled via so3_pd_attitude_controller -> torques via actuators M_body_x, etc.
    - Each satellite's +x body axis points at the debris.
"""

import os
import time
import numpy as np
import mujoco


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def skew_matrix_to_vector(mat):
    # Your definition
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


def R_from_euler_zyx(phi, theta, psi):
    # R = Rz(psi) * Ry(theta) * Rx(phi) (body->world)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x  # Combined Rotation Matrix


def R_from_quat(q):
    # MuJoCo uses (w,x,y,z)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])
    return R


def euler_zyx_from_R(R):
    """
    Inverse of R_from_euler_zyx.
    Returns (phi, theta, psi) for R = Rz(psi)*Ry(theta)*Rx(phi).
    """
    # theta from R[2,0]
    theta = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    # phi from R[2,1], R[2,2]
    phi = np.arctan2(R[2, 1], R[2, 2])
    # psi from R[1,0], R[0,0]
    psi = np.arctan2(R[1, 0], R[0, 0])
    return phi, theta, psi


# ---------------------------------------------------------------------
# Your SO(3) PD attitude controller
# ---------------------------------------------------------------------
def so3_pd_attitude_controller(x, param, euler_d=(0.0, 0.0, 0.0), omega_d=np.zeros(3)):
    """
    x: Newton-Euler state [r, (phi,theta,psi), v, omega_body]
    param: dict with "inertia" (3x3)

    Returns u in R^6, where u[3:6] are body torques.
    """
    I = param["inertia"]
    # Desired attitude & rates
    R_d = R_from_euler_zyx(*euler_d)
    # Current attitude & rates
    R = R_from_euler_zyx(*x[3:6])
    omega = x[9:12]                       # body frame

    # SO(3) attitude error (Tayebi/Lee et al.)
    e_R = 0.5 * skew_matrix_to_vector(R_d.T @ R - R.T @ R_d)
    # Angular velocity error
    e_omega = omega - (R.T @ R_d @ omega_d)   # body frame

    # Diagonal gains (Nm/rad and Nm/(rad/s))
    Kp = 0.1 * np.diag([50.0, 50.0, 30.0])
    Kd = 0.1 * np.diag([100.0, 100.0, 60.0])

    # Computed-torque PD (cancels -ω×Iω term in dynamics)
    tau = - Kp @ e_R - Kd @ e_omega + np.cross(omega, I @ omega)

    u = np.zeros(6)
    u[3:6] = tau
    return u

def cyclic_pursuit_centered_axis(p, pdot, R0, kp=0.5, kr=0.2, kd=0.3, alpha=np.deg2rad(60.0), k_axis=0.0, axis=np.array([0.0, 0.0, 1.0])):
    """
    Cyclic pursuit controller around a moving debris center in the XY plane,
    with optional extra rotation about an arbitrary axis through the debris.

    Parameters
    ----------
    p : Relative positions to debris (r_i - r_debris) in world frame.
    pdot : Relative velocities to debris (v_i - v_debris) in world frame.
    R0 : Desired ring radius in 3D (same as your original controller).
    kp, kr, kd : Tangential pursuit gain, radial gain, and damping gain.
    alpha : Pursuit angle (radians). 60–90 degrees works well.
    k_axis : Gain for extra rotation around `axis`. Set small (e.g. 0.1 or less).
    axis : Axis to rotate around (world frame). Will be normalized.

    Returns
    -------
    u : Control accelerations in world frame for each satellite.
    """
    p = np.asarray(p)
    pdot = np.asarray(pdot)
    N = p.shape[0]
    assert p.shape == pdot.shape == (N, 3)

    u = np.zeros_like(p)

    # Original XY-plane pursuit (about z) -------------------------------
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R_alpha = np.array([[ca, -sa], [sa,  ca]])

    # Normalize rotation axis for extra swirl
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) < 1e-9: axis = np.array([0.0, 0.0, 1.0])
    n = axis / np.linalg.norm(axis)

    for i in range(N):
        j = (i + 1) % N  # neighbor index
        pi = p[i]
        pj = p[j]
        vi = pdot[i]

        # --- Cyclic Pursuit in XY Plane --------------------------
        p_i_xy = pi[:2]
        p_j_xy = pj[:2]

        vij_xy = p_j_xy - p_i_xy
        vij_tan_xy = R_alpha @ vij_xy

        r_norm = np.linalg.norm(p_i_xy) + 1e-9
        e_r = r_norm - R0
        r_hat_xy = p_i_xy / r_norm

        u_xy = kp * vij_tan_xy - kr * e_r * r_hat_xy - kd * vi[:2]

        u_i = np.zeros(3)
        u_i[0] = u_xy[0]
        u_i[1] = u_xy[1]
        u_i[2] = -kd * vi[2]   # out-of-plane damping

        # --- Extra rotation around arbitrary axis (pure swirl) -------
        if k_axis != 0.0:
            # This is tangential to p_i (orthogonal), so it doesn't change |p_i|
            u_axis = k_axis * np.cross(n, pi)
            u_i += u_axis

        u[i] = u_i

    return u


# ---------------------------------------------------------------------
# Build desired attitude: body +x points at debris
# ---------------------------------------------------------------------
def build_Rd_face_debris(direction_world,
                         up_world=np.array([0.0, 0.0, 1.0])):
    """
    direction_world: vector FROM satellite TO debris, in world frame.
    Returns R_d (body->world) so that:
        body +x axis || direction_world
        body +z axis ~ up_world (fixes roll)
    """
    d = np.asarray(direction_world)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.eye(3)

    # Forward axis (body +x in world)
    x_w = d / n

    # Make z_w ≈ up_world but orthogonal to x_w
    up = up_world / (np.linalg.norm(up_world) + 1e-12)
    z_w = up - np.dot(up, x_w) * x_w
    z_n = np.linalg.norm(z_w)
    if z_n < 1e-8:
        z_w = np.array([0.0, 0.0, 1.0])
        z_n = 1.0
    z_w /= z_n

    # y_w completes right-handed frame
    y_w = np.cross(z_w, x_w)

    # Columns = body axes expressed in world frame
    R_d = np.column_stack((x_w, y_w, z_w))
    return R_d


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # ----- Simulation configuration -----
    N_SATS = 5
    T_sim = 120.0          # total simulation time [s]
    mj_dt = 0.02           # MuJoCo timestep [s]
    realtime_factor = 1.0  # 1.0 for realtime, <1 to slow down

    # Debris straight-line velocity and position
    v_debris0 = np.array([0.2, 0.0, 0.0])   # m/s in +x
    r_debris0 = np.array([0.0, -2.0, 0.0])  # start a bit "below" origin

    # Desired ring radius around debris
    R0 = 1.0

    # ----- Load MuJoCo model -----
    xml_path = os.path.join(os.path.dirname(__file__), "no_earth.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Could not find no_earth.xml at {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.gravity[:] = 0.0
    model.opt.timestep = mj_dt

    # ----- IDs -----
    body_names = ["spacecraft", "spacecraft2", "spacecraft3", "spacecraft4", "spacecraft5"]
    joint_names = ["base6dof", "base6dof2", "base6dof3", "base6dof4", "base6dof5"]

    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]

    deb_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "debris")
    deb_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "debris6dof")

    # ----- Set initial states -----
    # Debris: straight-line motion (no forces → coasting)
    qpos_adr_deb = model.jnt_qposadr[deb_jid]
    qvel_adr_deb = model.jnt_dofadr[deb_jid]

    # qpos: [x,y,z, qw,qx,qy,qz]
    data.qpos[qpos_adr_deb:qpos_adr_deb+3] = r_debris0
    data.qpos[qpos_adr_deb+3:qpos_adr_deb+7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat
    # qvel: [vx,vy,vz, wx,wy,wz]
    data.qvel[qvel_adr_deb:qvel_adr_deb+3] = v_debris0
    data.qvel[qvel_adr_deb+3:qvel_adr_deb+6] = 0.0

    # Satellites: ring around debris with zero relative velocity
    angles = np.linspace(0.0, 2.0 * np.pi * (1.0 - 1.0 / N_SATS), N_SATS)
    for bid, jid, ang in zip(bids, jids, angles):
        r_rel = np.array([R0 * np.cos(ang), R0 * np.sin(ang), 0.0])
        v_rel = np.zeros(3)

        r_i0 = r_debris0 + r_rel
        v_i0 = v_debris0 + v_rel

        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]

        data.qpos[qpos_adr:qpos_adr+3] = r_i0
        data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
        data.qvel[qvel_adr:qvel_adr+3] = v_i0
        data.qvel[qvel_adr+3:qvel_adr+6] = 0.0

    mujoco.mj_forward(model, data)

    # ----- Inertia (body frame) -----
    I_sats = []
    for bid in bids:
        I_vec = model.body_inertia[bid]   # (Ixx, Iyy, Izz)
        I_sats.append(np.diag(I_vec))
    I_sats = np.asarray(I_sats)

    # ----- Map torque actuators (M_body_x, M_body_x2, ...) -----
    M_x_ids, M_y_ids, M_z_ids = [], [], []
    for i in range(N_SATS):
        if i == 0:
            suffix = ""        # "M_body_x"
        else:
            suffix = str(i+1)  # "M_body_x2", ...

        name_Mx = f"M_body_x{suffix}"
        name_My = f"M_body_y{suffix}"
        name_Mz = f"M_body_z{suffix}"

        M_x_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_Mx))
        M_y_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_My))
        M_z_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_Mz))

    for i in range(N_SATS):
        if M_x_ids[i] < 0 or M_y_ids[i] < 0 or M_z_ids[i] < 0:
            print(f"[WARN] Missing torque actuators for satellite {i+1}.")

    # ----- Viewer -----
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        use_viewer = True

        # Tracking camera that follows debris
        if deb_bid == -1:
            print("[WARN] Body 'debris' not found, using free camera.")
        else:
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = deb_bid
            cam.distance = 0.8
            cam.elevation = -20.0
            cam.azimuth = 140.0
            print("[INFO] Tracking debris with mjCAMERA_TRACKING.")
    except Exception:
        class _Dummy:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def sync(self): pass
        viewer = _Dummy()
        use_viewer = False

    num_steps = int(T_sim / mj_dt)
    print(f"Starting simulation for {T_sim} s ({num_steps} steps)")

    with viewer:
        for step in range(num_steps):
            # IMPORTANT: attitude uses actuators only; translation via xfrc only
            # data.ctrl[:] = 0.0  # clear torques; forces handled by xfrc_applied

            # 1) Debris state
            r_debris = data.qpos[qpos_adr_deb:qpos_adr_deb+3].copy()
            v_debris = data.qvel[qvel_adr_deb:qvel_adr_deb+3].copy()

            # 2) Satellites: translation + rotation state
            p = np.zeros((N_SATS, 3))
            pdot = np.zeros((N_SATS, 3))

            r_sat_list = []
            R_list = []
            omega_body_list = []

            for i, jid in enumerate(jids):
                qpos_adr = model.jnt_qposadr[jid]
                qvel_adr = model.jnt_dofadr[jid]

                # qpos: [x,y,z, qw,qx,qy,qz]
                r_i = data.qpos[qpos_adr:qpos_adr+3].copy()
                q_i = data.qpos[qpos_adr+3:qpos_adr+7].copy()
                # qvel: [vx,vy,vz, wx,wy,wz]
                v_i = data.qvel[qvel_adr:qvel_adr+3].copy()
                w_world = data.qvel[qvel_adr+3:qvel_adr+6].copy()

                # Relative translation
                p[i] = r_i - r_debris
                pdot[i] = v_i - v_debris

                # Orientation & angular velocity in body frame
                R_i = R_from_quat(q_i)   # body->world
                R_bw = R_i.T             # world->body
                omega_body = R_bw @ w_world

                r_sat_list.append(r_i)
                R_list.append(R_i)
                omega_body_list.append(omega_body)

            # 3) Cyclic pursuit forces (unchanged)
            # u = cyclic_pursuit_centered_axis(
            #     p, pdot, R0=R0,
            #     kp=0.8, kr=0.4, kd=0.6,
            #     alpha=np.deg2rad(60.0),
            #     axis=np.array([1.0, 0.0, 1.0]),
            # )

            axis = np.array([0.0, 0.0, 1.0])
            u = cyclic_pursuit_centered_axis(p, pdot, R0=R0,
                                        kp=0.8, kr=0.4, kd=0.6,
                                        alpha=np.deg2rad(60.0),
                                        k_axis=0.01,  # start very small
                                        axis=axis)

            # Clear all external forces/torques
            data.xfrc_applied[:] = 0.0

            # Apply forces (world frame): F = m * a
            for i, bid in enumerate(bids):
                m_i = model.body_mass[bid]
                F_i = m_i * u[i]
                data.xfrc_applied[bid, :3] = F_i  # ONLY forces here

            # 4) Attitude control: nose (+x) points at debris
            for i, bid in enumerate(bids):
                r_i = r_sat_list[i]
                R_i = R_list[i]  # body->world
                omega_body = omega_body_list[i]
                I_i = I_sats[i]

                # Vector from satellite TO debris
                dir_i = r_debris - r_i
                if np.linalg.norm(dir_i) < 1e-6:
                    continue

                # Desired rotation so +x faces debris
                R_d_i = build_Rd_face_debris(dir_i, up_world=np.array([0.0, 0.0, 1.0]))
                phi_d, theta_d, psi_d = euler_zyx_from_R(R_d_i)
                euler_d = (phi_d, theta_d, psi_d)

                # Current euler from R_i
                phi, theta, psi = euler_zyx_from_R(R_i)

                # Build Newton-Euler state x = [r, euler, v, omega_body]
                x = np.zeros(12)
                x[0:3] = r_i
                x[3:6] = np.array([phi, theta, psi])
                x[6:9] = 0.0  # linear vel not used
                x[9:12] = omega_body

                param = {"inertia": I_i}

                u_att = so3_pd_attitude_controller(
                    x,
                    param,
                    euler_d=euler_d,
                    omega_d=np.zeros(3)
                )

                tau_body = u_att[3:6]  # body-frame torques

                # Convert to world-frame torque and apply via xfrc_applied

                tau_world = R_i @ tau_body  # body -> world
                data.xfrc_applied[bid, 3:6] = tau_world

            # 5) Step + render
            mujoco.mj_step(model, data)

            if use_viewer:
                viewer.sync()
                time.sleep(mj_dt / realtime_factor)

    print("Simulation finished.")


if __name__ == "__main__":
    main()