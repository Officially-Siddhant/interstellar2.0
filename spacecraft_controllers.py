
import numpy as np 
from spacecraft_dynamics import *

def empty_controller(x):
	return np.zeros(6,)



from spacecraft_dynamics import force_gravity_world_nd  # if not already imported elsewhere

def pursue_debris_with_grav_comp_PI(
    r_i_nd, v_i_nd,
    r_debris_nd, v_debris_nd,
    model, bid,
    e_r_int_prev, dt_nd,
    r_offset_nd=None,
    kp=1.0, kv=2.0, ki=0.05
):
    """
    Gravity compensation + PD + Integral pursuit controller (ND frame).
    """

    # 1) Gravity compensation
    Fg_nd = force_gravity_world_nd(r_i_nd)
    F_comp_nd = -Fg_nd

    # 2) Desired point to track
    if r_offset_nd is None:
        r_des = r_debris_nd
    else:
        r_des = r_debris_nd + r_offset_nd

    # 3) Errors
    e_r = r_des - r_i_nd
    e_v = v_debris_nd - v_i_nd

    # 4) Integrate position error
    e_r_int = e_r_int_prev + e_r * dt_nd

    # 5) PI tracking acceleration
    a_track = kp * e_r + kv * e_v + ki * e_r_int

    # 6) Convert to force
    m_nd = model.body_mass[bid]
    F_track_nd = m_nd * a_track

    return F_comp_nd + F_track_nd, e_r_int


# def cyclic_pursuit_controller(pos, vel):
#     """
#     Cyclic pursuit controller in 3D (world frame).
#
#     Args:
#         pos:    (N, 3) positions p_i
#         vel:    (N, 3) velocities v_i
#     Returns:
#         u: (N, 3) accelerations (this is F_c / m)
#     """
#     pos = np.asarray(pos, dtype=float)
#     vel = np.asarray(vel, dtype=float)
#     N, dim = pos.shape
#     assert dim == 3
#
#     # kd    = params.get("kd", 1.0)
#     # kc    = params.get("kc", 0.5)
#     # kg    = params.get("kg", 1.0)
#     # alpha = params.get("alpha", np.pi / 6.0)
#     kd = 1.0
#     kc = 0.5
#     kg = 1.0
#     alpha = np.pi / 6.0
#
#     # rotation about orbit normal (assumed z)
#     ca, sa = np.cos(alpha), np.sin(alpha)
#     Rz = np.array([[ca, -sa, 0.0],
#                    [sa,  ca, 0.0],
#                    [0.0, 0.0, 1.0]])
#
#     u = np.zeros_like(pos)
#
#     for i in range(N):
#         j = (i + 1) % N  # follower -> next agent
#
#         dx = pos[j] - pos[i]   # x_{i+1} - x_i
#         dv = vel[j] - vel[i]   # xdot_{i+1} - xdot_i
#
#         # TR(α)T^{-1} ≈ Rz in our simplified coordinates
#         rel_term  = kd * (Rz @ dx) + (Rz @ dv)
#         self_term = -kc * kd * pos[i] - (kc + kd / kg) * vel[i]
#
#         u[i] = kg * (rel_term + self_term)
#
#     return u



def cyclic_pursuit_controller(pos, vel, pos_debris, vel_debris, params):
    """
    Cyclic pursuit in a debris-centred LVLH frame.

    Args:
        pos        : (N, 3) spacecraft positions in world frame (ND)
        vel        : (N, 3) spacecraft velocities in world frame (ND)
        pos_debris : (3,)   debris position in world frame (ND)
        vel_debris : (3,)   debris velocity in world frame (ND)
        params     : dict with gains, e.g.
            {
                "kd": 1.0,         # pursuit gain
                "kc": 0.5,         # radial stiffness in LVLH
                "kv": 2.0,         # velocity tracking gain
                "alpha": np.pi/5,  # bearing angle (2π/5 for 5-craft ring)
            }

    Returns:
        accel_world : (N, 3) control accelerations in world frame (F_c / m)
    """

    pos        = np.asarray(pos, dtype=float)
    vel        = np.asarray(vel, dtype=float)
    pos_debris = np.asarray(pos_debris, dtype=float).reshape(3)
    vel_debris = np.asarray(vel_debris, dtype=float).reshape(3)

    N, dim = pos.shape
    assert dim == 3, "pos must be (N,3)"
    assert vel.shape == (N, 3)

    kd    = params.get("kd", 1.0)
    kc    = params.get("kc", 0.5)
    kv    = params.get("kv", 2.0)
    alpha = params.get("alpha", 2.0 * np.pi / N)  # default: equal spacing

    # ------------------------------------------------------------------
    # 1) Build LVLH frame at debris (origin = debris, x: radial, y: along-track, z: orbit normal)
    # ------------------------------------------------------------------
    r_d = pos_debris
    v_d = vel_debris

    r_norm = np.linalg.norm(r_d)
    if r_norm < 1e-8:
        raise ValueError("Debris position too close to origin for LVLH frame.")

    x_hat = r_d / r_norm  # radial

    h = np.cross(r_d, v_d)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-8:
        # Fallback: pick an arbitrary normal if v_d is nearly radial
        h = np.cross(r_d, np.array([0.0, 0.0, 1.0]))
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-8:
            h = np.cross(r_d, np.array([0.0, 1.0, 0.0]))
            h_norm = np.linalg.norm(h)
    z_hat = h / h_norm           # orbit normal
    y_hat = np.cross(z_hat, x_hat)  # along-track

    # Orthonormal basis, columns are LVLH axes expressed in world frame
    R_lvlh_to_world = np.column_stack((x_hat, y_hat, z_hat))  # 3x3
    R_world_to_lvlh = R_lvlh_to_world.T                       # rotation inverse

    # ------------------------------------------------------------------
    # 2) Express spacecraft states in debris-centred LVLH coords
    # ------------------------------------------------------------------
    # Relative w.r.t. debris in world
    r_rel_world = pos - r_d       # (N,3)
    v_rel_world = vel - v_d       # (N,3)

    # Rotate into LVLH
    r_rel_lvlh = (R_world_to_lvlh @ r_rel_world.T).T   # (N,3)
    v_rel_lvlh = (R_world_to_lvlh @ v_rel_world.T).T   # (N,3)

    # ------------------------------------------------------------------
    # 3) Cyclic pursuit + radial "spring" in LVLH
    # ------------------------------------------------------------------
    ca, sa = np.cos(alpha), np.sin(alpha)
    # rotation about LVLH z-axis (orbit normal)
    Rz = np.array([[ca, -sa, 0.0],
                   [sa,  ca, 0.0],
                   [0.0, 0.0, 1.0]])

    accel_lvlh = np.zeros_like(r_rel_lvlh)

    for i in range(N):
        j = (i + 1) % N  # neighbor in the ring

        dx = r_rel_lvlh[j] - r_rel_lvlh[i]  # neighbor relative position in LVLH
        dv = v_rel_lvlh[j] - v_rel_lvlh[i]  # neighbor relative velocity in LVLH

        # Single-integrator desired velocity field in LVLH
        #   v_des_i = kd * Rz * (x_j - x_i)  -  kc * x_i
        v_des_i = kd * (Rz @ dx) - kc * r_rel_lvlh[i]

        # Double-integrator control:
        #   u_i = -kv * (v_i - v_des_i)
        accel_lvlh[i] = -kv * (v_rel_lvlh[i] - v_des_i)

    # ------------------------------------------------------------------
    # 4) Rotate control accelerations back to world frame
    # ------------------------------------------------------------------
    accel_world = (R_lvlh_to_world @ accel_lvlh.T).T  # (N,3)

    return accel_world

def cyclic_pursuit_controller_no_debris(pos, vel, params):
    """
    Cyclic pursuit on a common orbital ring around Earth (no debris yet).

    Args:
        pos:    (N, 3) positions in world frame (ND)
        vel:    (N, 3) velocities in world frame (ND)
        params: dict with gains:
            "kd": pursuit gain
            "kc": radial stiffness in orbital plane
            "kv": velocity tracking gain
            "alpha": bearing angle (≈ 2π/N for equal spacing)

    Returns:
        accel_world: (N, 3) accelerations in world frame (F_c / m)
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    N, dim = pos.shape
    assert dim == 3

    kd    = params.get("kd", 1.0)
    kc    = params.get("kc", 0.05)
    kv    = params.get("kv", 2.0)
    alpha = params.get("alpha", 2.0 * np.pi / N)

    # ---- 1) Estimate common orbit normal (k_hat) from angular momentum ----
    h_vecs = np.cross(pos, vel)          # (N,3)
    h_bar  = h_vecs.mean(axis=0)
    h_norm = np.linalg.norm(h_bar)
    if h_norm < 1e-8:
        # Fallback normal if something is degenerate
        k_hat = np.array([0.0, 0.0, 1.0])
    else:
        k_hat = h_bar / h_norm           # orbit normal (unit)

    # ---- 2) Rodrigues rotation matrix around k_hat by alpha ----
    kx, ky, kz = k_hat
    Kx = np.array([[   0, -kz,  ky],
                   [  kz,   0, -kx],
                   [ -ky,  kx,   0]])
    I  = np.eye(3)
    R  = I + np.sin(alpha) * Kx + (1.0 - np.cos(alpha)) * (Kx @ Kx)

    accel_world = np.zeros_like(pos)

    for i in range(N):
        j = (i + 1) % N  # neighbor index

        dx = pos[j] - pos[i]     # world
        dv = vel[j] - vel[i]

        # --- Project everything into the orbital plane ---
        dx_plane = dx - np.dot(dx, k_hat) * k_hat
        r_i_plane = pos[i] - np.dot(pos[i], k_hat) * k_hat
        v_i_plane = vel[i] - np.dot(vel[i], k_hat) * k_hat

        # Single–integrator desired velocity in the plane:
        # v_des = kd * R * (x_j - x_i)_plane  -  kc * x_i_plane
        v_des = kd * (R @ dx_plane) - kc * r_i_plane

        # Double–integrator control:
        # a_i = -kv * (v_i_plane - v_des)
        a_plane = -kv * (v_i_plane - v_des)

        # Put it back into world frame (already in world basis since k_hat, R are)
        accel_world[i] = a_plane

    return accel_world


def pursue_debris_with_grav_comp(r_i_nd, v_i_nd, r_debris_nd, v_debris_nd,model, bid,kp=1.0, kv=2.0,r_offset_nd=None,):
    """
    Gravity-compensated debris pursuit (ND, world frame).

    Args:
        r_i_nd      : (3,) chaser position in ND frame
        v_i_nd      : (3,) chaser velocity in ND frame
        r_debris_nd : (3,) debris position in ND frame
        v_debris_nd : (3,) debris velocity in ND frame
        model       : MuJoCo model (to read ND mass)
        bid         : body id for this chaser
        kp, kv      : position / velocity gains
        r_offset_nd : (3,) desired offset from debris (ND), e.g. tangential ring.
                      If None, we just chase the debris directly.

    Returns:
        F_total_nd  : (3,) total control force in ND frame
                      (already includes gravity compensation).
    """
    # 1) Gravity compensation
    Fg_nd = force_gravity_world_nd(r_i_nd)   # existing function: central gravity (ND)
    F_comp_nd = -Fg_nd                       # cancel gravity

    # 2) Relative pursuit (free-space PD in ND frame)
    if r_offset_nd is None:
        r_des = r_debris_nd
    else:
        r_des = r_debris_nd + r_offset_nd    # track a point on the "ring" around debris

    e_r = r_des - r_i_nd                     # position error
    e_v = v_debris_nd - v_i_nd               # velocity matching error

    a_track_nd = kp * e_r + kv * e_v         # desired acceleration in ND

    m_nd = model.body_mass[bid]              # ND mass (often 1.0)
    F_track_nd = m_nd * a_track_nd

    # 3) Total force = gravity compensation + pursuit
    F_total_nd = F_comp_nd + F_track_nd
    return F_total_nd

# def pursue_debris_with_grav_comp_PI(
#     r_i_nd, v_i_nd,
#     r_debris_nd, v_debris_nd,
#     model, bid,
#     e_r_int_prev,  # (3,) integral of position error from previous step
#     dt_nd,         # ND timestep
#     kp=1.0, kv=2.0, ki=0.1,
#     r_offset_nd=None,
# ):
#     """
#     Gravity–compensated debris pursuit with PI on position error (ND).
#
#     Args:
#         r_i_nd      : (3,) chaser position in ND frame
#         v_i_nd      : (3,) chaser velocity in ND frame
#         r_debris_nd : (3,) debris position in ND frame
#         v_debris_nd : (3,) debris velocity in ND frame
#         model       : MuJoCo model
#         bid         : body id for this chaser
#         e_r_int_prev: (3,) integral of position error from previous step
#         dt_nd       : scalar, ND timestep
#         kp, kv, ki  : gains
#         r_offset_nd : (3,) desired offset from debris (ND) or None
#
#     Returns:
#         F_total_nd  : (3,) total control force (ND)
#         e_r_int_new : (3,) updated integral of position error
#     """
#     # 1) Gravity compensation
#     Fg_nd     = force_gravity_world_nd(r_i_nd)
#     F_comp_nd = -Fg_nd
#
#     # 2) Desired point (debris or ring offset)
#     if r_offset_nd is None:
#         r_des = r_debris_nd
#     else:
#         r_des = r_debris_nd + r_offset_nd
#
#     # 3) Errors
#     e_r = r_des - r_i_nd          # position error
#     # e_v = v_debris_nd - v_i_nd    # velocity error
#     e_v = v_i_nd
#
#     # 4) Integral update (simple forward Euler)
#     e_r_int_new = e_r_int_prev + dt_nd * e_r
#
#     # 5) PI tracking acceleration
#     a_track_nd = kp * e_r + kv * e_v + ki * e_r_int_new
#
#     # 6) Force
#     m_nd      = model.body_mass[bid]
#     F_track_nd = m_nd * a_track_nd
#
#     F_total_nd = F_comp_nd + F_track_nd
#     return F_total_nd, e_r_int_new

def pursue_debris_with_spacing_and_cyclic(
    R_nd, V_nd,
    r_debris_nd, v_debris_nd,
    model, bids,
    kp=1.0, kv=2.0,
    k_rep=0.5, d_min=0.1,
    k_cp=0.5, alpha=2.0*np.pi/5.0,
    ring_offsets_nd=None,
):
    """
    Gravity-compensated debris pursuit with inter-agent spacing + cyclic pursuit.

    Args:
        R_nd          : (N, 3) positions of all chasers in ND frame
        V_nd          : (N, 3) velocities of all chasers in ND frame
        r_debris_nd   : (3,) debris position in ND
        v_debris_nd   : (3,) debris velocity in ND
        model         : MuJoCo model (for ND mass)
        bids          : list of body ids, length N (same order as rows of R_nd, V_nd)
        kp, kv        : position / velocity gains for debris tracking
        k_rep         : repulsion gain between agents
        d_min         : desired minimum spacing between agents
        k_cp          : cyclic pursuit gain (tangential term)
        alpha         : look-ahead bearing angle for cyclic pursuit (e.g. 2π/5 for pentagon)
        ring_offsets_nd : list of (3,) offsets in ND, one per agent, or None

    Returns:
        F_all_nd      : (N, 3) forces for all agents (already includes gravity compensation)
    """
    R_nd = np.asarray(R_nd, dtype=float)
    V_nd = np.asarray(V_nd, dtype=float)
    N, dim = R_nd.shape
    assert dim == 3

    if ring_offsets_nd is None:
        ring_offsets_nd = [None] * N

    # Rotation about orbit normal (assume +z is orbit normal / out-of-plane)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rz = np.array([[ca, -sa, 0.0],
                   [sa,  ca, 0.0],
                   [0.0, 0.0, 1.0]])

    F_all_nd = np.zeros_like(R_nd)

    for i in range(N):
        r_i = R_nd[i]
        v_i = V_nd[i]
        bid = bids[i]

        # ---------- 1) Gravity compensation ----------
        Fg_nd   = force_gravity_world_nd(r_i)  # existing ND central gravity
        F_comp  = -Fg_nd                       # cancel gravity

        # ---------- 2) Debris tracking (with optional ring offset) ----------
        offset_i = ring_offsets_nd[i]
        if offset_i is None:
            r_des = r_debris_nd
        else:
            r_des = r_debris_nd + np.asarray(offset_i)

        e_r = r_des - r_i
        e_v = v_debris_nd - v_i
        a_track = kp * e_r + kv * e_v

        # ---------- 3) Inter-agent spacing (repulsion) ----------
        a_rep = np.zeros(3)
        for j in range(N):
            if j == i:
                continue
            dij = R_nd[i] - R_nd[j]
            dist = np.linalg.norm(dij)
            if dist < 1e-8:
                continue
            # Only repel if closer than desired spacing
            if dist < d_min:
                # Simple inverse-distance repulsion
                a_rep += k_rep * (1.0/dist - 1.0/d_min) * (dij / dist)

        # ---------- 4) Cyclic pursuit tangential term ----------
        # Use next agent in ring as "leader"
        j_next = (i + 1) % N
        dx = R_nd[j_next] - R_nd[i]   # neighbor direction
        dv = V_nd[j_next] - V_nd[i]

        # Rotate relative position into a tangential direction around debris
        # (approximate orbit normal = +z)
        dx_tan = Rz @ dx
        a_cp   = k_cp * dx_tan  # simple single-integrator-style pursuit

        # ---------- 5) Combine accelerations and map to force ----------
        a_total = a_track + a_rep + a_cp
        m_nd    = model.body_mass[bid]
        F_total = F_comp + m_nd * a_total

        F_all_nd[i] = F_total

    return F_all_nd

def grav_comp_pd_attitude_controller(x):
    Kp = 10*np.diag([10.0, 10.0, 10.0])
    Kd = 1*np.diag([5.0, 5.0, 5.0])
    euler_d = np.array([0,0,0])
    R_d = R_from_euler_zyx(*euler_d)
    omega_d = np.zeros(3,)
    R = R_from_euler_zyx(*x[3:6])
    error = 1/2 * skew_matrix_to_vector(R_d.T @ R - R.T @ R_d)
    error_rate = x[9:12] - R.T @ R_d @ omega_d
    u = np.zeros(6,)
    u[3:6] = - Kp @ error - Kd @ x[9:12]
    # print("u",u)
    return u 


def so3_pd_attitude_controller(x, param, euler_d=(0.0, 0.0, 0.0), omega_d=np.zeros(3)):
    """
    x: Newton-Euler state [r, (phi,theta,psi), v, omega_body]
    param: dict with "inertia" (3x3)
    """
    I = param["inertia"]
    # Desired attitude & rates
    R_d = R_from_euler_zyx(*euler_d)
    # Current attitude & rates
    R = R_from_euler_zyx(*x[3:6])                 # body->world
    omega = x[9:12]                                # body frame
    # SO(3) attitude error (Tayebi/Lee et al.)
    e_R = 0.5 * skew_matrix_to_vector(R_d.T @ R - R.T @ R_d)
    # Angular velocity error
    e_omega = omega - (R.T @ R_d @ omega_d)        # body frame

    # Diagonal gains (Nm/rad and Nm/(rad/s))
    # Kp = 15*np.diag([50.0, 50.0, 30.0]) 
    # Kd = 15*np.diag([100.0, 100.0, 60.0]) 
    Kp = 0.1*np.diag([50.0, 50.0, 30.0]) 
    Kd = 0.1*np.diag([100.0, 100.0, 60.0]) 

    # Computed-torque PD (cancels -ω×Iω term in dynamics)
    tau = - Kp @ e_R - Kd @ e_omega + np.cross(omega, I @ omega)

    # (Optional) clip to keep MuJoCo from hard-clipping after ND conversion
    # tau = np.clip(tau, -tau_max, tau_max)

    u = np.zeros(6)
    u[3:6] = tau
    return u



# ----------------------------------------------------------
# Helper: CW drift dynamics f(x)
# x = [x, y, z]   and xdot = [xd, yd, zd]
# ----------------------------------------------------------
def cw_drift(x, xd, n):
    """
    Compute CW drift dynamics f(x) so that control law can cancel it.

    State:
        x  = [x, y, z]
        xd = [xd, yd, zd]

    Returns:
        f = [xdd, ydd, zdd] (drift acceleration)
    """

    x1, x2, x3 = x
    v1, v2, v3 = xd

    fx = 3 * n**2 * x1 + 2 * n * v2
    fy = -2 * n * v1
    fz = -n**2 * x3

    return np.array([fx, fy, fz])


# ----------------------------------------------------------
# Cyclic Pursuit Controller in CW Frame
# ----------------------------------------------------------
def cw_cyclic_pursuit_controller(
    x_i,
    xd_i,
    x_j,
    xd_j,
    n,
    kd=1.0,
    kc=0.1,
    kg=1.0,
    alpha=np.deg2rad(72)   # default for 5-agent cycle
):
    """
    Implements the cyclic-pursuit controller from the paper:

        u_i = -f(x_i)
              + k_g [ k_d T R(α) T^{-1} (x_j - x_i)
                       + T R(α) T^{-1} (xd_j - xd_i) ]
              - k_c k_d x_i
              - (k_c + k_d/k_g) xd_i

    All states are assumed already in the Hill/CW frame.

    Inputs
    -------
    x_i      : (3,) agent i position in CW frame
    xd_i     : (3,) agent i velocity in CW frame
    x_j      : (3,) next agent position in CW frame
    xd_j     : (3,) next agent velocity in CW frame
    n        : mean motion (rad/s)
    kd, kc   : spacing & damping gains
    kg       : overall pursuit gain
    alpha    : rotation angle for cyclic pursuit (rad)

    Output
    -------
    u_i : (3,) control acceleration in CW frame
    """

    # --- 1) Drift dynamics term ---
    f_i = cw_drift(x_i, xd_i, n)

    # --- 2) Relative states ---
    dx  = x_j  - x_i
    dxd = xd_j - xd_i

    # --- 3) Rotation matrix R(α) in 2D plane ---
    c = np.cos(alpha)
    s = np.sin(alpha)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    # T and T^{-1} are identity in basic CW frame → simplify
    # If you later use tilted/transformed coordinates, replace T = something.

    T = np.eye(3)
    Tinv = np.eye(3)

    # Term: T R(α) T^{-1} dx
    Rdx  = T @ (R @ (Tinv @ dx))
    Rdxd = T @ (R @ (Tinv @ dxd))

    # --- 4) Cyclic pursuit tracking terms ---
    pursuit_term = kg * (kd * Rdx + Rdxd)

    # --- 5) Additional damping and spacing terms ---
    spacing_term = -kc * kd * x_i
    damping_term = -(kc + kd / kg) * xd_i

    # --- 6) Combine control ---
    u_i = -f_i + pursuit_term + spacing_term + damping_term

    return u_i