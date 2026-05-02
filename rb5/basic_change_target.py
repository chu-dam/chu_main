from time import time, sleep
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import SystemVariable, CobotData
import rbpodo as rb


######## torque servo
try:
    ROBOT_ADDRESS = "169.254.186.20"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()

    robot_data = CobotData(ROBOT_ADDRESS)
    state = robot_data.request_data()

    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)

    # robot.set_freedrive_mode(rc, on=False)

    t1 = 0.01  # 이동시간
    t2 = 0.05  # 유지시간

except Exception as e:
    print(f"No Robot Connection ..! {e}")
    pass
########


def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # ZYX (yaw-pitch-roll) 순서
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])

    Rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    return Rz @ Ry @ Rx


def rb_get_joint_state():
    jpos = []
    jvel = []

    for i in range(6):
        var_pos = getattr(SystemVariable, f"SD_J{i}_ANG")
        var_vel = getattr(SystemVariable, f"SD_J{i}_VEL")

        _, pos = robot.get_system_variable(rc, var_pos)
        _, vel = robot.get_system_variable(rc, var_vel)

        jpos.append(pos)
        jvel.append(vel)

    return np.array(jpos), np.array(jvel)


def rb_get_joint_position():
    global state

    if state is None:
        print("Failed to get robot state.")
        return np.array([])

    return np.array(state.sdata.jnt_ang)


def rb_get_tcp_pose():
    _, tcp_info = robot.get_tcp_info(rc)

    tcp_pos = np.array(tcp_info[0:3]) / 1000.0  # mm -> m
    tcp_rpy = np.array(tcp_info[3:6])           # deg

    return tcp_pos, tcp_rpy


def rb_move_j_and_wait(joint_deg, vel=60, acc=80):
    joint_deg = np.asarray(joint_deg, dtype=np.float64)

    print(f"[INIT_MOVE_J] target joint = {joint_deg}")

    robot.move_j(rc, joint_deg, vel, acc)
    rc.error().throw_if_not_empty()

    if robot.wait_for_move_started(rc, 0.5).is_success():
        robot.wait_for_move_finished(rc)
    else:
        print("[INIT_MOVE_J] move start check timeout, but continue waiting for finish")
        robot.wait_for_move_finished(rc)

    rc.error().throw_if_not_empty()

    print("[INIT_MOVE_J] finished")


# --------- initial move_j + desired setting ---------
INIT_JOINT_DEG = np.array(
    [90.0, 0.0, -90.0, 0.0, -90.0, 0.0],
    dtype=np.float64
)

# 목표 자세는 기존 방식 유지
desired_rpy = np.array([90.0, 0.0, 0.0], dtype=np.float64)

# target B는 월드 좌표계 기준 절대 TCP 위치 [m]
TARGET_B_XPOS_TCP = np.array([0.111, -0.503, 0.25], dtype=np.float64)

try:
    # 1. 원하는 초기 joint 자세로 이동
    rb_move_j_and_wait(INIT_JOINT_DEG, vel=60, acc=80)

    # 2. 이동 후 안정화 시간
    sleep(1.0)

    # 3. 현재 TCP pose 읽기
    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose()

    # 4. target A = move_j 이후 현재 TCP 위치
    target_a_xpos_tcp = current_tcp_pos.copy()

    # 5. target B = 월드 좌표계 기준 절대 TCP 위치
    target_b_xpos_tcp = TARGET_B_XPOS_TCP.copy()

    # 6. 처음 목표는 target A
    desired_xpos_tcp = target_a_xpos_tcp.copy()
    target_mode = "A"

    print("[INIT] Target A is set from current TCP after move_j")
    print(f"[INIT] current_tcp_pos   = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy   = {current_tcp_rpy}")
    print(f"[INIT] target_a_xpos_tcp = {target_a_xpos_tcp}")
    print(f"[INIT] target_b_xpos_tcp = {target_b_xpos_tcp}")
    print(f"[INIT] desired_xpos_tcp  = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy       = {desired_rpy}")
    print("[KEY] Press 's' in MuJoCo viewer to toggle target A <-> B")
    print("[KEY] Press 'q' in MuJoCo viewer to quit")

except Exception as e:
    print(f"[INIT] move_j failed or target setting failed: {e}")

    # 실패해도 현재 위치를 목표 위치로 잡아서 갑자기 튀는 것 방지
    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose()

    target_a_xpos_tcp = current_tcp_pos.copy()
    target_b_xpos_tcp = TARGET_B_XPOS_TCP.copy()

    desired_xpos_tcp = target_a_xpos_tcp.copy()
    target_mode = "A"

    print("[INIT] Fallback: Target A is set from current TCP")
    print(f"[INIT] current_tcp_pos   = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy   = {current_tcp_rpy}")
    print(f"[INIT] target_a_xpos_tcp = {target_a_xpos_tcp}")
    print(f"[INIT] target_b_xpos_tcp = {target_b_xpos_tcp}")
    print(f"[INIT] desired_xpos_tcp  = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy       = {desired_rpy}")
# ----------------------------------------------------


request_exit = False


def toggle_target():
    global desired_xpos_tcp
    global target_mode

    if target_mode == "A":
        desired_xpos_tcp = target_b_xpos_tcp.copy()
        target_mode = "B"
    else:
        desired_xpos_tcp = target_a_xpos_tcp.copy()
        target_mode = "A"

    print("====================================")
    print(f"[MUJOCO KEY] target_mode = {target_mode}")
    print(f"[MUJOCO KEY] desired_xpos_tcp = {desired_xpos_tcp}")
    print(f"[MUJOCO KEY] desired_rpy      = {desired_rpy}")
    print("====================================")


def mujoco_key_callback(keycode):
    global request_exit

    try:
        key = chr(keycode).lower()
    except Exception:
        return

    if key == "s":
        toggle_target()

    elif key == "q":
        request_exit = True
        print("[MUJOCO KEY] q pressed. Exit requested.")


model_path = "/home/chu/chu_main/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


# initial robot pose
try:
    jpos, jvel = rb_get_joint_state()
    print(f"Initial jpos : {jpos} | jvel : {jvel}")

    while len(jpos) == 0:
        print("Waiting for robot init data..")
        sleep(0.1)
        jpos, jvel = rb_get_joint_state()

    d.qpos[:] = np.deg2rad(jpos)
    d.qvel[:] = 0.0

    sleep(3)

except Exception as e:
    print(f"Can't get robot init data ..! {e}")

    d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
    d.qvel[:] = 0.0

    sleep(3)


M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

C0 = np.zeros((6, 6), dtype=np.float64)

K_a = 100.0
zeta_a = 5.0

K_o = 2.0
zeta_o = 4.0

varsigma = 1.0


######################
# Friction Coef
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396])
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388])
friction_curve_coef = 8 * 1e-1
fric_scale = np.array([0.73, 1.0, 1.0, 0.8, 0.8, 0.8])
######################


prev_time = time()
hz_window = []
prev_jpos = None


with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=mujoco_key_callback
) as viewer:
    t0 = time()

    while viewer.is_running():
        if request_exit:
            break

        # -----------------------------
        # Real robot state update
        # -----------------------------
        state = robot_data.request_data()

        if state is None:
            print("[WARN] robot_data.request_data() failed")
            sleep(0.01)
            continue

        if state.sdata.op_stat_collision_occur:
            print("Robot in Collision")
            break

        if state.sdata.op_stat_sos_flag == 4:
            print(f"Command Input Error | JVEL : {jvel}")
            break

        now = time()
        loop_dt = now - prev_time
        prev_time = now

        if loop_dt <= 0.0:
            continue

        mujoco.mj_step(m, d)
        # mujoco.mj_fullM(m, M, d.qM)

        try:
            # real data update
            jpos = rb_get_joint_position()

            if len(jpos) != 6:
                print("[WARN] invalid joint position data")
                sleep(0.01)
                continue

            jvel = np.zeros(6)

            if prev_jpos is not None:
                jvel = (jpos - prev_jpos) / loop_dt

            prev_jpos = jpos.copy()

            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)

        except Exception as e:
            print(f"real data update failed..! {e}")
            continue

        # -----------------------------
        # Gravity calculation
        # -----------------------------
        qvel_backup = deepcopy(d.qvel)

        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G)

        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)

        mujoco.mj_fullM(m, M, d.qM)

        # -----------------------------
        # Joint damping matrix C0
        # -----------------------------
        np.fill_diagonal(
            C0,
            varsigma * np.sqrt(np.sum(np.abs(M[0:6, 0:6]), axis=1))
        )

        # -----------------------------
        # Jacobian
        # -----------------------------
        tcp_site_id = m.site("tcp").id

        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        jacp0 = deepcopy(jacp[:, 0:6])
        jacr0 = deepcopy(jacr[:, 0:6])

        # -----------------------------
        # TCP pose
        # -----------------------------
        tcp_pos, tcp_rpy = rb_get_tcp_pose()

        # Position error
        xpos_err0 = tcp_pos - desired_xpos_tcp

        # Orientation error
        R_current = rpy_to_rotmat(*tcp_rpy)
        R_desired = rpy_to_rotmat(*desired_rpy)

        ori_err0 = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        )

        # Angular velocity
        w0 = jacr0 @ d.qvel[0:6]

        # Orientation force
        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # Linear velocity
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        # Linear force
        force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

        # Friction torque
        Tf = fric_scale * (
            Cfc * np.tanh(friction_curve_coef * jvel) +
            Vfc * jvel
        )

        # -----------------------------
        # Torque decomposition
        # -----------------------------
        tau_damp = -1 * C0 @ d.qvel[0:6]
        tau_pos = -1 * jacp0.T @ force0
        tau_g = 1 * G[0:6]
        tau_ori = -0 * jacr0.T @ F_ori_0
        tau_fric = 1 * Tf[0:6]

        # Torque total
        torque0 = (- 1 * C0 @ d.qvel[0:6]
                   - 1 * jacp0.T @ force0
                   + 1 * G[0:6]
                   - 1 * jacr0.T @ F_ori_0
                   + 1 * Tf[0:6])

        max_torque = 50
        d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)

        # Joint velocity limit
        if np.any(np.abs(jvel) > 70):
            i = np.where(np.abs(jvel) > 70)[0]

            if len(i) > 0:
                d.ctrl[i] = 0.0 * torque0[i]

                print(torque0)
                print(f"Joint Velocity is too fast ...! Joint{list(i)} | Jvel : {jvel[i]}")

        target_torque = d.ctrl[0:6]

        # -----------------------------
        # Print log
        # -----------------------------
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)

            if len(hz_window) > 30:
                hz_window.pop(0)

            print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f})")
            print(f"[TARGET_MODE] {target_mode}")

            print(f"[TARGET_A] {target_a_xpos_tcp}")
            print(f"[TARGET_B] {target_b_xpos_tcp}")

            print(f"[TCP_DES] {desired_xpos_tcp}")
            print(f"[TCP_CUR] {tcp_pos}")
            print(f"[TCP_ERR] {xpos_err0} | norm = {np.linalg.norm(xpos_err0):.4f}")

            print(f"[TARGET_DELTA] {target_b_xpos_tcp - target_a_xpos_tcp}")
            print(f"[FORCE_TASK] {force0}")
            print(f"[X_DOT] {xpos_dot0}")

            print(f"[RPY_DES] {desired_rpy}")
            print(f"[RPY_CUR] {tcp_rpy}")
            print(f"[ORI_ERR] {ori_err0} | norm = {np.linalg.norm(ori_err0):.4f}")
            print(f"[F_ORI] {F_ori_0}")

            print(f"[C0 diag] {np.diag(C0)}")

            print(f"[TAU_DAMP] {tau_damp}")
            print(f"[TAU_POS ] {tau_pos}")
            print(f"[TAU_G   ] {tau_g}")
            print(f"[TAU_ORI ] {tau_ori}")
            print(f"[TAU_FRIC] {tau_fric}")
            print(f"[TAU_SUM ] {torque0}")
            print(f"[TAU_CMD ] {d.ctrl[0:6]}")

            print("------------------------------------")

        # -----------------------------
        # Torque servoing input
        # -----------------------------
        try:
            target_torque = d.ctrl[0:6]

            ret = robot.move_servo_t(
                rc,
                target_torque,
                t1,
                t2,
                compensation=0
            )

            # comp 0 : u
            # comp 1 : u + g
            # comp 2 : u + f
            # comp 3 : u + g + f

            sleep(0.005)

            if not ret.is_success():
                print("move_servo_t 실패 ", ret)

        except Exception as e:
            print(f"T-servo Failed ..! {e}")

        viewer.sync()

print("[EXIT] Program finished")