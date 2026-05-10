from time import time, sleep
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb

# ============================================================
# Robot / Model Settings
# ============================================================
ROBOT_ADDRESS = "169.254.186.20"
MODEL_PATH = "/home/chu/chu_main/rb5/scene_rb5.xml"
INIT_JOINT_DEG = np.array([90.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float64)

# 목표 자세는 고정값으로 사용
DESIRED_RPY_DEG = np.array([90.0, 0.0, 0.0], dtype=np.float64)

# ============================================================
# VSD Controller Settings
# ============================================================
K_a = 150.0
zeta_a = 4.0

K_o = 3.0
zeta_o = 2.0

varsigma = 1.5

# ============================================================
# Virtual Target Offset Settings
# ============================================================
USE_VIRTUAL_RETURN = True

# 목표점 기준 70 mm 이내부터 virtual target을 사용
VIRTUAL_CONST_ERR_MM = 70.0

# 목표점 10 mm 이내에서는 offset을 줄여서 최종 수렴 유도
VIRTUAL_TAPER_DIST_MM = 18.0

# 목표점에서 멀어지는 중이면 offset을 끔
MOVING_AWAY_DOT_THRESH = 1e-5

# 축별 virtual target 방향 가중치
# y, z 방향 수렴성이 약하면 여기서 가중치를 크게 둠
VIRTUAL_AXIS_WEIGHT = np.array([1.0, 3.0, 3.0], dtype=np.float64)

# ============================================================
# Measured Friction Coefficients
# ============================================================
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396], dtype=np.float64)
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388], dtype=np.float64)

# tanh smoothing coefficient
friction_curve_coef = 8 * 1e-1

# joint별 마찰 보상 스케일
fric_scale = np.array([0.70, 1.0, 1.0, 0.7, 0.8, 0.8], dtype=np.float64)

# ============================================================
# Torque servo timing
# ============================================================
t1 = 0.01  # 이동시간
t2 = 0.05  # 유지시간

MAX_TORQUE = 50.0

# ============================================================
# Globals
# ============================================================
robot = None
rc = None
robot_data = None
state = None

request_exit = False


def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0, 1]
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


def connect_robot():
    global robot, rc, robot_data, state

    try:
        robot = rb.Cobot(ROBOT_ADDRESS)
        rc = rb.ResponseCollector()
        robot_data = CobotData(ROBOT_ADDRESS)
        state = robot_data.request_data()

        robot.set_operation_mode(rc, rb.OperationMode.Real)
        robot.set_speed_bar(rc, 0.5)
        rc.error().throw_if_not_empty()

        print("[ROBOT] Connected")
        print(f"[ROBOT] address = {ROBOT_ADDRESS}")

    except Exception as e:
        print(f"[ROBOT] No Robot Connection ..! {e}")
        raise


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

    return np.array(jpos, dtype=np.float64), np.array(jvel, dtype=np.float64)


def rb_get_joint_position():
    global state

    if state is None:
        raise RuntimeError("Failed to get robot state.")

    return np.array(state.sdata.jnt_ang, dtype=np.float64)


def rb_get_tcp_pose():
    _, tcp_info = robot.get_tcp_info(rc)
    tcp_info = np.array(tcp_info, dtype=np.float64)

    tcp_pos_m = tcp_info[0:3] / 1000.0  # mm -> m
    tcp_rpy_deg = tcp_info[3:6]         # deg

    return tcp_pos_m, tcp_rpy_deg


def rb_get_tcp_pose_from_state(current_state):
    """
    robot.get_tcp_info(rc)를 직접 호출하지 않고,
    robot_data.request_data()로 받은 state.sdata 안의 TCP pose를 사용합니다.

    반환:
        tcp_pos_m   : [x, y, z] m
        tcp_rpy_deg : [rx, ry, rz] deg
    """
    if current_state is None:
        raise RuntimeError("Failed to get robot state.")

    if hasattr(current_state.sdata, "tcp"):
        tcp_info = np.array(current_state.sdata.tcp, dtype=np.float64)

    elif hasattr(current_state.sdata, "tcp_pos"):
        tcp_info = np.array(current_state.sdata.tcp_pos, dtype=np.float64)

    elif hasattr(current_state.sdata, "cur_pos"):
        tcp_info = np.array(current_state.sdata.cur_pos, dtype=np.float64)

    else:
        raise RuntimeError("TCP pose field를 찾지 못했습니다. state.sdata field 이름을 확인해야 합니다.")

    tcp_pos_m = tcp_info[0:3] / 1000.0  # mm -> m
    tcp_rpy_deg = tcp_info[3:6]         # deg

    return tcp_pos_m, tcp_rpy_deg


def rb_move_j_and_wait(joint_deg, vel=60, acc=80):
    joint_deg = np.asarray(joint_deg, dtype=np.float64).reshape(6,)

    print("====================================")
    print(f"[MOVE_J] target joint = {joint_deg}")
    print("====================================")

    robot.move_j(rc, joint_deg, vel, acc)
    rc.error().throw_if_not_empty()

    if robot.wait_for_move_started(rc, 0.5).is_success():
        robot.wait_for_move_finished(rc)
    else:
        print("[MOVE_J] move start check timeout, but continue waiting for finish")
        robot.wait_for_move_finished(rc)

    rc.error().throw_if_not_empty()
    print("[MOVE_J] finished")


def virtual_offset_piecewise_mm(dist_m, const_err_mm=70.0, taper_dist_mm=10.0):
    """
    목표점까지 거리 dist_m에 따라 virtual offset 길이를 계산합니다.

    dist >= const_err:
        offset = 0
        멀리 있을 때는 실제 목표점으로 당김.

    taper_dist < dist < const_err:
        offset = const_err - dist
        목표 근처에서 일정한 pulling force가 유지되도록 virtual target을 goal 바깥으로 둠.

    dist <= taper_dist:
        offset을 0으로 점점 줄임.
        최종 목표점으로 수렴시키기 위한 구간.
    """
    dist_mm = dist_m * 1000.0

    if dist_mm >= const_err_mm:
        L_mm = 0.0

    elif dist_mm > taper_dist_mm:
        L_mm = const_err_mm - dist_mm

    else:
        if taper_dist_mm <= 1e-9:
            L_mm = 0.0
        else:
            L_mm = (const_err_mm - taper_dist_mm) * (dist_mm / taper_dist_mm)

    return L_mm * 0.001


def mujoco_key_callback(keycode):
    global request_exit

    try:
        key = chr(keycode).lower()
    except Exception:
        return

    if key == "q":
        request_exit = True
        print("[MUJOCO KEY] q pressed. Exit requested.")


def main():
    global state, request_exit

    connect_robot()

    # ------------------------------------------------------------
    # 1. Initial move_j and desired target setup
    # ------------------------------------------------------------
    rb_move_j_and_wait(INIT_JOINT_DEG, vel=60, acc=80)
    sleep(1.0)

    # 기존 rb_get_tcp_pose() 대신 state.sdata 기반 TCP pose 사용
    state = robot_data.request_data()
    if state is None:
        raise RuntimeError("[INIT] Failed to get robot state after move_j.")

    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose_from_state(state)

    # 현재 위치를 목표점으로 설정
    desired_xpos_tcp = current_tcp_pos.copy()
    desired_rpy = DESIRED_RPY_DEG.copy()

    print("====================================")
    print("[INIT] Desired TCP position is set from current TCP after move_j")
    print(f"[INIT] current_tcp_pos  = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy  = {current_tcp_rpy}")
    print(f"[INIT] desired_xpos_tcp = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy      = {desired_rpy}")
    print("====================================")

    # ------------------------------------------------------------
    # 2. MuJoCo model setup
    # ------------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    try:
        jpos, jvel = rb_get_joint_state()
        print(f"Initial jpos : {jpos} | jvel : {jvel}")

        if len(jpos) == 6:
            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)
        else:
            raise RuntimeError("Invalid joint state length")

        sleep(1.0)

    except Exception as e:
        print(f"Can't get robot init data ..! {e}")
        d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
        d.qvel[:] = 0.0
        sleep(1.0)

    M = np.zeros((m.nv, m.nv), dtype=np.float64)
    G = np.zeros((m.nv), dtype=np.float64)

    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)
    C0 = np.zeros((6, 6), dtype=np.float64)

    prev_time = time()
    hz_window = []
    prev_jpos = None

    # ------------------------------------------------------------
    # 3. Viewer and control loop
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_key_callback) as viewer:
        while viewer.is_running():
            if request_exit:
                break

            # ----------------------------------------------------
            # Robot state update
            # ----------------------------------------------------
            state = robot_data.request_data()
            if state is None:
                continue

            if state.sdata.op_stat_collision_occur:
                print("Robot in Collision")
                break

            if state.sdata.op_stat_sos_flag == 4:
                print("Command Input Error")
                break

            now = time()
            loop_dt = now - prev_time
            prev_time = now

            if loop_dt <= 1e-9:
                continue

            mujoco.mj_step(m, d)
            mujoco.mj_fullM(m, M, d.qM)

            # ----------------------------------------------------
            # Real robot data -> MuJoCo state
            # ----------------------------------------------------
            try:
                jpos = rb_get_joint_position()
                jvel = np.zeros(6, dtype=np.float64)

                if prev_jpos is not None:
                    jvel = (jpos - prev_jpos) / loop_dt

                prev_jpos = jpos.copy()

                d.qpos[:] = np.deg2rad(jpos)
                d.qvel[:] = np.deg2rad(jvel)

            except Exception as e:
                print(f"real data update failed..! {e}")
                continue

            # ----------------------------------------------------
            # Gravity calculation with qvel = 0
            # ----------------------------------------------------
            qvel_backup = deepcopy(d.qvel)

            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            mujoco.mj_rne(m, d, 0, G)

            d.qvel[:] = qvel_backup[:]
            mujoco.mj_forward(m, d)

            # ----------------------------------------------------
            # Joint damping matrix C0
            # ----------------------------------------------------
            np.fill_diagonal(
                C0,
                varsigma * np.sqrt(np.sum(np.abs(M[0:6, 0:6]), axis=1))
            )

            # ----------------------------------------------------
            # Jacobian
            # ----------------------------------------------------
            tcp_site_id = m.site("tcp").id
            mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

            jacp0 = deepcopy(jacp[:, 0:6])
            jacr0 = deepcopy(jacr[:, 0:6])

            # ----------------------------------------------------
            # TCP pose and errors
            # ----------------------------------------------------
            # 기존:
            # tcp_pos, tcp_rpy = rb_get_tcp_pose()
            #
            # 변경:
            # 같은 루프에서 이미 받은 state.sdata에서 TCP pose를 읽음
            tcp_pos, tcp_rpy = rb_get_tcp_pose_from_state(state)

            R_current = rpy_to_rotmat(*tcp_rpy)
            R_desired = rpy_to_rotmat(*desired_rpy)

            ori_err0 = (
                np.cross(R_desired[:, 0], R_current[:, 0]) +
                np.cross(R_desired[:, 1], R_current[:, 1]) +
                np.cross(R_desired[:, 2], R_current[:, 2])
            )

            w0 = jacr0 @ d.qvel[0:6]
            F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

            xpos_dot0 = jacp0 @ d.qvel[0:6]

            # ----------------------------------------------------
            # Virtual Target Offset
            # ----------------------------------------------------
            to_goal = desired_xpos_tcp - tcp_pos
            goal_dist = np.linalg.norm(to_goal)

            to_goal_weighted = VIRTUAL_AXIS_WEIGHT * to_goal
            weighted_goal_dist = np.linalg.norm(to_goal_weighted)

            if weighted_goal_dist > 1e-9:
                u_goal = to_goal_weighted / weighted_goal_dist
            else:
                u_goal = np.zeros(3, dtype=np.float64)

            goal_err = tcp_pos - desired_xpos_tcp

            # 현재 속도가 목표점에서 멀어지는 방향이면 True
            moving_away = np.dot(goal_err, xpos_dot0) > MOVING_AWAY_DOT_THRESH

            if USE_VIRTUAL_RETURN and not moving_away:
                L_virtual = virtual_offset_piecewise_mm(
                    goal_dist,
                    const_err_mm=VIRTUAL_CONST_ERR_MM,
                    taper_dist_mm=VIRTUAL_TAPER_DIST_MM,
                )
            else:
                L_virtual = 0.0

            virtual_xpos_tcp = desired_xpos_tcp + u_goal * L_virtual

            # ----------------------------------------------------
            # VSD force
            # ----------------------------------------------------
            xpos_err0 = tcp_pos - virtual_xpos_tcp

            force0 = (
                (K_a * xpos_err0)
                + (zeta_a * np.sqrt(K_a) * xpos_dot0)
            )

            # ----------------------------------------------------
            # Measured friction compensation
            # ----------------------------------------------------
            Tf = fric_scale * (
                Cfc * np.tanh(friction_curve_coef * jvel)
                + Vfc * jvel
            )

            # ----------------------------------------------------
            # Final torque command
            # ----------------------------------------------------
            torque0 = (
                -1 * C0 @ d.qvel[0:6]
                -1 * jacp0.T @ force0
                +1 * G[0:6]
                -1 * jacr0.T @ F_ori_0
                +1 * Tf[0:6]
            )

            d.ctrl[0:6] = np.clip(torque0, -MAX_TORQUE, MAX_TORQUE)

            if np.any(np.abs(jvel) > 70):
                fast_idx = np.where(np.abs(jvel) > 70)[0]

                if len(fast_idx) > 0:
                    d.ctrl[fast_idx] = 0.0
                    print(torque0)
                    print(
                        f"Joint Velocity is too fast ...! "
                        f"Joint{list(fast_idx)} | Jvel : {jvel[fast_idx]}"
                    )

            target_torque = d.ctrl[0:6]

            # ----------------------------------------------------
            # Debug print
            # ----------------------------------------------------
            goal_err_norm = np.linalg.norm(goal_err)
            vel_norm = np.linalg.norm(xpos_dot0)

            hz = 1.0 / loop_dt
            hz_window.append(hz)

            if len(hz_window) > 30:
                hz_window.pop(0)

            print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f})")
            print(f"[TCP_DES] {desired_xpos_tcp}")
            print(f"[TCP_VIRTUAL] {virtual_xpos_tcp}")
            print(f"[TCP_CUR] {tcp_pos}")
            print(f"[GOAL_ERR] {goal_err} | norm = {goal_err_norm:.4f}")
            print(f"[VIRTUAL_L] {L_virtual * 1000.0:.3f} mm")
            print(f"[MOVING_AWAY] {moving_away}")
            print(f"[VEL_NORM] {vel_norm:.5f}")
            print(f"[RPY_DES] {desired_rpy}")
            print(f"[RPY_CUR] {tcp_rpy}")
            print(f"[ORI_ERR] {ori_err0} | norm = {np.linalg.norm(ori_err0):.4f}")
            print(f"[F_ORI] {F_ori_0}")
            print(f"[C0 diag] {np.diag(C0)}")
            print(f"[Tf measured] {Tf}")
            print(f"[Target torque] {target_torque}")
            print("------------------------------------")

            # ----------------------------------------------------
            # Torque servo command
            # ----------------------------------------------------
            try:
                ret = robot.move_servo_t(
                    rc,
                    target_torque,
                    t1,
                    t2,
                    compensation=0
                )

                sleep(0.005)

                if not ret.is_success():
                    print(f"move_servo_t failed: {ret}")

            except Exception as e:
                print(f"T-servo Failed ..! {e}")

            viewer.sync()

    print("[EXIT] finished")


if __name__ == "__main__":
    main()