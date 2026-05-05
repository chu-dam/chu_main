from time import time, sleep
from copy import deepcopy
import os
import csv
from datetime import datetime

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

INIT_JOINT_DEG = np.array([89.94, -4.23, -115.11, 29.35, -89.83, -0.04], dtype=np.float64)

# 기존 방식 유지: 목표 자세는 고정값으로 사용
DESIRED_RPY_DEG = np.array([90.0, 0.0, 0.0], dtype=np.float64)

# ============================================================
# Automated 26-point evaluation settings
# ============================================================
AUTO_EXPERIMENT = True

# 정육면체 한 변 길이 = 2x 이므로, 여기 값이 x입니다.
# 예: 0.030 m -> 한 변 60 mm짜리 cube, 목표점 기준 ±30 mm
CUBE_HALF_LENGTH_M = 0.15

# move_l 절대 이동 속도/가속도
# rbpodo 예제에서는 300, 400을 사용했지만, 실제 장비에서는 보수적으로 시작하는 것을 권장합니다.
MOVE_L_SPEED = 80.0
MOVE_L_ACC = 120.0
MOVE_L_SETTLE_SEC = 0.5

# move_servo_t 마지막 명령이 완전히 끝나기 전에 move_l이 들어가면
# "previous motion is not finished" 계열 오류가 발생할 수 있습니다.
# t1 + t2보다 충분히 크게 둡니다.
SERVO_TO_MOVE_L_WAIT_SEC = 0.30

# move_l 이동 시 사용할 자세
# True  : DESIRED_RPY_DEG 사용
# False : move_j 후 측정된 현재 TCP RPY 사용
MOVE_L_USE_DESIRED_RPY = True

# 모든 point를 측정한 뒤 자동 종료
EXIT_AFTER_ALL_TESTS = True

# ============================================================
# VSD Controller Settings
# ============================================================
K_a = 150.0
zeta_a = 4.0

K_o = 4.0
zeta_o = 2.5

varsigma = 0.5

# Virtual Target
USE_VIRTUAL_RETURN = True
VIRTUAL_CONST_ERR_MM = 70.0
VIRTUAL_TAPER_DIST_MM = 10.0
MOVING_AWAY_DOT_THRESH = 1e-5

# ============================================================
# CSV Logging Settings
# ============================================================
EXPERIMENT_NAME = "fric03_offset_on_auto_26pt"
CSV_SAVE_DIR = f"./vsd_eval_logs/{EXPERIMENT_NAME}"

# TCP 속도가 이 값보다 작게 일정 시간 유지되면 수렴이 멈춘 것으로 판단
LOG_STOP_VEL_MPS = 0.002       # 2 mm/s
LOG_STOP_HOLD_SEC = 1.0
LOG_MIN_SAVE_SEC = 0.5

# ============================================================
# Friction Coefficients
# ============================================================
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396])
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388])
friction_curve_coef = 8 * 1e-1

# ============================================================
# Torque servo timing
# ============================================================
t1 = 0.01  # 이동시간
t2 = 0.05  # 유지시간

# ============================================================
# Globals
# ============================================================
robot = None
rc = None
robot_data = None
state = None

request_csv_start = False
request_exit = False

csv_logging = False
csv_log_data = []
csv_log_count = 0
csv_start_time = None
csv_stop_candidate_time = None
active_test_info = None


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


def rb_move_j_and_wait(joint_deg, vel=60, acc=80):
    joint_deg = np.asarray(joint_deg, dtype=np.float64).reshape(6,)

    print("====================================")
    print(f"[MOVE_J] target joint = {joint_deg}")
    print("====================================")

    ret = robot.move_j(rc, joint_deg, vel, acc)
    rc.error().throw_if_not_empty()

    if robot.wait_for_move_started(rc, 0.5).is_success():
        robot.wait_for_move_finished(rc)
    else:
        print("[MOVE_J] move start check timeout, but continue waiting for finish")
        robot.wait_for_move_finished(rc)

    rc.error().throw_if_not_empty()
    print("[MOVE_J] finished")


def rb_move_l_and_wait(point_mm_deg, speed=MOVE_L_SPEED, acceleration=MOVE_L_ACC):
    """
    Absolute linear TCP motion.

    point_mm_deg = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    """
    point_mm_deg = np.asarray(point_mm_deg, dtype=np.float64).reshape(6,)

    print("====================================")
    print(f"[MOVE_L] target TCP = {point_mm_deg}")
    print(f"[MOVE_L] speed={speed}, acc={acceleration}")
    print("====================================")

    ret = robot.move_l(rc, point_mm_deg, speed, acceleration)
    rc.error().throw_if_not_empty()

    if robot.wait_for_move_started(rc, 0.5).is_success():
        robot.wait_for_move_finished(rc)
    else:
        print("[MOVE_L] move start check timeout, but continue waiting for finish")
        robot.wait_for_move_finished(rc)

    rc.error().throw_if_not_empty()
    print("[MOVE_L] finished")


def make_tcp_point_mm_deg(pos_m, rpy_deg):
    pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3,)
    rpy_deg = np.asarray(rpy_deg, dtype=np.float64).reshape(3,)

    return np.concatenate([pos_m * 1000.0, rpy_deg])


def virtual_offset_piecewise_mm(dist_m, const_err_mm=70.0, taper_dist_mm=10.0):
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


def generate_26_cube_points(center_m, half_length_m):
    """
    목표점 G=(a,b,c)을 정육면체 중심으로 두고 26개 시작점을 생성합니다.

    한 변 길이 = 2x, half_length_m = x

    구성:
    - 윗면 z=c+x    : 9개
    - 중심 평면 z=c : 8개, 중앙 G 제외
    - 아랫면 z=c-x : 9개
    """
    center_m = np.asarray(center_m, dtype=np.float64).reshape(3,)
    x = float(half_length_m)

    grid = [-x, 0.0, x]
    layers = [
        ("top", +x, True),
        ("center", 0.0, False),
        ("lower", -x, True),
    ]

    points = []
    idx = 1

    for layer_name, dz, include_center in layers:
        for dy in [-x, 0.0, x]:
            for dx in [-x, 0.0, x]:
                if (not include_center) and abs(dx) < 1e-12 and abs(dy) < 1e-12:
                    continue

                offset_m = np.array([dx, dy, dz], dtype=np.float64)
                start_pos_m = center_m + offset_m

                points.append({
                    "index": idx,
                    "layer": layer_name,
                    "dx_m": dx,
                    "dy_m": dy,
                    "dz_m": dz,
                    "start_x_m": start_pos_m[0],
                    "start_y_m": start_pos_m[1],
                    "start_z_m": start_pos_m[2],
                    "start_pos_m": start_pos_m,
                })
                idx += 1

    if len(points) != 26:
        raise RuntimeError(f"26 points must be generated, but got {len(points)}")

    return points


def print_test_points(test_points):
    print("====================================")
    print("[TEST POINTS] 26 cube start points")
    print(f"[TEST POINTS] cube half length x = {CUBE_HALF_LENGTH_M * 1000.0:.1f} mm")
    print("------------------------------------")
    for p in test_points:
        print(
            f"pt{p['index']:02d} | {p['layer']:6s} | "
            f"offset(mm)=({p['dx_m']*1000.0:+.1f}, {p['dy_m']*1000.0:+.1f}, {p['dz_m']*1000.0:+.1f}) | "
            f"start(m)=({p['start_x_m']:.4f}, {p['start_y_m']:.4f}, {p['start_z_m']:.4f})"
        )
    print("====================================")


def save_csv_log(log_data, log_index, tag="complete", test_info=None):
    if len(log_data) == 0:
        return

    os.makedirs(CSV_SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if test_info is not None:
        point_tag = f"pt{int(test_info['index']):02d}_{test_info['layer']}"
    else:
        point_tag = "manual"

    filename = os.path.join(
        CSV_SAVE_DIR,
        f"{EXPERIMENT_NAME}_{tag}_{point_tag}_{timestamp}_{log_index:03d}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "test_index",
            "layer",
            "dx_m",
            "dy_m",
            "dz_m",
            "start_x_m",
            "start_y_m",
            "start_z_m",
            "goal_x_m",
            "goal_y_m",
            "goal_z_m",
            "time_sec",
            "tcp_x_m",
            "tcp_y_m",
            "tcp_z_m",
            "goal_err_x_m",
            "goal_err_y_m",
            "goal_err_z_m",
            "goal_err_norm_m",
            "goal_err_norm_mm",
            "vel_norm_mps",
        ])

        writer.writerows(log_data)

    print("====================================")
    print(f"[CSV SAVE] {filename}")
    print(f"[CSV SAVE] samples = {len(log_data)}")
    print("====================================")


def start_csv_logging(now, test_info=None):
    global csv_logging, csv_log_count, csv_log_data
    global csv_start_time, csv_stop_candidate_time, active_test_info

    if csv_logging:
        print("[CSV START] ignored. CSV logging is already running.")
        return

    csv_logging = True
    csv_log_count += 1
    csv_log_data = []
    csv_start_time = now
    csv_stop_candidate_time = None
    active_test_info = deepcopy(test_info) if test_info is not None else None

    print("====================================")
    if active_test_info is not None:
        print(
            f"[CSV START] pt{active_test_info['index']:02d} | "
            f"layer={active_test_info['layer']} | "
            f"offset(mm)=({active_test_info['dx_m']*1000.0:+.1f}, "
            f"{active_test_info['dy_m']*1000.0:+.1f}, "
            f"{active_test_info['dz_m']*1000.0:+.1f})"
        )
    else:
        print(f"[CSV START] manual log index = {csv_log_count}")
    print("====================================")


def stop_csv_logging_and_save(tag="complete"):
    global csv_logging, csv_log_data, csv_start_time
    global csv_stop_candidate_time, active_test_info

    save_csv_log(csv_log_data, csv_log_count, tag=tag, test_info=active_test_info)

    csv_logging = False
    csv_log_data = []
    csv_start_time = None
    csv_stop_candidate_time = None
    active_test_info = None


def mujoco_key_callback(keycode):
    global request_csv_start, request_exit

    try:
        key = chr(keycode).lower()
    except Exception:
        return

    if key == "s":
        request_csv_start = True
        print("====================================")
        print("[MUJOCO KEY] s pressed. CSV logging start requested.")
        print("[MUJOCO KEY] In AUTO_EXPERIMENT mode, s key is not needed.")
        print("====================================")

    elif key == "q":
        request_exit = True
        print("[MUJOCO KEY] q pressed. Exit requested.")


def main():
    global state, request_csv_start, request_exit
    global csv_logging, csv_log_data, csv_start_time, csv_stop_candidate_time
    global active_test_info

    connect_robot()

    # ------------------------------------------------------------
    # 1. Initial move_j and desired target setup
    # ------------------------------------------------------------
    rb_move_j_and_wait(INIT_JOINT_DEG, vel=60, acc=80)
    sleep(1.0)

    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose()
    desired_xpos_tcp = current_tcp_pos.copy()
    desired_rpy = DESIRED_RPY_DEG.copy()

    if MOVE_L_USE_DESIRED_RPY:
        move_l_rpy = desired_rpy.copy()
    else:
        move_l_rpy = current_tcp_rpy.copy()

    print("====================================")
    print("[INIT] Desired TCP position is set from current TCP after move_j")
    print(f"[INIT] current_tcp_pos  = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy  = {current_tcp_rpy}")
    print(f"[INIT] desired_xpos_tcp = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy      = {desired_rpy}")
    print(f"[INIT] move_l_rpy       = {move_l_rpy}")
    print("====================================")

    test_points = generate_26_cube_points(desired_xpos_tcp, CUBE_HALF_LENGTH_M)
    print_test_points(test_points)

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
        d.qvel[:] = 0
        sleep(1.0)

    M = np.zeros((m.nv, m.nv), dtype=np.float64)
    G = np.zeros((m.nv), dtype=np.float64)

    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)
    C0 = np.zeros((6, 6), dtype=np.float64)

    prev_time = time()
    hz_window = []
    prev_jpos = None

    experiment_state = "MOVE_TO_START" if AUTO_EXPERIMENT else "RUN_TEST"
    test_cursor = 0
    next_move_l_allowed_time = 0.0

    # ------------------------------------------------------------
    # 3. Viewer and control loop
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_key_callback) as viewer:
        while viewer.is_running():
            if request_exit:
                break

            # ----------------------------------------------------
            # AUTO mode: move_l로 다음 시작점 이동
            # ----------------------------------------------------
            if AUTO_EXPERIMENT and experiment_state == "MOVE_TO_START":
                # 직전 move_servo_t 유지 시간이 끝나기 전에 move_l을 보내지 않도록 대기
                if time() < next_move_l_allowed_time:
                    viewer.sync()
                    sleep(0.01)
                    continue

                if test_cursor >= len(test_points):
                    print("====================================")
                    print("[AUTO EXPERIMENT] All 26 test points completed.")
                    print("====================================")
                    if EXIT_AFTER_ALL_TESTS:
                        break
                    else:
                        experiment_state = "RUN_TEST"
                        continue

                test_info = test_points[test_cursor]
                test_cursor += 1

                target_pose_mm_deg = make_tcp_point_mm_deg(test_info["start_pos_m"], move_l_rpy)

                print("====================================")
                print(f"[AUTO EXPERIMENT] Moving to test point {test_info['index']:02d}/26")
                print(f"[AUTO EXPERIMENT] layer = {test_info['layer']}")
                print(
                    f"[AUTO EXPERIMENT] offset(mm) = "
                    f"({test_info['dx_m']*1000.0:+.1f}, "
                    f"{test_info['dy_m']*1000.0:+.1f}, "
                    f"{test_info['dz_m']*1000.0:+.1f})"
                )
                print("====================================")

                try:
                    rb_move_l_and_wait(target_pose_mm_deg, speed=MOVE_L_SPEED, acceleration=MOVE_L_ACC)
                    sleep(MOVE_L_SETTLE_SEC)

                    # move_l 이후 조인트 속도 추정이 튀지 않도록 초기화
                    prev_jpos = None
                    prev_time = time()

                    # move_l 완료 후, VSD 복귀 제어가 시작되는 순간부터 자동 로깅
                    start_csv_logging(time(), test_info=test_info)
                    experiment_state = "RUN_TEST"

                except Exception as e:
                    print(f"[AUTO EXPERIMENT] move_l failed at pt{test_info['index']:02d}: {e}")
                    break

                viewer.sync()
                continue

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
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            mujoco.mj_rne(m, d, 0, G)
            d.qvel[:] = qvel_backup[:]
            mujoco.mj_forward(m, d)

            np.fill_diagonal(C0, varsigma * np.sqrt(np.sum(np.abs(M[0:6, 0:6]), axis=1)))

            tcp_site_id = m.site("tcp").id
            mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
            jacp0 = deepcopy(jacp[:, 0:6])
            jacr0 = deepcopy(jacr[:, 0:6])

            # ----------------------------------------------------
            # TCP pose and errors
            # ----------------------------------------------------
            tcp_pos, tcp_rpy = rb_get_tcp_pose()

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
            # Virtual Target
            # ----------------------------------------------------
            to_goal = desired_xpos_tcp - tcp_pos
            goal_dist = np.linalg.norm(to_goal)

            virtual_axis_weight = np.array([1.0, 3.0, 2.0])
            to_goal_weighted = virtual_axis_weight * to_goal
            weighted_goal_dist = np.linalg.norm(to_goal_weighted)

            if weighted_goal_dist > 1e-9:
                u_goal = to_goal_weighted / weighted_goal_dist
            else:
                u_goal = np.zeros(3)

            goal_err = tcp_pos - desired_xpos_tcp
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
            # Control force / torque
            # ----------------------------------------------------
            xpos_err0 = tcp_pos - virtual_xpos_tcp
            force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

            fric_scale = np.array([0.70, 1.0, 1.0, 0.7, 0.8, 0.8])
            Tf = fric_scale * (Cfc * np.tanh(friction_curve_coef * jvel) + Vfc * jvel)

            torque0 = (
                -1 * C0 @ d.qvel[0:6]
                -1 * jacp0.T @ force0
                +1 * G[0:6]
                -1 * jacr0.T @ F_ori_0
                +0 * Tf[0:6]
            )

            max_torque = 50
            d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)

            if np.any(np.abs(jvel) > 70):
                fast_idx = np.where(np.abs(jvel) > 70)[0]
                if len(fast_idx) > 0:
                    d.ctrl[fast_idx] = 0.0
                    print(torque0)
                    print(f"Joint Velocity is too fast ...! Joint{list(fast_idx)} | Jvel : {jvel[fast_idx]}")

            target_torque = d.ctrl[0:6]

            # ----------------------------------------------------
            # CSV Logging
            # ----------------------------------------------------
            goal_err_norm = np.linalg.norm(goal_err)
            vel_norm = np.linalg.norm(xpos_dot0)
            test_finished = False

            # Manual mode에서만 s 키 로깅 허용
            if request_csv_start:
                request_csv_start = False
                if not AUTO_EXPERIMENT:
                    start_csv_logging(now, test_info=None)
                else:
                    print("[CSV START] ignored because AUTO_EXPERIMENT=True")

            if csv_logging:
                csv_t = now - csv_start_time

                if active_test_info is not None:
                    test_index = active_test_info["index"]
                    layer = active_test_info["layer"]
                    dx_m = active_test_info["dx_m"]
                    dy_m = active_test_info["dy_m"]
                    dz_m = active_test_info["dz_m"]
                    start_x_m = active_test_info["start_x_m"]
                    start_y_m = active_test_info["start_y_m"]
                    start_z_m = active_test_info["start_z_m"]
                else:
                    test_index = -1
                    layer = "manual"
                    dx_m = dy_m = dz_m = 0.0
                    start_x_m = start_y_m = start_z_m = np.nan

                csv_log_data.append([
                    test_index,
                    layer,
                    dx_m,
                    dy_m,
                    dz_m,
                    start_x_m,
                    start_y_m,
                    start_z_m,
                    desired_xpos_tcp[0],
                    desired_xpos_tcp[1],
                    desired_xpos_tcp[2],
                    csv_t,
                    tcp_pos[0],
                    tcp_pos[1],
                    tcp_pos[2],
                    goal_err[0],
                    goal_err[1],
                    goal_err[2],
                    goal_err_norm,
                    goal_err_norm * 1000.0,
                    vel_norm,
                ])

                if vel_norm < LOG_STOP_VEL_MPS:
                    if csv_stop_candidate_time is None:
                        csv_stop_candidate_time = now

                    elif ((now - csv_stop_candidate_time) >= LOG_STOP_HOLD_SEC) and (csv_t >= LOG_MIN_SAVE_SEC):
                        stop_csv_logging_and_save(tag="complete")
                        test_finished = True

                else:
                    csv_stop_candidate_time = None

            # 수렴 종료 직후에는 torque servo_t를 한 번 더 보내지 않고 다음 move_l로 넘어감
            if test_finished:
                if AUTO_EXPERIMENT:
                    # 마지막 move_servo_t 명령의 유지 시간이 끝난 뒤 다음 move_l을 보내도록 지연
                    next_move_l_allowed_time = time() + SERVO_TO_MOVE_L_WAIT_SEC
                    experiment_state = "MOVE_TO_START"
                    viewer.sync()
                    continue

            # ----------------------------------------------------
            # Debug print
            # ----------------------------------------------------
            hz = 1.0 / loop_dt
            hz_window.append(hz)
            if len(hz_window) > 30:
                hz_window.pop(0)

            print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f})")
            if active_test_info is not None:
                print(f"[TEST] pt{active_test_info['index']:02d}/26 | layer={active_test_info['layer']}")
            print(f"[TCP_DES] {desired_xpos_tcp}")
            print(f"[TCP_VIRTUAL] {virtual_xpos_tcp}")
            print(f"[TCP_CUR] {tcp_pos}")
            print(f"[GOAL_ERR] {goal_err} | norm = {goal_err_norm:.4f}")
            print(f"[VIRTUAL_L] {L_virtual * 1000.0:.3f} mm")
            print(f"[MOVING_AWAY] {moving_away}")
            print(f"[VEL_NORM] {vel_norm:.5f}")
            print(f"[CSV_LOGGING] {csv_logging}")
            print(f"[RPY_DES] {desired_rpy}")
            print(f"[RPY_CUR] {tcp_rpy}")
            print(f"[ORI_ERR] {ori_err0} | norm = {np.linalg.norm(ori_err0):.4f}")
            print(f"[F_ORI] {F_ori_0}")
            print(f"[C0 diag] {np.diag(C0)}")
            print(f"[Target torque] {target_torque}")
            print("------------------------------------")

            # ----------------------------------------------------
            # Torque servo command
            # ----------------------------------------------------
            try:
                ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation=2)
                sleep(0.005)

                if not ret.is_success():
                    print(f"move_servo_t failed: {ret}")

            except Exception as e:
                print(f"T-servo Failed ..! {e}")

            viewer.sync()

    if csv_logging and len(csv_log_data) > 0:
        stop_csv_logging_and_save(tag="partial")

    print("[EXIT] finished")


if __name__ == "__main__":
    main()
