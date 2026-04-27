from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import SystemVariable, CobotData
import rbpodo as rb


# =========================================================
# Robot connection
# =========================================================
try:
    ROBOT_ADDRESS = "169.254.186.20"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()

    robot_data = CobotData(ROBOT_ADDRESS)
    state = robot_data.request_data()

    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)

    t1 = 0.01  # 이동시간
    t2 = 0.05  # 유지시간

except Exception as e:
    print(f"No Robot Connection ..! {e}")
    raise SystemExit


# =========================================================
# Utility functions
# =========================================================
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
        [0,    0, 1],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ])

    Rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr],
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

    return np.array(jpos, dtype=np.float64), np.array(jvel, dtype=np.float64)


def rb_get_tcp_pose(current_state):
    """
    RB 컨트롤러에서 실제 TCP pose를 읽음.
    위치 단위: m
    자세 단위: deg, rotation matrix
    """
    if hasattr(current_state.sdata, "tcp"):
        tcp_info = np.array(current_state.sdata.tcp, dtype=np.float64)
    elif hasattr(current_state.sdata, "tcp_pos"):
        tcp_info = np.array(current_state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(current_state.sdata, "cur_pos"):
        tcp_info = np.array(current_state.sdata.cur_pos, dtype=np.float64)
    else:
        raise RuntimeError("TCP pose field를 찾지 못했습니다.")

    tcp_pos_m = tcp_info[0:3] * 0.001
    tcp_rpy = tcp_info[3:6]
    tcp_R = rpy_to_rotmat(*tcp_rpy)

    return tcp_pos_m, tcp_R, tcp_rpy


def request_valid_state(robot_data, retry=30, wait_sec=0.05):
    for _ in range(retry):
        state = robot_data.request_data()
        if state is not None:
            return state
        sleep(wait_sec)
    return None


def wait_move_finished(robot, rc, start_timeout=1.0):
    try:
        ret = robot.wait_for_move_started(rc, start_timeout)

        if ret.type() == rb.ReturnType.Success:
            robot.wait_for_move_finished(rc)
        else:
            print("[WARN] move_j 시작 확인 timeout. 3초 대기 후 진행합니다.")
            sleep(3.0)

        rc.error().throw_if_not_empty()

    except Exception as e:
        print(f"[WARN] move_j 대기 중 예외 발생: {e}")
        print("[WARN] 3초 대기 후 진행합니다.")
        sleep(3.0)


def move_j_to_initial_pose(robot, rc):
    init_joint_deg = np.array([90.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float64)

    movej_speed = 60.0
    movej_acc = 80.0

    print("\n[INIT] move_j로 초기 자세 이동 시작")
    print(f"[INIT] target joint deg = {init_joint_deg}")

    if hasattr(robot, "flush"):
        robot.flush(rc)

    robot.move_j(rc, init_joint_deg, movej_speed, movej_acc)
    wait_move_finished(robot, rc)

    sleep(0.3)
    print("[INIT] move_j 완료")


def virtual_offset_log_mm(
    dist_m,
    alpha_mm=0.7,
    C_mm=6.0,
    L_max_mm=12.0,
    stop_dist_m=0.005,
):
    """
    로그 함수 기반 virtual target offset 거리 계산.

    dist_m:
        현재 TCP와 실제 목표 TCP 사이 거리 [m]

    alpha_mm:
        로그항 계수 [mm]

    C_mm:
        복귀력 보강용 최소 가상 offset [mm]

    L_max_mm:
        가상 offset 최대 제한 [mm]

    stop_dist_m:
        이 거리 이내에서는 실제 목표에 도착한 것으로 보고 offset 제거 [m]
    """
    if dist_m <= stop_dist_m:
        return 0.0

    dist_mm = dist_m * 1000.0
    L_mm = alpha_mm * np.log(dist_mm + 1.0) + C_mm
    L_mm = np.clip(L_mm, 0.0, L_max_mm)

    return L_mm * 0.001


# =========================================================
# Move to initial pose and save target TCP
# =========================================================
move_j_to_initial_pose(robot, rc)

state = request_valid_state(robot_data)
if state is None:
    print("[ERROR] move_j 이후 로봇 상태를 읽지 못했습니다.")
    raise SystemExit

init_jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)
desired_xpos_tcp, R_desired, desired_rpy = rb_get_tcp_pose(state)


# =========================================================
# MuJoCo model
# =========================================================
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


# =========================================================
# Initial robot state
# =========================================================
try:
    state = request_valid_state(robot_data)

    if state is None:
        raise RuntimeError("초기 robot state를 읽지 못했습니다.")

    jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)

    jvel = np.zeros(6, dtype=np.float64)

    d.qpos[:] = np.deg2rad(jpos)
    d.qvel[:] = np.deg2rad(jvel)
    mujoco.mj_forward(m, d)

    print(f"Initial jpos : {jpos}")
    print(f"[TARGET POS FROM REAL TCP] {desired_xpos_tcp}")
    print(f"[TARGET RPY FROM REAL TCP] {desired_rpy}")

    sleep(1.0)

except Exception as e:
    print(f"Can't get robot init data ..! {e}")
    raise SystemExit


# =========================================================
# Buffers
# =========================================================
M = np.zeros((m.nv, m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

tcp_site_id = m.site("tcp").id


# =========================================================
# Control gains
# =========================================================
# 손으로 쉽게 밀리게 하기 위한 낮은 stiffness
K_a = 50.0
zeta_a = 0.4

K_o = 5.0
zeta_o = 1.5


# =========================================================
# Dynamic compensation scale
# =========================================================
M_SCALE = 0.3
C_SCALE = 0.0


# =========================================================
# Virtual return target parameters
# =========================================================
USE_VIRTUAL_RETURN = True

LOG_ALPHA_MM = 0.7
LOG_C_MM = 12.0
LOG_L_MAX_MM = 12.0
LOG_STOP_DIST_M = 0.005  # 5 mm 이내에서는 virtual target 제거

# 목표에서 멀어지는 중에는 virtual target을 끄기 위한 threshold
MOVING_AWAY_DOT_THRESH = 1e-5


# =========================================================
# Runtime variables
# =========================================================
prev_time = time()
hz_window = []
prev_jpos = None

print("\n[START] PD + inertia compensation + real TCP feedback + virtual return target")
print(f"[INFO] K_a = {K_a}, zeta_a = {zeta_a}")
print(f"[INFO] K_o = {K_o}, zeta_o = {zeta_o}")
print(f"[INFO] M_SCALE = {M_SCALE}")
print(f"[INFO] C_SCALE = {C_SCALE}")
print("[INFO] TCP position/orientation error = REAL ROBOT TCP 기준")
print("[INFO] Jacobian / Mass matrix = MuJoCo model 기준")
print("[INFO] servo_t compensation=3 사용: 내부 자중보상 + 30% 마찰보상 유지")
print("[INFO] 직접 입력토크 = PD torque + M(q)qdd_cmd + Coriolis/Centrifugal option")
print("[INFO] 중력항 G는 직접 더하지 않음")
print(f"[INFO] USE_VIRTUAL_RETURN = {USE_VIRTUAL_RETURN}")
print(f"[INFO] LOG_ALPHA_MM = {LOG_ALPHA_MM}")
print(f"[INFO] LOG_C_MM = {LOG_C_MM}")
print(f"[INFO] LOG_L_MAX_MM = {LOG_L_MAX_MM}")
print(f"[INFO] LOG_STOP_DIST_M = {LOG_STOP_DIST_M}")


# =========================================================
# Control loop
# =========================================================
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
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

        if loop_dt <= 0.0:
            continue

        loop_dt = max(loop_dt, 0.001)

        # ---------------------------------------------------------
        # Real robot joint state update
        # ---------------------------------------------------------
        try:
            jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)

            jvel = np.zeros(6, dtype=np.float64)
            if prev_jpos is not None:
                jvel = (jpos - prev_jpos) / loop_dt

            prev_jpos = jpos.copy()

            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)

        except Exception as e:
            print(f"real joint data update failed..! {e}")
            continue

        # ---------------------------------------------------------
        # Real robot TCP pose update
        # ---------------------------------------------------------
        try:
            real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)
        except Exception as e:
            print(f"real TCP data update failed..! {e}")
            continue

        # ---------------------------------------------------------
        # Forward kinematics / dynamics update in MuJoCo
        # ---------------------------------------------------------
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, M, d.qM)

        mujoco_tcp_pos = d.site("tcp").xpos.copy()

        # ---------------------------------------------------------
        # Coriolis / centrifugal term from MuJoCo bias force
        # qfrc_bias contains gravity + Coriolis/centrifugal.
        # Therefore, subtract static bias at qvel=0 to remove gravity.
        # ---------------------------------------------------------
        bias_with_vel = d.qfrc_bias.copy()

        qvel_backup = d.qvel.copy()

        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        bias_static = d.qfrc_bias.copy()

        d.qvel[:] = qvel_backup
        mujoco.mj_forward(m, d)

        tau_coriolis = C_SCALE * (bias_with_vel[0:6] - bias_static[0:6])

        # ---------------------------------------------------------
        # Jacobian from MuJoCo
        # ---------------------------------------------------------
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        jacp0 = jacp[:, 0:6].copy()
        jacr0 = jacr[:, 0:6].copy()

        # ---------------------------------------------------------
        # Position error from REAL TCP + virtual return target
        # ---------------------------------------------------------
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        to_goal = desired_xpos_tcp - real_tcp_pos
        goal_dist = np.linalg.norm(to_goal)

        if goal_dist > 1e-9:
            u_goal = to_goal / goal_dist
        else:
            u_goal = np.zeros(3)

        # e = current - target
        goal_err = real_tcp_pos - desired_xpos_tcp
        goal_err_norm = np.linalg.norm(goal_err)

        # 현재 TCP가 목표에서 더 멀어지는 방향으로 움직이는지 확인
        # dot(e, xdot) > 0 이면 목표에서 멀어지는 중
        moving_away = np.dot(goal_err, xpos_dot0) > MOVING_AWAY_DOT_THRESH

        if USE_VIRTUAL_RETURN and not moving_away:
            L_virtual = virtual_offset_log_mm(
                goal_dist,
                alpha_mm=LOG_ALPHA_MM,
                C_mm=LOG_C_MM,
                L_max_mm=LOG_L_MAX_MM,
                stop_dist_m=LOG_STOP_DIST_M,
            )
        else:
            L_virtual = 0.0

        virtual_xpos_tcp = desired_xpos_tcp + u_goal * L_virtual

        # 제어는 virtual target 기준
        xpos_cur = real_tcp_pos.copy()
        xpos_err0 = real_tcp_pos - virtual_xpos_tcp

        force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

        # ---------------------------------------------------------
        # Orientation error from REAL TCP
        # ---------------------------------------------------------
        R_current = real_tcp_R.copy()

        ori_err0 = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        )

        # Angular velocity from MuJoCo Jacobian + real joint velocity
        w0 = jacr0 @ d.qvel[0:6]

        # Orientation spring-damper command
        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # ---------------------------------------------------------
        # Original PD torque
        # ---------------------------------------------------------
        tau_pd = - jacp0.T @ force0 - jacr0.T @ F_ori_0

        # ---------------------------------------------------------
        # Inertia compensation
        # ---------------------------------------------------------
        # force0 = K*xerr + D*xdot
        # 복귀 방향 task-space acceleration-like command는 -force0
        xdd_cmd = -force0

        # Task-space acceleration command -> joint acceleration command
        qdd_cmd = np.linalg.pinv(jacp0) @ xdd_cmd

        # Joint-space inertia compensation
        tau_inertia = M_SCALE * (M[0:6, 0:6] @ qdd_cmd)

        # ---------------------------------------------------------
        # Final torque
        # ---------------------------------------------------------
        torque0 = tau_pd + tau_inertia + tau_coriolis

        max_torque = 50.0
        target_torque = np.clip(torque0, -max_torque, max_torque)

        d.ctrl[0:6] = target_torque

        # ---------------------------------------------------------
        # Joint velocity safety
        # ---------------------------------------------------------
        if np.any(np.abs(jvel) > 70):
            idx = np.where(np.abs(jvel) > 70)[0]

            if len(idx) > 0:
                target_torque[idx] = 0.0
                d.ctrl[idx] = 0.0

                print(torque0)
                print(f"Joint Velocity is too fast ...! Joint{list(idx)} | Jvel : {jvel[idx]}")

        # ---------------------------------------------------------
        # Hz measurement
        # ---------------------------------------------------------
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)

            if len(hz_window) > 30:
                hz_window.pop(0)

        # ---------------------------------------------------------
        # Torque servo command
        # ---------------------------------------------------------
        try:
            ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation=3)
            sleep(0.005)

            if not ret.is_success():
                print("move_servo_t 실패 ", ret)

        except Exception as e:
            print(f"T-servo Failed ..! {e}")

        # ---------------------------------------------------------
        # Logs
        # ---------------------------------------------------------
        vel_norm = np.linalg.norm(xpos_dot0)

        R_err = R_desired.T @ R_current
        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        ori_err_deg = np.rad2deg(np.arccos(cos_angle))

        rpy_err = real_tcp_rpy - desired_rpy
        rpy_err = (rpy_err + 180.0) % 360.0 - 180.0

        print(f"[GOAL ERR NORM] {goal_err_norm:.4f} | [CTRL ERR NORM] {np.linalg.norm(xpos_err0):.4f} | [VEL] {vel_norm:.4f} | [HZ] {np.mean(hz_window):.2f}")
        print(f"[GOAL ERR REAL] {goal_err}")
        print(f"[VIRTUAL L] {L_virtual * 1000.0:.3f} mm")
        print(f"[MOVING AWAY] {moving_away}")
        print(f"[VIRTUAL TARGET] {virtual_xpos_tcp}")
        print(f"[REAL TCP POS] {real_tcp_pos}")
        print(f"[MUJOCO TCP POS] {mujoco_tcp_pos}")
        print(f"[TCP POS DIFF real-mujoco] {real_tcp_pos - mujoco_tcp_pos}")
        print(f"[F_POS] {force0}")
        print(f"[TARGET RPY] {desired_rpy}")
        print(f"[REAL RPY] {real_tcp_rpy}")
        print(f"[RPY ERR DEG] {rpy_err}")
        print(f"[ORI ERR] {ori_err0}")
        print(f"[ORI ERR DEG] {ori_err_deg:.3f}")
        print(f"[TAU_PD] {tau_pd}")
        print(f"[TAU_INERTIA] {tau_inertia}")
        print(f"[TAU_CORIOLIS] {tau_coriolis}")
        print(f"[TAR TOR] {target_torque}")
        print("--------------------------------------")

        viewer.sync()