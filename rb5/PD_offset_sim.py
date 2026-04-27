from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np
import rbpodo as rb


# =========================================================
# Robot connection
# =========================================================
ROBOT_ADDRESS = "169.254.186.20"

try:
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = rb.CobotData(ROBOT_ADDRESS)

    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)

    # move_servo_t timing parameters
    t1, t2 = 0.01, 0.05

except Exception as e:
    print(f"로봇 연결 실패! {e}")
    raise SystemExit


# =========================================================
# Utility functions
# =========================================================
def rpy_to_rotmat(roll, pitch, yaw):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

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
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])

    return Rz @ Ry @ Rx


def rb_get_tcp_pose(current_state):
    if hasattr(current_state.sdata, "tcp"):
        tcp_info = np.array(current_state.sdata.tcp, dtype=np.float64)
    elif hasattr(current_state.sdata, "tcp_pos"):
        tcp_info = np.array(current_state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(current_state.sdata, "cur_pos"):
        tcp_info = np.array(current_state.sdata.cur_pos, dtype=np.float64)
    else:
        raise RuntimeError("TCP pose field를 찾지 못했습니다.")

    tcp_pos_m = tcp_info[0:3] * 0.001
    tcp_R = rpy_to_rotmat(*tcp_info[3:6])
    tcp_rpy = tcp_info[3:6]

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
    alpha_mm=1.0,
    C_mm=1.5,
    L_max_mm=6.0,
    stop_dist_m=0.001,
):
    """
    로그 함수 기반 virtual target offset 거리 계산.

    dist_m:
        현재 TCP와 실제 목표 TCP 사이 거리 [m]

    alpha_mm:
        로그항 계수 [mm]

    C_mm:
        목표 근처에서 마찰 극복을 위해 유지할 최소 가상 offset 거리 [mm]

    L_max_mm:
        가상 offset 최대 제한 [mm]

    stop_dist_m:
        이 거리 이내에서는 실제 목표에 도착한 것으로 보고 offset 제거 [m]

    사용 식:
        dist_mm = dist_m * 1000
        L_mm = alpha_mm * ln(dist_mm + 1) + C_mm
    """
    if dist_m <= stop_dist_m:
        return 0.0

    dist_mm = dist_m * 1000.0
    L_mm = alpha_mm * np.log(dist_mm + 1.0) + C_mm
    L_mm = np.clip(L_mm, 0.0, L_max_mm)

    return L_mm * 0.001


# =========================================================
# 1. Move to initial joint pose and save current TCP as target
# =========================================================
move_j_to_initial_pose(robot, rc)

state = request_valid_state(robot_data)
if state is None:
    print("[ERROR] move_j 이후 로봇 상태를 읽지 못했습니다.")
    raise SystemExit

init_jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)
desired_xpos_tcp, R_desired, desired_rpy = rb_get_tcp_pose(state)

print("\n[TARGET SET] move_j 이후 실제 TCP pose를 목표로 저장했습니다.")
print(f"[TARGET POS m]  {desired_xpos_tcp}")
print(f"[TARGET RPY deg] {desired_rpy}")
print(f"[CURRENT JOINT deg] {init_jpos}")


# =========================================================
# 2. MuJoCo model and Jacobian buffers
# =========================================================
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

d.qpos[:] = np.deg2rad(init_jpos)
mujoco.mj_forward(m, d)

tcp_site_id = m.site("tcp").id

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)


# =========================================================
# 3. PD gains
# =========================================================
K_a_mat = np.diag([200.0, 400.0, 300.0])
zeta_a_mat = np.diag([2.0, 4.0, 3.0])

K_o = 10.0
zeta_o = 4.0

max_torque = 50.0
tau_rate_limit = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 10.0])


# =========================================================
# 4. Virtual target parameters
# =========================================================
USE_VIRTUAL_TARGET = True

LOG_ALPHA_MM = 1.0       # 로그항 계수 [mm]
LOG_C_MM = 1.5           # 최소 가상 offset [mm]
LOG_L_MAX_MM = 6.0       # 최대 가상 offset [mm]
LOG_STOP_DIST_M = 0.001  # 1 mm 이내에서는 offset 제거


# =========================================================
# 5. Runtime variables
# =========================================================
prev_target_torque = np.zeros(6)
prev_time = time()
prev_jpos = None

print("\n[START] PD + log virtual target 제어 시작")
print("[INFO] 목표 TCP는 move_j 이후 저장된 TCP pose입니다.")
print("[INFO] 외부 입력토크 = 위치 PD(virtual target 기준) + 자세 PD")
print("[INFO] 로봇 내부 compensation=3 사용: 내부 중력보상 + 내부 마찰보상은 유지됩니다.")
print(f"[INFO] USE_VIRTUAL_TARGET = {USE_VIRTUAL_TARGET}")
print(f"[INFO] LOG_ALPHA_MM = {LOG_ALPHA_MM}")
print(f"[INFO] LOG_C_MM = {LOG_C_MM}")
print(f"[INFO] LOG_L_MAX_MM = {LOG_L_MAX_MM}")
print(f"[INFO] LOG_STOP_DIST_M = {LOG_STOP_DIST_M}")


# =========================================================
# 6. Control loop
# =========================================================
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        state = robot_data.request_data()
        if state is None:
            continue

        if state.sdata.op_stat_collision_occur or state.sdata.op_stat_sos_flag == 4:
            print("stop")
            break

        now = time()
        loop_dt = now - prev_time
        prev_time = now

        if loop_dt <= 0.0:
            continue

        jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)
        real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)

        jvel = np.zeros(6)
        if prev_jpos is not None:
            jvel = (jpos - prev_jpos) / loop_dt

        prev_jpos = jpos.copy()

        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = np.deg2rad(jvel)

        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        jacp0 = jacp[:, 0:6]
        jacr0 = jacr[:, 0:6]

        # ---------------------------------------------------------
        # Actual target error
        # ---------------------------------------------------------
        to_goal = desired_xpos_tcp - real_tcp_pos
        goal_dist = np.linalg.norm(to_goal)

        if goal_dist > 1e-9:
            u_goal = to_goal / goal_dist
        else:
            u_goal = np.zeros(3)

        goal_err = real_tcp_pos - desired_xpos_tcp
        goal_err_norm = np.linalg.norm(goal_err)

        # ---------------------------------------------------------
        # Log virtual target
        # ---------------------------------------------------------
        if USE_VIRTUAL_TARGET:
            L_virtual = virtual_offset_log_mm(
                goal_dist,
                alpha_mm=LOG_ALPHA_MM,
                C_mm=LOG_C_MM,
                L_max_mm=LOG_L_MAX_MM,
                stop_dist_m=LOG_STOP_DIST_M,
            )
            virtual_xpos_tcp = desired_xpos_tcp + u_goal * L_virtual
        else:
            L_virtual = 0.0
            virtual_xpos_tcp = desired_xpos_tcp.copy()

        # ---------------------------------------------------------
        # Position PD
        # ---------------------------------------------------------
        # 현재 코드의 부호 구조는 raw_err = current - target, torque = -J^T F 입니다.
        # 따라서 virtual target을 target 자리에 넣으면 됩니다.
        raw_err = real_tcp_pos - virtual_xpos_tcp
        ctrl_err_norm = np.linalg.norm(raw_err)

        xpos_dot0 = jacp0 @ d.qvel[0:6]

        sqrt_Ka_mat = np.diag(np.sqrt(np.diag(K_a_mat)))

        F_p = K_a_mat @ raw_err
        F_d = zeta_a_mat @ sqrt_Ka_mat @ xpos_dot0

        force0 = F_p + F_d

        # ---------------------------------------------------------
        # Orientation PD
        # ---------------------------------------------------------
        ori_err0 = (
            np.cross(R_desired[:, 0], real_tcp_R[:, 0]) +
            np.cross(R_desired[:, 1], real_tcp_R[:, 1]) +
            np.cross(R_desired[:, 2], real_tcp_R[:, 2])
        )

        w0 = jacr0 @ d.qvel[0:6]

        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # ---------------------------------------------------------
        # Task force/moment -> joint torque
        # ---------------------------------------------------------
        torque0 = - (jacp0.T @ force0) - (jacr0.T @ F_ori_0)
        target_torque_raw = np.clip(torque0, -max_torque, max_torque)

        # Torque rate limit
        d_tau_max = tau_rate_limit * loop_dt
        d_tau = target_torque_raw - prev_target_torque
        d_tau = np.clip(d_tau, -d_tau_max, d_tau_max)

        target_torque = prev_target_torque + d_tau
        prev_target_torque = target_torque.copy()

        d.ctrl[0:6] = target_torque

        if np.any(np.abs(jvel) > 70):
            print("over V")
            target_torque[:] = 0.0
            prev_target_torque[:] = 0.0

        try:
            robot.move_servo_t(rc, target_torque, t1, t2, compensation=3)
            sleep(0.005)
        except Exception as e:
            print(f"제어 명령 전송 실패: {e}")

        # ---------------------------------------------------------
        # Logs
        # ---------------------------------------------------------
        R_err = R_desired.T @ real_tcp_R

        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        ori_err_deg = np.rad2deg(np.arccos(cos_angle))

        rpy_err = real_tcp_rpy - desired_rpy
        rpy_err = (rpy_err + 180.0) % 360.0 - 180.0

        print(f"[GOAL ERR NORM] {goal_err_norm:.4f} | [CTRL ERR NORM] {ctrl_err_norm:.4f} | [VEL] {np.linalg.norm(xpos_dot0):.4f}")
        print(f"[GOAL ERR] {goal_err}")
        print(f"[VIRTUAL L] {L_virtual * 1000.0:.3f} mm")
        print(f"[VIRTUAL TARGET] {virtual_xpos_tcp}")
        print(f"[CTRL ERR] {raw_err}")
        print(f"[TARGET POS] {desired_xpos_tcp}")
        print(f"[REAL POS] {real_tcp_pos}")
        print(f"[F_P] {F_p} | [F_D] {F_d}")
        print(f"[TARGET RPY] {desired_rpy}")
        print(f"[REAL RPY] {real_tcp_rpy}")
        print(f"[RPY ERR DEG] {rpy_err}")
        print(f"[ORI ERR] {ori_err0}")
        print(f"[ORI ERR DEG] {ori_err_deg:.3f}")
        print(f"[TAR TOR RAW] {target_torque_raw}")
        print(f"[TAR TOR] {target_torque}")
        print("--------------------------------------")

        viewer.sync()
