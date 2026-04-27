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
# 2. MuJoCo model and Dynamics buffers
# =========================================================
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

d.qpos[:] = np.deg2rad(init_jpos)
mujoco.mj_forward(m, d)

tcp_site_id = m.site("tcp").id

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

M_dense = np.zeros((m.nv, m.nv), dtype=np.float64)
eye_6 = np.eye(6)
epsilon = 1e-5


# =========================================================
# 3. Operational Space Gains
# =========================================================
K_a_mat = np.diag([400.0, 400.0, 400.0])    # 병진 제어 대역폭: wn = 20 rad/s
zeta_a_mat = np.diag([15.0, 15.0, 15.0])    # 사용자 설정 감쇠값

K_o = 40.0                                # 회전 복원 토크 계수 (Lambda 우회)
zeta_o = 14.0                             # 사용자 설정 감쇠값

max_torque = 50.0
tau_rate_limit = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 10.0])


# =========================================================
# 4. Runtime variables
# =========================================================
prev_target_torque = np.zeros(6)
prev_time = time()
prev_jpos = None

print("\n[START] Task-Decoupled Hybrid Impedance Control 시작")
print("[INFO] 병진 운동: Lambda(q) 통과 (동역학 보상)")
print("[INFO] 회전 운동: Lambda(q) 우회 (순수 토크 맵핑)")
print("[INFO] 로봇 내부 compensation=3 사용: 내부 중력보상 + 내부 마찰보상은 유지됩니다.")


# =========================================================
# 5. Control loop
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

        # ---------------------------------------------------------
        # Kinematics & Mass Matrix Update
        # ---------------------------------------------------------
        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        mujoco.mj_fullM(m, M_dense, d.qM)

        jacp0 = jacp[:, 0:6]
        jacr0 = jacr[:, 0:6]

        # ---------------------------------------------------------
        # Operational Space Inertia Matrix (Lambda) Calculation
        # ---------------------------------------------------------
        M_inv = np.linalg.inv(M_dense[0:6, 0:6])
        J_6D = np.vstack((jacp0, jacr0))

        # Lambda = (J * M_inv * J^T + epsilon*I)^-1
        J_Minv_Jt = J_6D @ M_inv @ J_6D.T
        Lambda = np.linalg.inv(J_Minv_Jt + eye_6 * epsilon)

        # ---------------------------------------------------------
        # Position Error Dynamics
        # ---------------------------------------------------------
        raw_err = real_tcp_pos - desired_xpos_tcp
        err_norm = np.linalg.norm(raw_err)

        xpos_dot0 = jacp0 @ d.qvel[0:6]
        sqrt_Ka_mat = np.diag(np.sqrt(np.diag(K_a_mat)))

        F_p = K_a_mat @ raw_err
        F_d = zeta_a_mat @ sqrt_Ka_mat @ xpos_dot0

        force0 = F_p + F_d

        # ---------------------------------------------------------
        # Orientation Error Dynamics
        # ---------------------------------------------------------
        ori_err0 = (
            np.cross(R_desired[:, 0], real_tcp_R[:, 0]) +
            np.cross(R_desired[:, 1], real_tcp_R[:, 1]) +
            np.cross(R_desired[:, 2], real_tcp_R[:, 2])
        )

        w0 = jacr0 @ d.qvel[0:6]

        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # ---------------------------------------------------------
        # Task force/moment -> joint torque (Hybrid Decoupling)
        # ---------------------------------------------------------
        # 1. 병진 가속도 지령: Lambda를 통과시켜 Wrench(힘+커플링 모멘트)로 변환
        F_trans_6D = np.hstack((force0, np.zeros(3)))
        W_trans = Lambda @ F_trans_6D

        # 2. 회전 토크 지령: Lambda를 우회하여 직접적인 Wrench(순수 모멘트) 구성
        W_ori = np.hstack((np.zeros(3), F_ori_0))

        # 3. 최종 Wrench 합산 및 토크 변환
        W_task = W_trans + W_ori
        
        # raw_err 수식이 (real - desired) 이므로 - 부호를 유지하여 복원력을 생성합니다.
        torque0 = - (J_6D.T @ W_task)
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

        print(f"[ERR NORM] {err_norm:.4f} | [VEL] {np.linalg.norm(xpos_dot0):.4f}")
        print(f"[POS ERR] {raw_err}")
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