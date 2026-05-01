from time import time, sleep
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb

# =========================================================
# 1. 로봇 및 환경 설정
# =========================================================
ROBOT_ADDRESS = "169.254.186.20" 
MODEL_PATH = "/home/chu/chu_main/rb5/scene_rb5.xml"

try:
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = CobotData(ROBOT_ADDRESS)
    
    robot.set_operation_mode(rc, rb.OperationMode.Real) 
    robot.set_speed_bar(rc, 0.5)
    
    t1, t2 = 0.01, 0.05
    print(f"[INFO] RB5 시스템 연결 성공 ({ROBOT_ADDRESS})")
except Exception as e:
    print(f"[ERROR] 초기화 실패: {e}"); raise SystemExit

# =========================================================
# 2. 초기 자세 정렬 (Homing)
# =========================================================
home_joint_pose = [90.0, 0.0, -90.0, 0.0, -90.0, 0.0]
print(f"[Step 1] Home 자세 정렬 시작: {home_joint_pose}")
robot.move_j(rc, home_joint_pose, 30.0, 60.0)
robot.wait_for_move_finished(rc)
sleep(1.0)

# =========================================================
# 3. 제어 파라미터 및 유틸리티
# =========================================================
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

jpos_init = np.array(robot_data.request_data().sdata.jnt_ang)
d.qpos[:] = np.deg2rad(jpos_init)
mujoco.mj_forward(m, d)

target_home = deepcopy(d.site("tcp").xpos)
target_toggle = np.array([-0.11, -0.402, 0.35]) 

desired_xpos_tcp = deepcopy(target_home)
desired_rpy = np.array([90.0, 0.0, 0.0])
is_home_target = True

# --- 강의 자료 기반 제어 파라미터 ---
K_a = 100.0        # k: spring stiffness
K_o = 2.0
zeta_a = 1.0       # zeta: damping factor
zeta_o = 2.0

# Synergistic Damping 계수
varsigma = 0.5     # 자료의 varsigma (0 <= varsigma < 1)
alpha = 0.1        # LPF 필터 계수

# 마찰 보상 파라미터
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396])
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388])
friction_curve_coef = 0.5  # tanh 기울기
deadband_vel = 0.1

def mat_to_rpy(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x, y, z = np.arctan2(R[2,1], R[2,2]), np.arctan2(-R[2,0], sy), np.arctan2(R[1,0], R[0,0])
    else:
        x, y, z = np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0
    return np.rad2deg(np.array([x, y, z]))

def rpy_to_rotmat(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def key_callback(key):
    global desired_xpos_tcp, is_home_target
    if key == glfw.KEY_S:
        is_home_target = not is_home_target
        desired_xpos_tcp = target_home if is_home_target else target_toggle

# =========================================================
# 4. 고정 주기(200Hz) 메인 제어 루프
# =========================================================
control_hz = 200
dt = 1.0 / control_hz
next_loop_time = time()

prev_jpos = None
filtered_jvel = np.zeros(6)
M, G = np.zeros((m.nv, m.nv)), np.zeros(m.nv)
jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
C0 = np.zeros((6, 6))
loop_cnt = 0

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    print(f"\n[INFO] Synergistic Damping 제어 시작")
    
    while viewer.is_running():
        start_time = time()
        state = robot_data.request_data()
        
        # 1. 상태 업데이트 및 필터링
        try:
            jpos = np.array(state.sdata.jnt_ang)
            if prev_jpos is not None:
                raw_jvel = (jpos - prev_jpos) / dt 
                filtered_jvel = alpha * raw_jvel + (1 - alpha) * filtered_jvel
            prev_jpos = jpos
            d.qpos[:], d.qvel[:] = np.deg2rad(jpos), np.deg2rad(filtered_jvel)
        except Exception: continue

        # 2. 동역학 연산
        mujoco.mj_step(m, d)
        mujoco.mj_fullM(m, M, d.qM)
        qvel_bk = deepcopy(d.qvel)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G) # Gravity compensation g(q)
        d.qvel[:] = qvel_bk
        tcp_id = m.site("tcp").id
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_id)

        # 3. 강의 자료 기반 Synergistic Damping (C0) 계산
        # c_i = varsigma * sqrt(sum(|H_Mij|)) 수식 적용
        for i in range(6):
            row_abs_sum = np.sum(np.abs(M[i, 0:6]))
            C0[i, i] = varsigma * np.sqrt(row_abs_sum)

        # 4. 태스크 공간 제어력 계산
        x_curr = d.site("tcp").xpos
        x_err = x_curr - desired_xpos_tcp
        F_lin = (K_a * x_err) + (zeta_a * np.sqrt(K_a) * (jacp[:,:6] @ d.qvel[:6]))

        R_curr = d.site(tcp_id).xmat.reshape(3, 3)
        R_des = rpy_to_rotmat(*desired_rpy)
        ori_err = (np.cross(R_des[:, 0], R_curr[:, 0]) +
                   np.cross(R_des[:, 1], R_curr[:, 1]) +
                   np.cross(R_des[:, 2], R_curr[:, 2]))
        F_ori = (K_o * ori_err) + (zeta_o * np.sqrt(K_o) * (jacr[:,:6] @ d.qvel[:6]))

        # 5. 마찰 보상 계산
        Tf = np.zeros(6)
        for i in range(6):
            v_deg = filtered_jvel[i]
            if abs(v_deg) > deadband_vel:
                Tf[i] = (Cfc[i] * np.tanh(friction_curve_coef * v_deg) + Vfc[i] * v_deg)

        # 6. 최종 토크 합산 (강의 자료 제어 루프 구조 유지)[cite: 1]
        # u = -C0*q_dot - J^T(k*dp + zeta*sqrt(k)*p_dot) + g(q)[cite: 1]
        torque0 = (- 1 * C0 @ d.qvel[0:6] 
                   - 1 * jacp[:, 0:6].T @ F_lin 
                   + 1 * G[0:6] 
                   - 0 * jacr[:, 0:6].T @ F_ori 
                   + 1 * Tf[0:6])

        # 7. 토크 전송 및 모니터링
        d.ctrl[:6] = np.clip(torque0, -50.0, 50.0)
        if np.any(np.abs(filtered_jvel) > 70): d.ctrl[:6] = 0.0
        
        robot.move_servo_t(rc, d.ctrl[:6], t1, t2, compensation=0)

        loop_cnt += 1
        if loop_cnt % 20 == 0:
            print("\033[H\033[J") 
            print(f"{'--- RB5 Synergistic Control Monitoring ---':^50}")
            print(f"Pos  Error: {np.linalg.norm(x_err):.4f} m")
            print(f"C0 Diagonal: {np.diag(C0).round(2)}")
            print(f"Loop Freq : {1.0/(time()-start_time + 1e-9):.1f} Hz")
            print("=" * 50)

        next_loop_time += dt
        sleep_time = next_loop_time - time()
        if sleep_time > 0: sleep(sleep_time)
        else: next_loop_time = time() 

        viewer.sync()