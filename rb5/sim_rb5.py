from time import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np

def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    return Rz @ Ry @ Rx

def orientation_error(R_des, R_cur):
    e = 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0]) +
        np.cross(R_cur[:, 1], R_des[:, 1]) +
        np.cross(R_cur[:, 2], R_des[:, 2])
    )
    return e

# ---------------------------------------------------------
# 1. 최대 관성 행렬(H_M) 오프라인 계산 함수
# ---------------------------------------------------------
def calculate_H_M(m, d, num_samples=30000):
    print("\n[초기화] 최적 관절 댐핑을 위한 H_M 오프라인 연산을 시작합니다 (샘플 수: {})...".format(num_samples))
    nv = m.nv
    nq = m.nq
    H_M = np.zeros((nv, nv))
    M = np.zeros((nv, nv))

    qpos_backup = np.copy(d.qpos)
    jnt_range = m.jnt_range

    for _ in range(num_samples):
        q_rand = np.zeros(nq)
        for i in range(nq):
            low, high = jnt_range[i]
            if low == high == 0.0:
                low, high = -np.pi, np.pi
            q_rand[i] = np.random.uniform(low, high)

        d.qpos[:] = q_rand
        
        mujoco.mj_kinematics(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_fullM(m, M, d.qM) 

        H_M = np.maximum(H_M, np.abs(M))

    d.qpos[:] = qpos_backup
    mujoco.mj_kinematics(m, d)
    print("[완료] H_M 연산 완료!\n")
    return H_M

# ---------------------------------------------------------
# [설정] 목표 좌표 정의 (기본 및 토글 목표)
# ---------------------------------------------------------
target_1 = np.array([0.11, -0.502, 0.493])  # 기본 추종 목표 (시작 즉시 여기로 이동)
target_2 = np.array([-0.11, -0.302, 0.2])   # 토글 목표

target_rpy_tcp = np.array([90.0, 0.0, 0.0])
target_R_tcp = rpy_to_rotmat(*target_rpy_tcp)

# 주의: 사용 중인 모델의 절대 경로로 맞추어 주세요.
model_path = "/home/chu/chu_main/rb5/scene_rb5.xml"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# 초기 자세 설정
d.qpos[:] = [-0.0, -0.5, 2.0, 0.0, 0.0, 0.0]
d.qvel[:] = 0.0

mujoco.mj_forward(m, d)

# ---------------------------------------------------------
# 2. H_M 기반 시너지 관절 댐핑 행렬 (C0) 도출
# ---------------------------------------------------------
H_M_matrix = calculate_H_M(m, d, num_samples=30000)

zeta_joint = 0.7  # Joint space 댐핑 계수
c_diag = np.zeros(m.nv)

for i in range(m.nv):
    c_diag[i] = zeta_joint * np.sqrt(np.sum(H_M_matrix[i, :]))

C0_matrix = np.diag(c_diag)

print("=== 자동 계산된 관절 댐핑 행렬 (C0_matrix 대각 성분) ===")
print(np.round(c_diag, 3))
print("========================================================\n")

tcp_site_id = m.site("tcp").id

hold_xpos_tcp = d.site("tcp").xpos.copy()
hold_R_tcp = d.site(tcp_site_id).xmat.reshape(3, 3).copy()

print("[INIT TCP POS]", hold_xpos_tcp)
print("[TARGET 1 POS]", target_1)
print("[TARGET 2 POS]", target_2)
print("[TARGET TCP RPY]", target_rpy_tcp)
print("▶ 시뮬레이션 시작과 동시에 TARGET 1으로 자동 이동합니다.")
print("▶ 이동 중 's' 키를 누르면 TARGET 2와 TARGET 1 사이를 토글합니다.")

# ---------------------------------------------------------
# 3. 토글 상태 관리 및 키 콜백
# ---------------------------------------------------------
state = {
    "mode": 1  # 0(대기)가 아닌 1(Target 1)로 초기화하여 즉시 이동 시작
}

def key_callback(keycode):
    if keycode == ord('S') or keycode == ord('s'):
        if state["mode"] == 1:
            state["mode"] = 2
            print("\n[KEY] S pressed -> Target 2 활성화")
        else:
            state["mode"] = 1
            print("\n[KEY] S pressed -> Target 1 활성화 (기본 목표)")

G = np.zeros(m.nv, dtype=np.float64)
jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

# Task Space 제어 이득
K_pos = np.diag([10.0, 10.0, 10.0])
D_pos = np.diag([5.0, 5.0, 5.0])

K_ori = np.diag([3.0, 3.0, 3.0])
D_ori = np.diag([1.0, 1.0, 1.0])

max_total_torque = 150.0
max_ori_torque = 30.0

with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=key_callback
) as viewer:

    t0 = time()

    while viewer.is_running():
        t = time() - t0

        # 상태에 따른 목표 지점 실시간 할당
        if state["mode"] == 1:
            desired_xpos_tcp = target_1.copy()
            mode_str = "TARGET 1"
        elif state["mode"] == 2:
            desired_xpos_tcp = target_2.copy()
            mode_str = "TARGET 2"

        desired_R_tcp = target_R_tcp.copy()

        # 1. Gravity compensation torque G(q)
        qvel_backup = deepcopy(d.qvel)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G)
        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)
        gravity_torque = G[0:6]

        # 2. TCP Jacobian
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        Jp = jacp[:, 0:6]
        Jr = jacr[:, 0:6]

        # 3. Current TCP state
        current_xpos_tcp = d.site("tcp").xpos.copy()
        current_R_tcp = d.site(tcp_site_id).xmat.reshape(3, 3).copy()
        current_xvel_tcp = Jp @ d.qvel[0:6]
        current_w_tcp = Jr @ d.qvel[0:6]

        # 4. Position Virtual Spring-Damper
        xpos_err = desired_xpos_tcp - current_xpos_tcp
        xvel_err = -current_xvel_tcp
        F_vsd = K_pos @ xpos_err + D_pos @ xvel_err
        tau_pos = Jp.T @ F_vsd

        # 5. Orientation Virtual Spring-Damper
        ori_err = orientation_error(desired_R_tcp, current_R_tcp)
        w_err = -current_w_tcp
        M_vsd = K_ori @ ori_err + D_ori @ w_err
        tau_ori = Jr.T @ M_vsd
        tau_ori = np.clip(tau_ori, -max_ori_torque, max_ori_torque)

        # 6. 최적화된 C0 행렬을 이용한 Joint Damping
        tau_joint_damping = -(C0_matrix[0:6, 0:6] @ d.qvel[0:6])

        # 7. Total torque
        torque0 = gravity_torque + tau_pos + tau_ori + tau_joint_damping

        d.ctrl[0:6] = np.clip(torque0, -max_total_torque, max_total_torque)

        # 제어 상태 모니터링 (터미널 버벅임 방지를 위해 0.5초마다 출력)
        if int(time() * 10) % 5 == 0:
            print(f"[{mode_str}] POS ERR: {np.linalg.norm(xpos_err):.4f} | TCP X: {current_xpos_tcp[0]:.3f}, Y: {current_xpos_tcp[1]:.3f}, Z: {current_xpos_tcp[2]:.3f}")

        mujoco.mj_step(m, d)
        viewer.sync()