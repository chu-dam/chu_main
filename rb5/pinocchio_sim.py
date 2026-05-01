from time import time
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

# =========================================================
# 1. 경로 설정 및 모델 초기화
# =========================================================
model_path = "/home/chu/chu_main/rb5/scene_rb5.xml"
urdf_path = "/home/chu/chu_main/rb5/rb5_850e.urdf"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

# 초기 자세
q0 = np.array([0.0, -0.5, 2.0, 0.0, 0.0, 0.0])
d.qpos[:] = q0
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)

print("[INFO] 피노키오 마찰 관측기 전용 시뮬레이션을 시작합니다.")
print("[INFO] 로봇이 가볍게 앞뒤로 움직이며 마찰력을 추정합니다.")

# =========================================================
# 2. 기본 제어기 및 관측기 변수 초기화
# =========================================================
# 로봇을 부드럽게 움직이기 위한 조인트 공간 PD 게인
Kp = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
Kd = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
G = np.zeros(m.nv)

# [핵심] 피노키오 마찰 관측기용 변수
K_obs = np.array([30.0, 30.0, 30.0, 10.0, 10.0, 10.0]) # 관측기 반응 속도
p0 = None
integral_term = np.zeros(6)
r_est = np.zeros(6)
tau_cmd_prev = np.zeros(6)

step_count = 0

with mujoco.viewer.launch_passive(m, d) as viewer:
    
    dt = m.opt.timestep 

    while viewer.is_running():
        t = d.time

        # ---------------------------------------------------------
        # [Step 1] 로봇 움직임 생성 (제자리에서 사인파 형태로 부드럽게 스윙)
        # ---------------------------------------------------------
        # 마찰을 측정하려면 관절이 움직여야 하므로 목표 궤적을 줍니다.
        q_des = q0 + 0.2 * np.sin(2.0 * np.pi * 0.5 * t)
        dq_des = 0.2 * 2.0 * np.pi * 0.5 * np.cos(2.0 * np.pi * 0.5 * t)

        # 조인트 PD 제어 + 중력 보상
        mujoco.mj_rne(m, d, 0, G)
        tau_pd = Kp * (q_des - d.qpos[0:6]) + Kd * (dq_des - d.qvel[0:6])
        tau_cmd = tau_pd + G[0:6]

        # ---------------------------------------------------------
        # [Step 2] 피노키오 백그라운드 연산 및 외란(마찰) 추정
        # ---------------------------------------------------------
        pin.computeAllTerms(pin_model, pin_data, d.qpos[0:6], d.qvel[0:6])
        M_mat = pin_data.M
        C_mat = pin_data.C
        g_vec = pin_data.g

        # 현재 운동량(p) 계산
        p_current = M_mat @ d.qvel[0:6]
        if p0 is None:
            p0 = p_current.copy()

        # 베타 및 적분항 업데이트
        beta = g_vec - (C_mat.T @ d.qvel[0:6])
        integral_term += (tau_cmd_prev - beta + r_est) * dt
        
        # 외란(마찰력) 추정
        r_est = K_obs * (p_current - p0 - integral_term)
        
        # 예열: 시뮬레이션 시작 후 0.5초 동안은 튀는 값 무시
        if t < 0.5:
            r_est[:] = 0.0
            integral_term[:] = p_current - p0

        # ---------------------------------------------------------
        # [Step 3] 모터 명령 및 로그 출력
        # ---------------------------------------------------------
        # 이 코드에서는 마찰력을 지우지 않고 "관찰"만 하므로 제어에는 더하지 않습니다.
        d.ctrl[0:6] = np.clip(tau_cmd, -150.0, 150.0)
        tau_cmd_prev = d.ctrl[0:6].copy()

        # 터미널에 로그가 너무 빨리 올라가는 것을 막기 위해 100스텝(0.2초)마다 한 번씩 출력
        step_count += 1
        if step_count % 100 == 0:
            print(f"[TIME] {t:.2f}s | [FRIC EST] {np.round(r_est, 4)}")

        mujoco.mj_step(m, d)
        viewer.sync()