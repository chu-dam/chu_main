from time import time
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin

# ==============================================================================
# [파트 1] 들뜸 경로 생성기 & 피노키오 관찰 행렬 계산기
# ==============================================================================

def get_exciting_trajectory(t, num_joints=6):
    """
    푸리에 급수를 이용해 시간에 따른 각 관절의 목표 위치, 속도, 가속도를 생성합니다.
    """
    q_des = np.zeros(num_joints)
    dq_des = np.zeros(num_joints)
    ddq_des = np.zeros(num_joints)
    
    # 임의의 푸리에 파라미터 (실제로는 최적화 알고리즘으로 찾은 값을 넣습니다)
    base_freq = 0.5 * np.pi
    a = np.array([0.2, 0.15, 0.1, 0.05, 0.02])
    b = np.array([0.1, 0.2, 0.05, 0.08, 0.01])
    
    for i in range(num_joints):
        for k in range(1, 6): # 5개의 하모닉스
            wk = base_freq * k
            q_des[i]   +=  a[k-1] * np.sin(wk * t) + b[k-1] * np.cos(wk * t)
            dq_des[i]  +=  a[k-1] * wk * np.cos(wk * t) - b[k-1] * wk * np.sin(wk * t)
            ddq_des[i] += -a[k-1] * (wk**2) * np.sin(wk * t) - b[k-1] * (wk**2) * np.cos(wk * t)
            
    return q_des, dq_des, ddq_des

def get_full_regressor(model_pin, data_pin, q, dq, ddq):
    """강체 동역학 관찰 행렬과 마찰력 관찰 행렬을 합칩니다."""
    # 1. 강체 파라미터 행렬 (6 x 60)
    Y_rigid = pin.computeJointTorqueRegressor(model_pin, data_pin, q, dq, ddq)
    
    # 2. 마찰 파라미터 행렬 (6 x 12)
    nv = model_pin.nv
    Y_friction = np.zeros((nv, 2 * nv))
    for i in range(nv):
        Y_friction[i, 2*i] = np.sign(dq[i])  # 쿨롱 마찰 자리
        Y_friction[i, 2*i + 1] = dq[i]       # 점성 마찰 자리
        
    # 3. 전체 결합 (6 x 72)
    Y_total = np.hstack((Y_rigid, Y_friction))
    return Y_total


# ==============================================================================
# [파트 2] 환경 초기화 및 데이터 수집 준비
# ==============================================================================

# 1. MuJoCo 초기화
model_path = "/home/chu/chu_main/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# 2. Pinocchio 초기화
urdf_path = "/home/chu/chu_main/rb5/rb5_850e.urdf" # 실행 위치에 맞게 경로 수정 필요
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

# 초기 자세 설정
d.qpos[:] = [0.0, -0.5, 1.5, 0.0, 0.0, 0.0]
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)

# 상태 제어 변수
state = {"started": False, "data_collected": False}

def key_callback(keycode):
    if keycode == ord('S') or keycode == ord('s'):
        if not state["started"]:
            state["started"] = True
            print("\n[KEY] 'S' 입력됨 -> 들뜸 경로 추종 및 데이터 수집 시작!\n")

print("MuJoCo 뷰어 창에서 's' 키를 누르면 로봇이 춤을 추며 데이터를 수집합니다.")

# 조인트 제어를 위한 PD 게인 (로봇이 목표 궤적을 잘 따라가게 만듭니다)
Kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])
Kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
G = np.zeros(m.nv)

# 데이터를 쌓을 리스트
Y_list = []
tau_list = []

# ==============================================================================
# [파트 3] 시뮬레이션 루프 및 최소자승법 연산
# ==============================================================================

collection_time = 10.0 # 10초 동안 데이터 수집
start_t = 0.0

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    
    while viewer.is_running():
        
        # S키를 누르기 전에는 제자리에 멈춰있습니다.
        if not state["started"]:
            mujoco.mj_rne(m, d, 0, G)
            d.ctrl[0:6] = G[0:6] - Kd * d.qvel[0:6] # 중력 보상 + 감쇠 유지
            start_t = d.time
            
        # S키를 누르면 데이터 수집 모드 돌입
        elif state["started"] and not state["data_collected"]:
            t = d.time - start_t
            
            # 1. 들뜸 경로에서 목표 위치, 속도 받아오기
            q_des, dq_des, ddq_des = get_exciting_trajectory(t, m.nv)
            
            # 2. 로봇 제어 (조인트 공간 PD 제어 + 중력 보상)
            mujoco.mj_rne(m, d, 0, G)
            gravity_torque = G[0:6]
            
            tau_pd = Kp * (q_des - d.qpos[0:6]) + Kd * (dq_des - d.qvel[0:6])
            tau_total = tau_pd + gravity_torque
            d.ctrl[0:6] = np.clip(tau_total, -150.0, 150.0)
            
            # 3. ★ 핵심: 관찰 행렬(Y)과 실제 토크(tau) 수집 ★
            # 시뮬레이션이므로 d.qacc(실제 가속도)를 오차 없이 바로 쓸 수 있습니다.
            Y = get_full_regressor(pin_model, pin_data, d.qpos[0:6], d.qvel[0:6], d.qacc[0:6])
            
            Y_list.append(Y)
            tau_list.append(tau_total) # 모터가 실제로 낸 힘
            
            # 10초가 지나면 수집 종료 및 계산
            if t > collection_time:
                state["data_collected"] = True
                print("\n[완료] 10초간의 데이터 수집이 끝났습니다. 최소자승법 연산을 시작합니다...")
                
                # 리스트에 쌓인 행렬들을 하나의 거대한 표로 합치기
                Y_stacked = np.vstack(Y_list)
                tau_stacked = np.concatenate(tau_list)
                
                # 최소자승법 (Least Squares) 실행
                theta_est, _, _, _ = np.linalg.lstsq(Y_stacked, tau_stacked, rcond=None)
                
                print("\n================== 식별된 마찰 계수 ==================")
                for i in range(6):
                    # 60개의 강체 파라미터 뒤에 12개의 마찰 파라미터가 있습니다.
                    fc = theta_est[60 + 2*i]
                    fv = theta_est[60 + 2*i + 1]
                    print(f"Joint {i+1} - 쿨롱 마찰(Fc): {fc:7.4f}, 점성 마찰(Fv): {fv:7.4f}")
                print("======================================================")
                print("뷰어를 닫아도 됩니다.")

        mujoco.mj_step(m, d)
        viewer.sync()