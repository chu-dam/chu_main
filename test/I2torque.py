import json
import numpy as np
import mujoco

# ==========================================
# 1. 환경 설정 및 데이터 입력
# ==========================================
MODEL_PATH = "/home/chu/chu_main/rb5/scene_rb5.xml"
JSON_FILES = {
    10: "/home/chu/chu_main/dynamic_friction_10_iters.json",
    30: "/home/chu/chu_main/dynamic_friction_30_iters.json",
    50: "/home/chu/chu_main/dynamic_friction_50_iters.json"
}

# 500개 샘플링을 통해 얻은 통계적 평균 전류 (A)[cite: 4]
I_static = np.array([0.1838, 2.5325, 2.6314, 0.2327, 0.0892, 0.0684])

# ==========================================
# 2. MuJoCo 이론적 중력 계산
# ==========================================
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
# 검증 자세: [90, 0, -90, 0, -90, 0]
d.qpos[:] = np.deg2rad([90.0, 0.0, -90.0, 0.0, -90.0, 0.0])
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)
G_theory = np.zeros(m.nv)
mujoco.mj_rne(m, d, 0, G_theory) # G_theory에 Nm 단위 중력 저장

# 실험적 토크 상수 (Nm/A) 산출
# J0은 중력 방향이 수직이라 G=0이므로, J1/J2의 평균값을 빌려 쓰거나 별도 처리
K_total = np.zeros(6)
for i in range(1, 6): # J1~J5 계산
    K_total[i] = abs(G_theory[i] / I_static[i])
K_total[0] = (K_total[1] + K_total[2]) / 2.0 # J0은 유사 모터인 J1, J2 값 원용

# ==========================================
# 3. 데이터 로드 및 선형 회귀 (Regression)
# ==========================================
final_params = {}

for j in range(6):
    speeds = []
    frictions = []
    
    for s, file_path in JSON_FILES.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
            fric_val = data[f"joint_{j}"]["average_friction"]
            
            # [중요] Joint 1의 50 deg/s 이상치 제거 로직[cite: 4]
            if j == 1 and s == 50: continue 
                
            speeds.append(s)
            frictions.append(fric_val)
    
    # 최소자승법 선형 회귀 (I = Ic + Iv * v)
    A = np.vstack([np.ones(len(speeds)), speeds]).T
    Ic_A, Iv_A = np.linalg.lstsq(A, frictions, rcond=None)[0]
    
    # Nm 단위로 변환
    tc_Nm = Ic_A * K_total[j]
    tv_Nm = Iv_A * K_total[j]
    
    final_params[j] = {"Tc": tc_Nm, "Tv": tv_Nm}

# ==========================================
# 4. 최종 결과 출력
# ==========================================
print("\n" + "="*50)
print("계산 완료! 제어기에 입력할 최종 마찰 보상 파라미터 (Nm)")
print("="*50)
for j, p in final_params.items():
    print(f"Joint {j}:")
    print(f"  - 쿨롱 마찰 (Tc): {p['Tc']:8.4f} Nm")
    print(f"  - 점성 마찰 (Tv): {p['Tv']:8.4f} Nm/(deg/s)")
print("="*50)