import numpy as np
import matplotlib.pyplot as plt

# 1. 물리 및 제어 파라미터 설정
dt = 0.001
t_end = 5.0
time = np.arange(0, t_end, dt)

mass = 1.0
F_static = 1.5   # 정지 마찰력 (이 힘을 넘어야 움직임)
F_coulomb = 1.0  # 운동 마찰력
F_viscous = 0.1  # 점성 마찰 계수

Kp = 50.0
Kd = 10.0
target_x = 1.0

# 로그 가상 목표 파라미터
alpha = 0.05 
beta = 0.1
epsilon = 1e-4

def simulate(mode='standard'):
    curr_x, curr_v = 0.0, 0.0
    history_x = []
    
    for t in time:
        error = target_x - curr_x
        
        # 제어 입력 계산
        if mode == 'standard':
            u = Kp * error - Kd * curr_v
        else:
            # 성현님의 로그 가상 목표 로직
            d = alpha * np.log(1 + beta / (abs(error) + epsilon))
            virtual_error = error + np.sign(error) * d
            u = Kp * virtual_error - Kd * curr_v

        # 마찰력 모델 (Stiction 반영)
        friction = 0
        if abs(curr_v) < 1e-3:
            if abs(u) < F_static:
                friction = u # 정지 상태 유지
            else:
                friction = F_static * np.sign(u)
        else:
            friction = (F_coulomb + F_viscous * abs(curr_v)) * np.sign(curr_v)
        
        # 가속도 및 상태 업데이트
        accel = (u - friction) / mass
        curr_v += accel * dt
        curr_x += curr_v * dt
        history_x.append(curr_x)
        
    return history_x

# 실행
std_results = simulate(mode='standard')
log_results = simulate(mode='log_virtual')

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(time, [target_x]*len(time), 'r--', label='Target')
plt.plot(time, std_results, label='Standard PD (Stops early)')
plt.plot(time, log_results, label='Log-Virtual PD (Reaches target)')
plt.title("Comparison: Standard PD vs Log-based Virtual Target PD")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()
plt.grid(True)
plt.show()