import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. 파라미터 설정
# =========================================================
# 실험으로 도출된 6개 관절의 마찰 계수 (Nm 단위)
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396]) # C_cf
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388]) # C_vf

# tanh 함수의 기울기를 결정하는 계수 (교수님 자료의 alpha)
friction_curve_coef = 0.8 # 8 * 1e-1 반영

# 속도 범위 설정 (-50 to 50 [deg/s])
v = np.linspace(-50, 50, 1000)

# =========================================================
# 2. 그래프 생성 및 스타일 설정
# =========================================================
# 가로 길이를 넉넉히 하여 Joint 3, 6 잘림 방지
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(r'RB5 Joint Friction Model Analysis: $\tau_{fric} = C_{cf} \tanh(\alpha \dot{q}) + C_{vf} \dot{q}$', 
             fontsize=20, fontweight='bold', y=0.96)

for i in range(6):
    ax = axes[i//3, i%3]
    
    # 각 성분별 계산
    # 1. Coulomb friction: 정지 상태 부근에서 지배적
    coulomb_part = Cfc[i] * np.tanh(friction_curve_coef * v)
    # 2. Viscous friction: 속도에 비례하여 선형 증가
    viscous_part = Vfc[i] * v
    # 3. Total friction: 두 성분의 합
    total_friction = coulomb_part + viscous_part
    
    # --- 그래프 플로팅 (선 스타일 및 색상 구분) ---
    # 전체 마찰력: 굵은 파란색 실선
    ax.plot(v, total_friction, color='blue', linestyle='-', linewidth=2.5, label='Total Friction')
    # 쿨롱 성분: 빨간색 점선
    ax.plot(v, coulomb_part, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Coulomb Component')
    # 점성 성분: 초록색 점선
    ax.plot(v, viscous_part, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Viscous Component')
    
    # --- 교수님 자료 스타일 축 라벨 설정 ---
    ax.set_title(f'Joint {i+1}', fontsize=15, pad=10)
    ax.set_xlabel(r'Joint velocity $\dot{q}$ [deg/s]', fontsize=12, labelpad=8)
    ax.set_ylabel(r'Friction torque $\tau_{fric}$ [Nm]', fontsize=12, labelpad=8)
    
    # 그리드 및 레이아웃 디테일
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5) # x축 기준선
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5) # y축 기준선
    
    # 1번 관절 그래프에만 범례 표시
    if i == 0:
        ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)

# =========================================================
# 3. 여백 및 간격 수동 조정 (잘림 및 겹침 방지)
# =========================================================
# right=0.94로 설정하여 오른쪽 끝 차트가 잘리지 않도록 함
plt.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.1, hspace=0.35, wspace=0.3)

print(f"[SUCCESS] Friction Model Visualization (alpha={friction_curve_coef})")
plt.show()