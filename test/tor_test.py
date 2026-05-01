import time
import numpy as np
import rbpodo as rb
from rbpodo import Cobot, CobotData, SystemVariable

# =========================================================
# 1. 초기화 및 이동
# =========================================================
ROBOT_ADDRESS = "169.254.186.20"
SAMPLE_COUNT = 500  # 측정할 샘플 개수 (500개 = 약 5초)

try:
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    print("[INFO] 로봇 연결 성공!")
except Exception as e:
    print(f"[ERROR] 연결 실패: {e}"); raise SystemExit

# 검증 자세 이동
target_pose = [90.0, 0.0, -90.0, 0.0, -90.0, 0.0]
print(f"\n[Step 1] 검증 자세로 이동 중...")
robot.move_j(rc, target_pose, 30.0, 40.0)
robot.wait_for_move_finished(rc)

print(" -> 이동 완료! 기계적 진동이 사라지도록 3초간 대기합니다.")
time.sleep(3.0)

# =========================================================
# 2. 통계적 데이터 채집 (Sampling)
# =========================================================
print(f"\n[Step 2] {SAMPLE_COUNT}개의 샘플 채집 시작...")
raw_data = [[] for _ in range(6)] # 6개 관절 데이터 저장용

for s in range(SAMPLE_COUNT):
    for i in range(6):
        var = getattr(SystemVariable, f"SD_J{i}_CUR")
        _, val = robot.get_system_variable(rc, var)
        raw_data[i].append(val)
    
    if (s + 1) % 50 == 0:
        print(f"  - 진행률: {s + 1}/{SAMPLE_COUNT}")
    time.sleep(0.01) # 100Hz 수준으로 채집

# =========================================================
# 3. 결과 분석 및 출력
# =========================================================
print("\n" + "="*60)
print(f"{'관절':^6} | {'평균 전류 (A)':^15} | {'표준편차 (Std)':^15}")
print("-" * 60)

final_avg_currents = []

for i in range(6):
    mean_val = np.mean(raw_data[i])
    std_val = np.std(raw_data[i])
    final_avg_currents.append(round(mean_val, 4))
    print(f" J{i}   | {mean_val:15.4f} | {std_val:15.4f}")

print("="*60)
print(f"\n[최종 결과] I_static = {final_avg_currents}")
print("이 리스트를 'I2torque.py'의 I_static 변수에 그대로 복사해서 넣으세요.")