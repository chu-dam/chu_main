import json
import time
import numpy as np
import mujoco
from copy import deepcopy

import rbpodo as rb
from rbpodo import Cobot, CobotData, SystemVariable

# =========================================================
# 1. 로봇 및 시뮬레이터 초기화
# =========================================================
ROBOT_ADDRESS = "169.254.186.20"
MODEL_PATH = "/home/chu/chu_main/rb5/scene_rb5.xml"
OUTPUT_JSON_PATH = "static_friction_results.json"

try:
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = CobotData(ROBOT_ADDRESS)
    
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    print("[INFO] 로봇 연결 성공")
except Exception as e:
    print(f"[ERROR] 로봇 연결 실패: {e}")
    raise SystemExit

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
G = np.zeros(m.nv, dtype=np.float64)

# =========================================================
# 2. 유틸리티 함수
# =========================================================
def get_joint_state():
    jpos, jvel = [], []
    for i in range(6):
        _, pos = robot.get_system_variable(rc, getattr(SystemVariable, f"SD_J{i}_ANG"))
        _, vel = robot.get_system_variable(rc, getattr(SystemVariable, f"SD_J{i}_VEL"))
        jpos.append(pos)
        jvel.append(vel)
    return np.array(jpos), np.array(jvel)

def get_mujoco_gravity(jpos_deg):
    """현재 관절 각도를 바탕으로 MuJoCo에서 중력 보상 토크를 계산합니다."""
    d.qpos[:] = np.deg2rad(jpos_deg)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_rne(m, d, 0, G)
    return G[0:6].copy()

def move_to_initial_pose():
    """안전한 측정을 위해 초기 자세로 이동합니다."""
    init_pose = np.array([90.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float64)
    print(f"\n[INIT] 초기 자세 {init_pose} 로 이동 중...")
    robot.move_j(rc, init_pose, 40.0, 60.0)
    robot.wait_for_move_finished(rc)
    
    # 질문자님의 엄격한 기준 반영: 완벽한 정지 상태 확보
    print("[INIT] 이동 완료. 잔류 진동 소산 및 시스템 안정화를 위해 5초 대기합니다...")
    time.sleep(5.0)

# =========================================================
# 3. 핵심 측정 로직
# =========================================================
def measure_breakaway(joint_idx, direction, max_torque, step_torque=0.05, vel_threshold=0.3):
    """
    특정 관절에 토크를 서서히 증가시켜 움직이기 시작하는 순간의 토크를 찾습니다.
    - direction: 1 (+방향) 또는 -1 (-방향)
    - step_torque: 루프당 증가시킬 토크 량 (Nm)
    - vel_threshold: 움직였다고 판단할 속도 임계값 (deg/s)
    """
    test_torque = 0.0
    dir_str = "+" if direction > 0 else "-"
    print(f" -> Joint {joint_idx} [{dir_str}] 방향 측정 시작...", end="", flush=True)

    t1, t2 = 0.01, 0.05
    
    while abs(test_torque) <= max_torque:
        jpos, jvel = get_joint_state()
        
        # 1. 움직임 감지 (속도가 임계값을 넘으면 Breakaway 발생)
        if abs(jvel[joint_idx]) > vel_threshold:
            print(f" 완료! (Torque: {test_torque:.3f} Nm, Vel: {jvel[joint_idx]:.3f})")
            
            # 측정 직후 로봇이 계속 날아가지 않도록 현재 위치에서 중력만 보상하며 정지
            tau_g = get_mujoco_gravity(jpos)
            for _ in range(10): 
                robot.move_servo_t(rc, tau_g, t1, t2, compensation=0)
                time.sleep(0.01)
            return test_torque

        # 2. 중력 계산 및 토크 인가
        tau_g = get_mujoco_gravity(jpos)
        test_torque += direction * step_torque
        
        cmd_torque = tau_g.copy()
        cmd_torque[joint_idx] += test_torque
        
        # compensation=0: 로봇 내부 보상 끄고 MuJoCo 중력(tau_g) + 테스트 토크만 인가
        robot.move_servo_t(rc, cmd_torque, t1, t2, compensation=0)
        time.sleep(0.02)
        
    print(f" [경고] 최대 토크 {max_torque}Nm 도달. 관절이 움직이지 않습니다.")
    return test_torque

# =========================================================
# 4. 메인 테스트 시나리오
# =========================================================
results = {}

# 관절별 최대 허용 토크 세팅 (안전을 위해 Base쪽은 크게, Wrist쪽은 작게 설정)
max_torques = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

for i in range(6):
    print(f"\n{'='*40}\n[TEST] Joint {i} 정지 마찰력 측정\n{'='*40}")
    
    # 1. 항상 동일한 조건에서 측정하기 위해 초기 자세 셋업
    move_to_initial_pose()
    
    # 2. + 방향 측정
    tau_plus = measure_breakaway(i, direction=1, max_torque=max_torques[i])
    time.sleep(1.0)
    
    # 3. - 방향 측정
    # (+ 방향 측정으로 각도가 미세하게 틀어졌으므로, 다시 초기 자세로 맞출지 여기서 판단 가능합니다. 
    # 통상적으로 정지 마찰은 각도 변화가 극히 적으므로 바로 이어서 진행해도 무방합니다.)
    tau_minus = measure_breakaway(i, direction=-1, max_torque=max_torques[i])
    
    # 4. 순수 정지 마찰력(Coulomb) 및 편향 오차 계산
    friction = (tau_plus - tau_minus) / 2.0
    gravity_error = (tau_plus + tau_minus) / 2.0 # 양수/음수 비대칭성 (MuJoCo 모델 오차 등)
    
    results[f"joint_{i}"] = {
        "plus_breakaway": round(tau_plus, 4),
        "minus_breakaway": round(tau_minus, 4),
        "static_friction": round(friction, 4),
        "model_error_offset": round(gravity_error, 4)
    }

# =========================================================
# 5. 결과 JSON 저장
# =========================================================
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n[SUCCESS] 모든 측정이 완료되었습니다. 결과가 '{OUTPUT_JSON_PATH}'에 저장되었습니다.")