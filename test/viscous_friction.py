import json
import time
import numpy as np
import mujoco

import rbpodo as rb
from rbpodo import Cobot, CobotData, SystemVariable

# =========================================================
# 1. 초기화 및 환경 설정
# =========================================================
ROBOT_ADDRESS = "169.254.186.20"
MODEL_PATH = "/home/chu/chu_main/rb5/scene_rb5.xml"
OUTPUT_JSON_PATH = "dynamic_friction_50_iters.json"

# 충돌을 방지하기 위한 나머지 관절들의 기본 안전 자세
BASE_POSE = np.array([90.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float64)

try:
    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = CobotData(ROBOT_ADDRESS)
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 1.0) # 테스트 속도를 정확히 맞추기 위해 1.0으로 설정
    print("[INFO] 로봇 연결 성공")
except Exception as e:
    print(f"[ERROR] 연결 실패: {e}")
    raise SystemExit

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
G = np.zeros(m.nv, dtype=np.float64)

def get_joint_state():
    jpos, jvel = [], []
    for i in range(6):
        _, pos = robot.get_system_variable(rc, getattr(SystemVariable, f"SD_J{i}_ANG"))
        _, vel = robot.get_system_variable(rc, getattr(SystemVariable, f"SD_J{i}_VEL"))
        jpos.append(pos)
        jvel.append(vel)
    return np.array(jpos), np.array(jvel)

# =========================================================
# 2. [핵심] 단일 스트로크 측정 함수 (가속/감속 구간 Trimming)
# =========================================================
def execute_stroke_and_measure(target_pose, joint_idx, speed, accel, stroke_deg):
    """목표 자세로 이동하며 완벽한 등속 구간에서만 토크를 수집합니다."""
    
    # 등속 구간 타이밍 계산 (물리 법칙 적용)
    t_accel = speed / accel
    d_accel_decel = (speed ** 2) / accel # 가속 및 감속에 소요되는 총 각도
    d_const = stroke_deg - d_accel_decel
    t_const = d_const / speed

    if t_const <= 0.5:
        raise ValueError(f"스트로크가 너무 짧아 등속 구간({t_const:.2f}s) 확보 불가. 속도를 낮추거나 각도를 늘리세요.")

    wait_before_measure = t_accel + 0.2
    measure_duration = t_const - 0.4

    # 이동 명령 (Non-blocking)
    robot.move_j(rc, target_pose, speed, accel)
    
    # 가속 구간 동안 대기
    time.sleep(wait_before_measure)

    # 데이터 수집 (순수 등속 구간)
    torque_log = []
    start_t = time.time()
    
    while (time.time() - start_t) < measure_duration:
        jpos, jvel = get_joint_state()
        
        # MuJoCo 중력 계산
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G)
        
        # 실제 전류(토크 대용) 읽기 - 에러 수정됨 (_CUR 사용)
        _, actual_torque = robot.get_system_variable(rc, getattr(SystemVariable, f"SD_J{joint_idx}_CUR"))
        
        # 순수 구동 토크 = 실제 전류 - 중력 보상 토크
        net_torque = actual_torque - G[joint_idx]
        torque_log.append(net_torque)
        time.sleep(0.01)

    # 로봇이 목적지에 도달하여 멈출 때까지 안전하게 대기
    robot.wait_for_move_finished(rc)
    time.sleep(0.5)

    return np.mean(torque_log)

# =========================================================
# 3. [핵심] 파라미터화된 조인트 양방향 측정 함수
# =========================================================
def measure_joint_dynamic_friction(joint_idx, start_angle, end_angle, speed, accel):
    """
    지정된 파라미터로 해당 관절을 start -> end, end -> start로 1회 왕복 측정합니다.
    주의: 호출 전 로봇은 이미 start_angle 위치(pose_A)에 있어야 합니다.
    """
    stroke_deg = abs(end_angle - start_angle)
    
    # 도착 자세(B)와 돌아올 자세(A) 배열 생성
    pose_A = BASE_POSE.copy()
    pose_A[joint_idx] = start_angle
    
    pose_B = BASE_POSE.copy()
    pose_B[joint_idx] = end_angle

    # Stroke 1: A -> B
    print(f"    [->] {start_angle}° to {end_angle}° 이동 중...", end="", flush=True)
    tau_1 = execute_stroke_and_measure(pose_B, joint_idx, speed, accel, stroke_deg)
    print(f" 완료 (평균: {tau_1:.3f})")

    # Stroke 2: B -> A
    print(f"    [<-] {end_angle}° to {start_angle}° 이동 중...", end="", flush=True)
    tau_2 = execute_stroke_and_measure(pose_A, joint_idx, speed, accel, stroke_deg)
    print(f" 완료 (평균: {tau_2:.3f})")

    # 방향 판별 (각도가 커지는 쪽이 +방향)
    if end_angle > start_angle:
        tau_plus, tau_minus = tau_1, tau_2
    else:
        tau_plus, tau_minus = tau_2, tau_1

    # 순수 동마찰력 계산 (중력 소거)
    friction = (tau_plus - tau_minus) / 2.0
    
    return float(tau_plus), float(tau_minus), float(friction)

# =========================================================
# 4. 메인 시나리오 (10회 반복 측정 및 JSON 저장)
# =========================================================
# 관절별 측정 설정값 
test_configs = {
    0: {"start": 0.0, "end": 180.0, "speed": 50.0, "accel": 100.0},
    1: {"start": -20.0, "end": 90.0, "speed": 50.0, "accel": 100.0},
    2: {"start": -90.0, "end": 90.0, "speed": 50.0, "accel": 100.0},
    3: {"start": -90.0, "end": 90.0, "speed": 50.0, "accel": 100.0},
    4: {"start": -135.0, "end": -45.0, "speed": 50.0, "accel": 100.0},
    5: {"start": -90.0, "end": 90.0, "speed": 50.0, "accel": 100.0}
}

iterations = 10
results = {}

for j_idx in range(6):
    conf = test_configs[j_idx]
    print(f"\n{'='*50}")
    print(f"[TEST] Joint {j_idx} 동마찰력 10회 반복 측정 시작")
    print(f"설정: {conf['start']}° -> {conf['end']}° | 속도: {conf['speed']} | 가속도: {conf['accel']}")
    print(f"{'='*50}")
    
    # [수정된 핵심 포인트] 반복 측정 시작 전에 딱 한 번만 시작 위치(A)로 이동
    pose_A = BASE_POSE.copy()
    pose_A[j_idx] = conf["start"]
    print(f"  [준비] {conf['start']}° 위치로 초기화 이동 중...")
    robot.move_j(rc, pose_A, 40.0, 60.0)
    robot.wait_for_move_finished(rc)
    time.sleep(1.0) # 잔류 진동 소산 대기
    print("  [준비] 완료. 본격적인 측정을 시작합니다.")
    
    j_results = []
    
    for i in range(1, iterations + 1):
        print(f"  [Iter {i}/{iterations}]")
        t_plus, t_minus, fric = measure_joint_dynamic_friction(
            j_idx, conf["start"], conf["end"], conf["speed"], conf["accel"]
        )
        
        j_results.append({
            "iteration": i,
            "tau_plus": round(t_plus, 4),
            "tau_minus": round(t_minus, 4),
            "friction": round(fric, 4)
        })
        time.sleep(0.5)

    # 10회 평균 계산
    avg_friction = np.mean([r["friction"] for r in j_results])
    print(f"\n[결과] Joint {j_idx} 10회 평균 동마찰력: {avg_friction:.4f}\n")
    
    results[f"joint_{j_idx}"] = {
        "config": conf,
        "average_friction": round(float(avg_friction), 4),
        "measurements": j_results
    }

# =========================================================
# 5. JSON 파일 저장
# =========================================================
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"[SUCCESS] 모든 측정이 완료되었습니다! 결과가 '{OUTPUT_JSON_PATH}'에 저장되었습니다.")