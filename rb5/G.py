from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np
import rbpodo as rb

# 1. 로봇 초기 연결 및 설정
try:
    ROBOT_ADDRESS = "169.254.186.20"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    robot_data = rb.CobotData(ROBOT_ADDRESS)
    
    state = robot_data.request_data()
    
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    robot.set_freedrive_mode(rc, on=False)
    
    t1, t2 = 0.01, 0.05
    
except Exception as e:
    print(f"로봇 연결 실패! {e}")
    raise SystemExit

def rb_get_joint_position(current_state):
    if current_state is None:
        return np.zeros(6)
    return np.array(current_state.sdata.jnt_ang)

# 2. MuJoCo 모델 로드 (뷰어용)
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

if state is not None:
    d.qpos[:] = np.deg2rad(rb_get_joint_position(state))
    mujoco.mj_forward(m, d)

prev_time = time()
prev_jpos = None

# 파이썬 측에서는 아무런 힘을 가하지 않음 (오직 로봇 자체 보상에 의존)
zero_torque = np.zeros(6)

print("\n[Zero Torque Mode] 로봇 자체 중력/마찰 보상(comp=3) 실행 중...")

# 3. 메인 제어 루프
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        state = robot_data.request_data()
        if state is None:
            continue
            
        # 안전장치 1: 충돌 및 에러 감지
        if state.sdata.op_stat_collision_occur or state.sdata.op_stat_sos_flag == 4:
            print("안전 정지 (충돌 또는 에러)")
            break
        
        now = time()
        loop_dt = now - prev_time
        prev_time = now

        if loop_dt <= 0.0:
            continue

        # 현재 관절 위치 및 속도 계산
        jpos = rb_get_joint_position(state)
        
        jvel = np.zeros(6)
        if prev_jpos is not None:
            jvel = (jpos - prev_jpos) / loop_dt
        prev_jpos = jpos
        
        # 뷰어(디지털 트윈) 업데이트
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = np.deg2rad(jvel)
        mujoco.mj_forward(m, d)
 
        # 안전장치 2: 관절 속도 초과 감지
        if np.any(np.abs(jvel) > 70):
            print(f"속도 초과 보호 작동! | Jvel : {jvel}")
            continue
            
        try:
            # 핵심: 입력 토크는 0이지만, compensation=3(u + g + f) 설정으로 인해 
            # 로봇의 내부 제어기가 자체적으로 중력(g)과 마찰(f)을 완벽하게 계산하여 버텨줍니다.
            robot.move_servo_t(rc, zero_torque, t1, t2, compensation=3)
            sleep(0.005)
        except Exception as e:
            print(f"제어 명령 전송 실패: {e}")

        viewer.sync()