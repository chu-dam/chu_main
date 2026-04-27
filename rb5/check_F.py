from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np
import rbpodo as rb

# =========================================================
# 1. 로봇 초기 연결 및 설정
# =========================================================
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

# =========================================================
# 2. MuJoCo 모델 로드 (뷰어용)
# =========================================================
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

if state is not None:
    d.qpos[:] = np.deg2rad(rb_get_joint_position(state))
    mujoco.mj_forward(m, d)

prev_time = time()
prev_jpos = None

# 로그가 너무 빨리 도배되는 것을 막기 위한 타이머 변수
last_print_time = 0.0 

# 파이썬 측에서는 아무런 힘을 가하지 않음
zero_torque = np.zeros(6)

print("\n[Zero Torque Mode] 로봇 자체 중력/마찰 보상(comp=3) 실행 중...")
print("로봇 끝단(TCP)이나 관절을 손으로 밀어서 외력 감지 로그가 뜨는지 확인하세요!\n")

# =========================================================
# 3. 메인 제어 루프
# =========================================================
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
        
        # ---------------------------------------------------------
        # [핵심] 외력 감지 및 모니터링 (eft 변수 사용)
        # ---------------------------------------------------------
        # 힘(Force) 성분 X, Y, Z
        tcp_ext_force = np.array([
            state.sdata.eft_fx, 
            state.sdata.eft_fy, 
            state.sdata.eft_fz
        ])
        
        # 모멘트(Moment) 성분 X, Y, Z
        tcp_ext_moment = np.array([
            state.sdata.eft_mx, 
            state.sdata.eft_my, 
            state.sdata.eft_mz
        ])
        
        # 크기(Norm) 계산
        force_norm = np.linalg.norm(tcp_ext_force)
        moment_norm = np.linalg.norm(tcp_ext_moment)

        # 임계값 설정 (실제 로봇의 평상시 노이즈 수준을 보고 조금 높게 설정하세요)
        FORCE_THRESH = 15.0   # N (약 1.5kgf)
        MOMENT_THRESH = 3.0   # Nm 


        jnt_currents = np.array(state.sdata.jnt_cur)
        
        # 특정 관절(예: 1, 2, 3번 축)에 전류가 확 튀는지 확인
        max_current = np.max(np.abs(jnt_currents))
        if now - last_print_time > 0.5:
            print(f"[디버깅] 현재 Max 전류량: {max_current:.1f} mA | 전체: {np.round(jnt_currents, 1)}")
            last_print_time = now

        

    

        # 뷰어(디지털 트윈) 업데이트
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = np.deg2rad(jvel)
        mujoco.mj_forward(m, d)
 
        # 안전장치 2: 관절 속도 초과 감지
        if np.any(np.abs(jvel) > 70):
            print(f"속도 초과 보호 작동! | Jvel : {jvel}")
            continue
            
        try:
            # 입력 토크는 0, compensation=3 설정으로 로봇이 자체 보상으로 버티게 함
            robot.move_servo_t(rc, zero_torque, t1, t2, compensation=3)
            sleep(0.005)
        except Exception as e:
            print(f"제어 명령 전송 실패: {e}")

        viewer.sync()