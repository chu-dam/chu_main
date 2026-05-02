from time import time, sleep
from copy import deepcopy
import os
import csv
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb
import numpy as np

######## torque servo
try:
    ROBOT_ADDRESS = "169.254.186.20" 

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()
    
    robot_data = CobotData(ROBOT_ADDRESS)
    state = robot_data.request_data()
    
    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    #robot.set_freedrive_mode(rc, on=False)
    t1 = 0.01 #이동시간
    t2 = 0.05 #유지시간
    
except Exception as e:
    print(f"No Robot Connection ..! {e}")
    pass
########

def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # ZYX (yaw-pitch-roll) 순서
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    # R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def rb_get_joint_state():
    jpos = []
    jvel = []
    for i in range(6):
        var_pos = getattr(SystemVariable, f"SD_J{i}_ANG")
        var_vel = getattr(SystemVariable, f"SD_J{i}_VEL")
        _, pos = robot.get_system_variable(rc, var_pos)
        _, vel = robot.get_system_variable(rc, var_vel)
        jpos.append(pos)
        jvel.append(vel)
    return np.array(jpos), np.array(jvel)

def rb_get_joint_position():
    jpos = []
    if state is None:
        print("Failed to get robot state.")
    else:
        jpos = state.sdata.jnt_ang  # 조인트 위치 (deg)
    return np.array(jpos)

def rb_get_tcp_pose():
    tcp_pos = []
    tcp_rpy = []
    _, tcp_info = robot.get_tcp_info(rc)
    tcp_pos = np.array(tcp_info[0:3]) / 1000.0  # mm -> m
    tcp_rpy = np.array(tcp_info[3:6])           # deg

    return np.array(tcp_pos), np.array(tcp_rpy)

def rb_move_j_and_wait(joint_deg, vel=60, acc=80):
    joint_deg = np.asarray(joint_deg, dtype=np.float64)

    print(f"[INIT_MOVE_J] target joint = {joint_deg}")

    ret = robot.move_j(rc, joint_deg, vel, acc)
    rc.error().throw_if_not_empty()

    if robot.wait_for_move_started(rc, 0.5).is_success():
        robot.wait_for_move_finished(rc)
    else:
        print("[INIT_MOVE_J] move start check timeout, but continue waiting for finish")
        robot.wait_for_move_finished(rc)

    rc.error().throw_if_not_empty()

    print("[INIT_MOVE_J] finished")

def virtual_offset_piecewise_mm(
    dist_m,
    const_err_mm=70.0,
    taper_dist_mm=10.0,
):
    dist_mm = dist_m * 1000.0

    if dist_mm >= const_err_mm:
        L_mm = 0.0

    elif dist_mm > taper_dist_mm:
        L_mm = const_err_mm - dist_mm

    else:
        if taper_dist_mm <= 1e-9:
            L_mm = 0.0
        else:
            L_mm = (const_err_mm - taper_dist_mm) * (dist_mm / taper_dist_mm)

    return L_mm * 0.001

def save_csv_log(log_data, log_index, tag="complete"):
    if len(log_data) == 0:
        return

    os.makedirs(CSV_SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        CSV_SAVE_DIR,
        f"vsd_return_{tag}_{timestamp}_{log_index:03d}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_sec",
            "tcp_x_m",
            "tcp_y_m",
            "tcp_z_m",
            "goal_err_x_m",
            "goal_err_y_m",
            "goal_err_z_m",
            "goal_err_norm_m",
            "goal_err_norm_mm",
        ])

        writer.writerows(log_data)

    print("====================================")
    print(f"[CSV SAVE] {filename}")
    print(f"[CSV SAVE] samples = {len(log_data)}")
    print("====================================")

def mujoco_key_callback(keycode):
    global request_csv_start
    global request_exit

    try:
        key = chr(keycode).lower()
    except Exception:
        return

    if key == "s":
        request_csv_start = True
        print("====================================")
        print("[MUJOCO KEY] s pressed. CSV logging start requested.")
        print("====================================")

    elif key == "q":
        request_exit = True
        print("[MUJOCO KEY] q pressed. Exit requested.")

# --------- initial move_j + desired setting ---------
INIT_JOINT_DEG = np.array([90.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float64)

# 기존 방식 유지: 목표 자세는 고정값으로 사용
desired_rpy = np.array([90.0, 0.0, 0.0], dtype=np.float64)

try:
    # 1. 원하는 초기 joint 자세로 이동
    rb_move_j_and_wait(INIT_JOINT_DEG, vel=60, acc=80)

    # 2. 이동 후 안정화 시간
    sleep(1.0)

    # 3. 현재 TCP pose 읽기
    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose()

    # 4. 위치 목표만 현재 TCP 위치로 설정
    desired_xpos_tcp = current_tcp_pos.copy()

    print("[INIT] Desired TCP position is set from current TCP after move_j")
    print(f"[INIT] current_tcp_pos  = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy  = {current_tcp_rpy}")
    print(f"[INIT] desired_xpos_tcp = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy      = {desired_rpy}")

except Exception as e:
    print(f"[INIT] move_j failed or desired TCP position setting failed: {e}")

    # 실패해도 현재 위치를 목표 위치로 잡아서 갑자기 튀는 것 방지
    current_tcp_pos, current_tcp_rpy = rb_get_tcp_pose()
    desired_xpos_tcp = current_tcp_pos.copy()

    print("[INIT] Fallback: Desired TCP position is set from current TCP")
    print(f"[INIT] current_tcp_pos  = {current_tcp_pos}")
    print(f"[INIT] current_tcp_rpy  = {current_tcp_rpy}")
    print(f"[INIT] desired_xpos_tcp = {desired_xpos_tcp}")
    print(f"[INIT] desired_rpy      = {desired_rpy}")
# ----------------------------------------------------

model_path = "/home/chu/chu_main/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# initial robot pose
try:
    jpos, jvel = rb_get_joint_state()
    print(f"Initial jpos : {jpos} | jvel : {jvel}")

    while (len(jpos)==0):
        print("Waiting for robot init data..")
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = 0 #TODO
    sleep(3)
except Exception as e:
    print(f"Can't get robot init data ..! {e}")
    d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
    d.qvel[:] = 0
    sleep(3)


M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

C0 = np.zeros((6,6))

K_a = 150.0
zeta_a = 4.0

K_o = 4.0
zeta_o = 2.5

varsigma = 0.5

######################
# Virtual Target
USE_VIRTUAL_RETURN = True

# 목표점과 TCP 사이 거리가 이 값보다 크면 가상 목표점 사용 안 함
VIRTUAL_CONST_ERR_MM = 70.0

# 목표점과 TCP 사이 거리가 이 값보다 작으면 가상 목표점이 다시 TCP 쪽으로 줄어듦
VIRTUAL_TAPER_DIST_MM = 10.0

MOVING_AWAY_DOT_THRESH = 1e-5
######################

######################
# CSV Logging

CSV_SAVE_DIR = "./vsd_return_logs"

# s를 누른 순간부터 CSV 저장 시작
# TCP 속도가 이 값보다 작게 일정 시간 유지되면 "수렴이 멈췄다"고 판단
LOG_STOP_VEL_MPS = 0.002       # 2 mm/s
LOG_STOP_HOLD_SEC = 1.0

# 너무 짧은 로그 저장 방지
LOG_MIN_SAVE_SEC = 0.5

######################

######################
# Friction Coef
Cfc = np.array([6.7569, 3.5644, 3.0893, 1.8396, 1.8396, 1.8396])
Vfc = np.array([0.1515, 0.4009, 0.3245, 0.0388, 0.0388, 0.0388])
friction_curve_coef = 8*1e-1

######################
prev_time = time()
hz_window = []
prev_jpos = None  

request_csv_start = False
request_exit = False

csv_logging = False
csv_log_data = []
csv_log_count = 0
csv_start_time = None
csv_stop_candidate_time = None

with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=mujoco_key_callback
) as viewer:
    t0 = time()
    while viewer.is_running():
        if request_exit:
            break

        state = robot_data.request_data()

        if state is None:
            continue
        
        if state.sdata.op_stat_collision_occur:
            print("Robot in Collision")
            break
        if state.sdata.op_stat_sos_flag==4:
            print(f"Command Input Error | JVEL : {jvel}")
            break
        
        now = time()
        loop_dt = now - prev_time
        prev_time = now

        mujoco.mj_step(m, d)
        mujoco.mj_fullM(m, M, d.qM)

        try:
        ## real data update ##
            jpos = rb_get_joint_position()
            jvel = np.zeros(6)
            if prev_jpos is not None:
                jvel = (jpos - prev_jpos) / loop_dt
                # print(f"jvel calc :: {jvel} = {jpos}-{prev_jpos}/{loop_dt}")
            prev_jpos = jpos
            
            # print(f"JP : {jpos} | JV : {jvel}")
            
            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel)
        except Exception as e:
            print(f"real data update failed..! {e}")
        ######################
        
        qvel_backup = deepcopy(d.qvel)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G) 
        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)        

        np.fill_diagonal(C0, varsigma * np.sqrt(np.sum(np.abs(M[0:6, 0:6]), axis=1)))

        tcp_site_id = m.site("tcp").id
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        jacp0 = deepcopy(jacp[:, 0:6])
        jacr0 = deepcopy(jacr[:, 0:6])
        # print(f"jr : {jacr0}")    

        # Position Error
        # Position Error
        tcp_pos, tcp_rpy = rb_get_tcp_pose()

        # Orientation Error (RPY 입력 반영)
        R_current = rpy_to_rotmat(*tcp_rpy)
        # print(f"Ro : {R_current}")
        
        R_desired = rpy_to_rotmat(*desired_rpy)
        # print(f"Rd : {R_desired}")
        # R_desired = R_desired@R_desired
        ori_err0 = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        ) # e  = (x X xd) + (y X yd) + (z X zd)
        
        # Angular Velocity
        w0 = jacr0 @ d.qvel[0:6]
        # print(f"wo : {w0}")

        # Orientation Force
        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        # Linear Velocity
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        # Virtual Target
        to_goal = desired_xpos_tcp - tcp_pos
        goal_dist = np.linalg.norm(to_goal)

        virtual_axis_weight = np.array([1.0, 3.0, 2.0])

        to_goal_weighted = virtual_axis_weight * to_goal
        weighted_goal_dist = np.linalg.norm(to_goal_weighted)

        if weighted_goal_dist > 1e-9:
            u_goal = to_goal_weighted / weighted_goal_dist
        else:
            u_goal = np.zeros(3)

        goal_err = tcp_pos - desired_xpos_tcp
        moving_away = np.dot(goal_err, xpos_dot0) > MOVING_AWAY_DOT_THRESH

        if USE_VIRTUAL_RETURN and not moving_away:
            L_virtual = virtual_offset_piecewise_mm(
                goal_dist,
                const_err_mm=VIRTUAL_CONST_ERR_MM,
                taper_dist_mm=VIRTUAL_TAPER_DIST_MM,
            )
        else:
            L_virtual = 0.0

        virtual_xpos_tcp = desired_xpos_tcp + u_goal * L_virtual

        # Control Error
        xpos_err0 = tcp_pos - virtual_xpos_tcp

        # Linear Force 
        force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)

        fric_scale = np.array([0.70, 1.0, 1.0, 0.7, 0.8, 0.8])
        Tf = fric_scale * (Cfc * np.tanh(friction_curve_coef * jvel) + Vfc * jvel)
        # print(f"Friction Torque : {Tf}")
        
        # Torque (Coli + Gravity + Damping + Orientation)
        torque0 = (- 1 * C0 @ d.qvel[0:6] 
                   - 1 * jacp0.T @ force0
                   + 1 * G[0:6] 
                   - 1 * jacr0.T @ F_ori_0
                   + 1 * Tf[0:6])
        # print(f"torque 0 : {torque0}")
        
        max_torque = 50
        d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
 
        if np.any(jvel > 70): # Joint Vel Limit
            i = np.where(jvel > 70)[0]  # 튜플에서 실제 인덱스만 가져옴

            if len(i) > 0:
                d.ctrl[i] = 0.0 * torque0[i]
                print(torque0)
                print(f"Joint Velocity is too fast ...! Joint{list(i)} | Jvel : {jvel[i]}")
                
        # print(f"Taget torque : {d.ctrl[0:6]}")
        
        target_torque =  d.ctrl[0:6]

        # CSV Logging
        goal_err_norm = np.linalg.norm(goal_err)
        vel_norm = np.linalg.norm(xpos_dot0)

        # 1. s 키가 눌렸으면 그 시점부터 CSV 기록 시작
        if request_csv_start:
            request_csv_start = False

            if not csv_logging:
                csv_logging = True
                csv_log_count += 1
                csv_log_data = []
                csv_start_time = now
                csv_stop_candidate_time = None

                print("====================================")
                print(f"[CSV START] log index = {csv_log_count}")
                print(f"[CSV START] goal_err_norm = {goal_err_norm * 1000.0:.2f} mm")
                print(f"[CSV START] vel_norm = {vel_norm * 1000.0:.2f} mm/s")
                print("====================================")

            else:
                print("====================================")
                print("[CSV START] ignored. CSV logging is already running.")
                print("====================================")

        # 2. 기록 중이면 TCP 좌표와 실제 목표 오차 저장
        if csv_logging:
            csv_t = now - csv_start_time

            csv_log_data.append([
                csv_t,
                tcp_pos[0],
                tcp_pos[1],
                tcp_pos[2],
                goal_err[0],
                goal_err[1],
                goal_err[2],
                goal_err_norm,
                goal_err_norm * 1000.0,
            ])

            # 3. TCP 속도가 충분히 작게 유지되면 수렴이 멈춘 것으로 판단
            if vel_norm < LOG_STOP_VEL_MPS:
                if csv_stop_candidate_time is None:
                    csv_stop_candidate_time = now

                elif ((now - csv_stop_candidate_time) >= LOG_STOP_HOLD_SEC) and (csv_t >= LOG_MIN_SAVE_SEC):
                    save_csv_log(csv_log_data, csv_log_count, tag="complete")

                    csv_logging = False
                    csv_log_data = []
                    csv_start_time = None
                    csv_stop_candidate_time = None

            else:
                csv_stop_candidate_time = None

         # Hz 측정 (1 / 주기)
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)
            if len(hz_window) > 30:  # 최근 30프레임 평균
                hz_window.pop(0)
            
            print(f"[move_servo_t] Hz = {hz:.2f} (avg={np.mean(hz_window):.2f})")
            print(f"[TCP_DES] {desired_xpos_tcp}")
            print(f"[TCP_VIRTUAL] {virtual_xpos_tcp}")
            print(f"[TCP_CUR] {tcp_pos}")
            print(f"[GOAL_ERR] {goal_err} | norm = {goal_err_norm:.4f}")
            #print(f"[TCP_ERR] {xpos_err0} | norm = {np.linalg.norm(xpos_err0):.4f}")
            print(f"[VIRTUAL_L] {L_virtual * 1000.0:.3f} mm")
            print(f"[MOVING_AWAY] {moving_away}")
            print(f"[VEL_NORM] {vel_norm:.5f}")
            print(f"[CSV_LOGGING] {csv_logging}")
            print(f"[RPY_DES] {desired_rpy}")
            print(f"[RPY_CUR] {tcp_rpy}")
            print(f"[ORI_ERR] {ori_err0} | norm = {np.linalg.norm(ori_err0):.4f}")
            print(f"[F_ORI] {F_ori_0}")
            print(f"[C0 diag] {np.diag(C0)}")
            print(f"[Target torque] {d.ctrl[0:6]}")
            print("------------------------------------")
       
        # 토크 서보잉 입력
        try:
            target_torque =  d.ctrl[0:6]
            ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation=0)
            # # comp 0 : u / 1 : u + g / 2 : u + f / 3 : u + g + f
            sleep(0.005) # t2 = 0.05
            if not ret.is_success():
                print(f"move_servo_t 실패 ", ret)
                
        except Exception as e:
            print(f"T-servo Failed ..! {e}")

        viewer.sync()

if csv_logging and len(csv_log_data) > 0:
    save_csv_log(csv_log_data, csv_log_count, tag="partial")