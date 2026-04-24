from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np

from rbpodo import Cobot, SystemVariable, CobotData
import rbpodo as rb


######## torque servo
try:
    ROBOT_ADDRESS = "192.169.1.200"

    robot = rb.Cobot(ROBOT_ADDRESS)
    rc = rb.ResponseCollector()

    robot_data = CobotData(ROBOT_ADDRESS)
    state = robot_data.request_data()

    robot.set_operation_mode(rc, rb.OperationMode.Simulation)
    # 실제 로봇 구동 시에는 충분히 확인 후 Real로 변경
    # robot.set_operation_mode(rc, rb.OperationMode.Real)

    robot.set_speed_bar(rc, 0.5)
    # robot.set_freedrive_mode(rc, on=False)

    t1 = 0.01  # 이동시간
    t2 = 0.05  # 유지시간

except Exception as e:
    print(f"No Robot Connection ..! {e}")
    raise SystemExit
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


def orientation_error(R_des, R_cur):
    e = 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0]) +
        np.cross(R_cur[:, 1], R_des[:, 1]) +
        np.cross(R_cur[:, 2], R_des[:, 2])
    )
    return e


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


def rb_get_joint_position(state):
    if state is None:
        raise RuntimeError("Failed to get robot state.")

    jpos = state.sdata.jnt_ang  # 조인트 위치 (deg)
    return np.array(jpos, dtype=np.float64)


def rb_get_tcp_pose(state):
    """
    실제 로봇 TCP pose를 받아옵니다.

    state.sdata.tcp_info:
        [x, y, z, rx, ry, rz]

    단위:
        x, y, z: mm
        rx, ry, rz: deg

    반환:
        tcp_pos_m: [x, y, z] in meter
        tcp_R: rotation matrix
        tcp_rpy_deg: [rx, ry, rz] in degree
    """
    if state is None:
        raise RuntimeError("Failed to get robot state.")

    tcp_info = np.array(state.sdata.tcp_info, dtype=np.float64)

    tcp_pos_mm = tcp_info[0:3]
    tcp_rpy_deg = tcp_info[3:6]

    tcp_pos_m = tcp_pos_mm * 0.001
    tcp_R = rpy_to_rotmat(*tcp_rpy_deg)

    return tcp_pos_m, tcp_R, tcp_rpy_deg


# --------- desired ---------
# 위치 단위: meter
desired_xpos_tcp = np.array([0.0, -0.45, 0.35])

# 자세 단위: degree
desired_rpy = np.array([90.0, 0.0, 0.0])  # roll, pitch, yaw 입력
R_desired = rpy_to_rotmat(*desired_rpy)
# -----------------------------


model_path = "/home/kdh/Desktop/delto/delto_description2/rb3_single/scene_rb3.xml"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


# initial robot pose
try:
    state = robot_data.request_data()

    jpos, jvel = rb_get_joint_state()
    print(f"Initial jpos : {jpos} | jvel : {jvel}")

    d.qpos[:] = np.deg2rad(jpos)
    d.qvel[:] = np.deg2rad(jvel)

    mujoco.mj_forward(m, d)

    real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)

    print(f"Initial real TCP pos : {real_tcp_pos}")
    print(f"Initial real TCP rpy : {real_tcp_rpy}")

    sleep(3)

except Exception as e:
    print(f"Can't get robot init data ..! {e}")

    d.qpos[:] = [-0.5, -0.3, 1.3, 0.4, 1.57, 0.0]
    d.qvel[:] = 0.0

    mujoco.mj_forward(m, d)
    sleep(3)


tcp_site_id = m.site("tcp").id

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)


# Position VSD
# K_pos: N/m
# D_pos: N/(m/s)
K_pos = np.diag([500.0, 500.0, 500.0])
D_pos = np.diag([1.2, 1.2, 1.2])


# Orientation VSD
# K_ori: Nm/rad
# D_ori: Nm/(rad/s)
K_ori = np.diag([1.0, 1.0, 1.0])
D_ori = np.diag([0.3, 0.3, 0.3])


prev_time = time()
hz_window = []
prev_jpos = None

jvel_filtered = np.zeros(6)
alpha_vel = 0.15


max_total_torque = 30.0
max_pos_torque = 30.0
max_ori_torque = 10.0


state_key = {
    "started": False
}


def key_callback(keycode):
    if keycode == ord('S') or keycode == ord('s'):
        state_key["started"] = True
        print("\n[KEY] S pressed -> torque servoing started\n")


print("[TARGET TCP POS]", desired_xpos_tcp)
print("[TARGET TCP RPY]", desired_rpy)
print("Press 's' in the MuJoCo viewer window to start torque servoing.")


with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=key_callback
) as viewer:

    t0 = time()

    while viewer.is_running():
        state = robot_data.request_data()

        if state is None:
            print("Failed to request robot data.")
            viewer.sync()
            sleep(0.01)
            continue

        if state.sdata.op_stat_collision_occur:
            print("Robot in Collision")
            break

        if state.sdata.op_stat_sos_flag == 4:
            print("Command Input Error")
            break

        now = time()
        loop_dt = now - prev_time
        prev_time = now

        try:
            ## real data update ##
            jpos = rb_get_joint_position(state)

            jvel_raw = np.zeros(6)
            if prev_jpos is not None and loop_dt > 0:
                jvel_raw = (jpos - prev_jpos) / loop_dt
                # print(f"jvel calc :: {jvel_raw} = {jpos}-{prev_jpos}/{loop_dt}")

            jvel_filtered = alpha_vel * jvel_raw + (1.0 - alpha_vel) * jvel_filtered

            prev_jpos = jpos.copy()

            # print(f"JP : {jpos} | JV : {jvel_filtered}")

            d.qpos[:] = np.deg2rad(jpos)
            d.qvel[:] = np.deg2rad(jvel_filtered)

            mujoco.mj_forward(m, d)

            # 실제 로봇 TCP pose
            real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)

        except Exception as e:
            print(f"real data update failed..! {e}")
            viewer.sync()
            sleep(0.01)
            continue
        ######################

        # s 누르기 전에는 토크 명령을 보내지 않음
        if not state_key["started"]:
            mujoco_tcp_pos = d.site("tcp").xpos.copy()

            print(f"[MODE] WAIT")
            print(f"[REAL TCP POS]   {real_tcp_pos}")
            print(f"[REAL TCP RPY]   {real_tcp_rpy}")
            print(f"[MUJOCO TCP POS] {mujoco_tcp_pos}")
            print(f"[TCP DIFF]       {real_tcp_pos - mujoco_tcp_pos}")
            print("Press 's' to start.")
            print("--------------------------------------")

            viewer.sync()
            sleep(0.01)
            continue

        # 1. TCP Jacobian
        # Jacobian은 실제 관절각을 반영한 MuJoCo 모델에서 계산
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        jacp0 = jacp[:, 0:6].copy()
        jacr0 = jacr[:, 0:6].copy()

        # 2. Current TCP state
        # 위치/자세 오차 계산에는 MuJoCo TCP가 아니라 실제 로봇 TCP를 사용
        current_xpos_tcp = real_tcp_pos.copy()
        R_current = real_tcp_R.copy()

        # 3. Velocity
        # 속도는 실제 관절속도를 MuJoCo Jacobian에 통과시켜 계산
        xpos_dot0 = jacp0 @ d.qvel[0:6]
        w0 = jacr0 @ d.qvel[0:6]

        # 4. Position Virtual Spring-Damper
        xpos_err0 = desired_xpos_tcp - current_xpos_tcp

        # desired linear velocity = 0
        xvel_err0 = -xpos_dot0

        F_spring = K_pos @ xpos_err0
        F_damp = D_pos @ xvel_err0
        force0 = F_spring + F_damp

        tau_pos = jacp0.T @ force0
        tau_pos = np.clip(tau_pos, -max_pos_torque, max_pos_torque)

        # 5. Orientation Virtual Spring-Damper
        ori_err0 = orientation_error(R_desired, R_current)

        # desired angular velocity = 0
        w_err0 = -w0

        M_spring = K_ori @ ori_err0
        M_damp = D_ori @ w_err0
        F_ori_0 = M_spring + M_damp

        tau_ori = jacr0.T @ F_ori_0
        tau_ori = np.clip(tau_ori, -max_ori_torque, max_ori_torque)

        # 6. Total torque
        torque0 = tau_pos + tau_ori

        d.ctrl[0:6] = np.clip(torque0, -max_total_torque, max_total_torque)

        if np.any(np.abs(jvel_filtered) > 70):  # Joint Vel Limit
            i = np.where(np.abs(jvel_filtered) > 70)[0]

            if len(i) > 0:
                d.ctrl[i] = 0.0
                print(torque0)
                print(f"Joint Velocity is too fast ...! Joint{list(i)} | Jvel : {jvel_filtered[i]}")

        target_torque = 0.0 #+ d.ctrl[0:6]

        # Hz 측정 (1 / 주기)
        if loop_dt > 0:
            hz = 1.0 / loop_dt
            hz_window.append(hz)
            if len(hz_window) > 30:  # 최근 30프레임 평균
                hz_window.pop(0)

        mujoco_tcp_pos = d.site("tcp").xpos.copy()

        print(f"[Desired POS]        {desired_xpos_tcp}")
        print(f"[TCP POS]   {real_tcp_pos}")
        print(f"[TCP RPY]   {real_tcp_rpy}")
        print(f"[POS ERR]        {xpos_err0}")
        print(f"[ORI ERR]        {ori_err0}")
        print("--------------------------------------")

        # 토크 서보잉 입력
        try:
            ret = robot.move_servo_t(rc, target_torque, t1, t2, compensation=3)
            # comp 0 : u / 1 : u + g / 2 : u + f / 3 : u + g + f

            sleep(0.005)  # t2 = 0.05

            if not ret.is_success():
                print(f"move_servo_t 실패 ", ret)

        except Exception as e:
            print(f"T-servo Failed ..! {e}")

        viewer.sync()