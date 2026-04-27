from time import time, sleep
import mujoco
import mujoco.viewer
import numpy as np
import rbpodo as rb

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


def rpy_to_rotmat(roll, pitch, yaw):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])

    return Rz @ Ry @ Rx


def rb_get_joint_position(current_state):
    if current_state is None:
        return np.zeros(6)
    return np.array(current_state.sdata.jnt_ang, dtype=np.float64)


def rb_get_tcp_pose(current_state):
    if hasattr(current_state.sdata, "tcp"):
        tcp_info = np.array(current_state.sdata.tcp, dtype=np.float64)
    elif hasattr(current_state.sdata, "tcp_pos"):
        tcp_info = np.array(current_state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(current_state.sdata, "cur_pos"):
        tcp_info = np.array(current_state.sdata.cur_pos, dtype=np.float64)
    else:
        return np.zeros(3), np.eye(3), np.zeros(3)

    tcp_pos_m = tcp_info[0:3] * 0.001
    tcp_R = rpy_to_rotmat(*tcp_info[3:6])
    tcp_rpy = tcp_info[3:6]

    return tcp_pos_m, tcp_R, tcp_rpy


model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

if state is not None:
    d.qpos[0:6] = np.deg2rad(rb_get_joint_position(state))
    mujoco.mj_forward(m, d)

tcp_site_id = m.site("tcp").id

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

zero_torque = np.zeros(6)

prev_time = time()
prev_jpos = None

AXIS_NAMES = ["x", "y", "z"]

TEST_DIRECTIONS = [1.0, -1.0]

FORCE_START_N = 0.0
FORCE_STEP_N = 0.25
FORCE_MAX_N = 25.0

HOLD_TIME_PER_STEP = 0.30
SETTLE_TIME = 0.80

MOVE_AXIS_THRESH_M = 0.0010
MOVE_NORM_THRESH_M = 0.0015

MAX_TEST_TORQUE = 30.0
MAX_JVEL_DEG_S = 70.0

results = []


def read_robot_and_update_model():
    global prev_time, prev_jpos

    current_state = robot_data.request_data()
    if current_state is None:
        return None

    if current_state.sdata.op_stat_collision_occur or current_state.sdata.op_stat_sos_flag == 4:
        raise RuntimeError("안전 정지: 충돌 또는 에러 감지")

    now = time()
    loop_dt = now - prev_time
    prev_time = now

    if loop_dt <= 0.0:
        return None

    jpos = rb_get_joint_position(current_state)

    jvel = np.zeros(6)
    if prev_jpos is not None:
        jvel = (jpos - prev_jpos) / loop_dt

    prev_jpos = jpos.copy()

    d.qpos[0:6] = np.deg2rad(jpos)
    d.qvel[0:6] = np.deg2rad(jvel)

    mujoco.mj_forward(m, d)
    mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

    jacp0 = jacp[:, 0:6]
    tcp_pos, tcp_R, tcp_rpy = rb_get_tcp_pose(current_state)
    tcp_dot = jacp0 @ d.qvel[0:6]

    return {
        "state": current_state,
        "jpos": jpos,
        "jvel": jvel,
        "tcp_pos": tcp_pos,
        "tcp_R": tcp_R,
        "tcp_rpy": tcp_rpy,
        "jacp0": jacp0,
        "tcp_dot": tcp_dot,
    }


def send_torque(tau_cmd):
    tau_cmd = np.asarray(tau_cmd, dtype=np.float64)
    tau_cmd = np.clip(tau_cmd, -MAX_TEST_TORQUE, MAX_TEST_TORQUE)
    robot.move_servo_t(rc, tau_cmd, t1, t2, compensation=3)
    return tau_cmd


def hold_zero_torque(viewer, duration):
    start = time()

    while viewer.is_running() and time() - start < duration:
        data = read_robot_and_update_model()
        if data is None:
            continue

        if np.any(np.abs(data["jvel"]) > MAX_JVEL_DEG_S):
            print(f"[WARN] 속도 초과 감지, zero torque 유지 | JVEL = {data['jvel']}")
            send_torque(zero_torque)
            viewer.sync()
            sleep(0.005)
            continue

        send_torque(zero_torque)
        viewer.sync()
        sleep(0.005)


def test_axis_direction(viewer, axis_idx, direction):
    axis_name = AXIS_NAMES[axis_idx]
    dir_name = "+" if direction > 0 else "-"

    print("\n======================================")
    print(f"[TEST START] Axis = {dir_name}{axis_name}")
    print("======================================")

    hold_zero_torque(viewer, SETTLE_TIME)

    data = read_robot_and_update_model()
    if data is None:
        print("[ERROR] 초기 상태 읽기 실패")
        return None

    start_tcp = data["tcp_pos"].copy()

    print(f"[START TCP] {start_tcp}")

    force_values = np.arange(
        FORCE_START_N + FORCE_STEP_N,
        FORCE_MAX_N + FORCE_STEP_N * 0.5,
        FORCE_STEP_N,
    )

    for force_mag in force_values:
        step_start = time()

        max_axis_delta = 0.0
        max_norm_delta = 0.0
        last_tau_cmd = np.zeros(6)

        while viewer.is_running() and time() - step_start < HOLD_TIME_PER_STEP:
            data = read_robot_and_update_model()
            if data is None:
                continue

            if np.any(np.abs(data["jvel"]) > MAX_JVEL_DEG_S):
                print(f"[WARN] 속도 초과 보호 작동 | JVEL = {data['jvel']}")
                send_torque(zero_torque)
                viewer.sync()
                sleep(0.005)
                continue

            F_task = np.zeros(3)
            F_task[axis_idx] = direction * force_mag

            tau_cmd = data["jacp0"].T @ F_task
            tau_cmd = send_torque(tau_cmd)
            last_tau_cmd = tau_cmd.copy()

            tcp_delta = data["tcp_pos"] - start_tcp
            axis_delta = tcp_delta[axis_idx]
            intended_axis_delta = direction * axis_delta
            norm_delta = np.linalg.norm(tcp_delta)

            max_axis_delta = max(max_axis_delta, abs(axis_delta))
            max_norm_delta = max(max_norm_delta, norm_delta)

            moved_by_axis = abs(axis_delta) >= MOVE_AXIS_THRESH_M
            moved_by_norm = norm_delta >= MOVE_NORM_THRESH_M

            if moved_by_axis or moved_by_norm:
                result = {
                    "axis": axis_name,
                    "direction": dir_name,
                    "force_N": force_mag,
                    "tau_cmd": last_tau_cmd.copy(),
                    "start_tcp": start_tcp.copy(),
                    "detected_tcp": data["tcp_pos"].copy(),
                    "tcp_delta": tcp_delta.copy(),
                    "axis_delta": axis_delta,
                    "intended_axis_delta": intended_axis_delta,
                    "norm_delta": norm_delta,
                }

                print("\n[DETECTED]")
                print(f"Axis Direction       : {dir_name}{axis_name}")
                print(f"Threshold Force [N]  : {force_mag:.3f}")
                print(f"Joint Torque [Nm]    : {last_tau_cmd}")
                print(f"TCP Delta [m]        : {tcp_delta}")
                print(f"Axis Delta [m]       : {axis_delta:.6f}")
                print(f"Intended Delta [m]   : {intended_axis_delta:.6f}")
                print(f"Norm Delta [m]       : {norm_delta:.6f}")
                print("--------------------------------------")

                hold_zero_torque(viewer, SETTLE_TIME)
                return result

            viewer.sync()
            sleep(0.005)

        print(
            f"[{dir_name}{axis_name}] "
            f"F = {force_mag:6.2f} N | "
            f"max_axis_delta = {max_axis_delta*1000:6.3f} mm | "
            f"max_norm_delta = {max_norm_delta*1000:6.3f} mm"
        )

    print(f"[NOT DETECTED] Axis = {dir_name}{axis_name}, up to {FORCE_MAX_N:.1f} N")
    hold_zero_torque(viewer, SETTLE_TIME)

    return {
        "axis": axis_name,
        "direction": dir_name,
        "force_N": None,
        "tau_cmd": None,
        "start_tcp": start_tcp.copy(),
        "detected_tcp": None,
        "tcp_delta": None,
        "axis_delta": None,
        "intended_axis_delta": None,
        "norm_delta": None,
    }


print("\n[Force Threshold Test]")
print("compensation=3 상태에서 task-space force를 축별로 증가시키며 움직임 시작점을 측정합니다.")
print(f"FORCE_STEP_N       = {FORCE_STEP_N} N")
print(f"FORCE_MAX_N        = {FORCE_MAX_N} N")
print(f"MOVE_AXIS_THRESH   = {MOVE_AXIS_THRESH_M * 1000:.2f} mm")
print(f"MOVE_NORM_THRESH   = {MOVE_NORM_THRESH_M * 1000:.2f} mm")
print("--------------------------------------")

try:
    with mujoco.viewer.launch_passive(m, d) as viewer:
        hold_zero_torque(viewer, 1.0)

        for axis_idx in range(3):
            for direction in TEST_DIRECTIONS:
                result = test_axis_direction(viewer, axis_idx, direction)
                results.append(result)

        print("\n======================================")
        print("[SUMMARY]")
        print("======================================")

        for result in results:
            axis = result["axis"]
            direction = result["direction"]
            force_N = result["force_N"]

            if force_N is None:
                print(f"{direction}{axis}: not detected up to {FORCE_MAX_N:.1f} N")
            else:
                print(
                    f"{direction}{axis}: "
                    f"{force_N:.3f} N | "
                    f"tau = {result['tau_cmd']} | "
                    f"delta = {result['tcp_delta']}"
                )

        print("\n[TEST DONE] zero torque 유지 중입니다. 창을 닫으면 종료됩니다.")

        while viewer.is_running():
            hold_zero_torque(viewer, 0.05)

except RuntimeError as e:
    print(e)

finally:
    try:
        robot.move_servo_t(rc, zero_torque, t1, t2, compensation=3)
    except Exception:
        pass