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


def key_callback(keycode):
    global desired_xpos_tcp, is_at_target_a

    if keycode == ord("S") or keycode == ord("s"):
        if is_at_target_a:
            desired_xpos_tcp = target_b.copy()
            is_at_target_a = False
            print(f"\n[TOGGLE] Target B로 이동합니다: {target_b}")
        else:
            desired_xpos_tcp = target_a.copy()
            is_at_target_a = True
            print(f"\n[TOGGLE] Target A로 되돌아갑니다: {target_a}")


target_a = np.array([0.1, -0.5, 0.5])
target_b = np.array([-0.1, -0.4, 0.4])

desired_xpos_tcp = target_a.copy()
is_at_target_a = True

desired_rpy = np.array([90.0, 0.0, 0.0])
R_desired = rpy_to_rotmat(*desired_rpy)

model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

if state is not None:
    d.qpos[:] = np.deg2rad(state.sdata.jnt_ang)
    mujoco.mj_forward(m, d)

tcp_site_id = m.site("tcp").id

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

K_a_mat = np.diag([200.0, 400.0, 300.0])
zeta_a_mat = np.diag([2.0, 4.0, 3.0])

K_o = 10.0
zeta_o = 4.0
ori_torque_scale = 0.3

F_ff_move_pos_max = np.array([7.75, 3.50, 12.75])
F_ff_move_neg_max = np.array([6.75, 17.75, 2.25])

F_ff_scale_far = 0.90
F_ff_scale_near = 0.35

F_ff_near_thresh = np.array([0.005, 0.005, 0.005])
F_ff_deadband = np.array([0.0015, 0.0015, 0.0015])

max_torque = 50.0
tau_rate_limit = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 10.0])

prev_target_torque = np.zeros(6)
prev_time = time()
prev_jpos = None

print("\n start")

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        state = robot_data.request_data()

        if state is None:
            continue

        if state.sdata.op_stat_collision_occur or state.sdata.op_stat_sos_flag == 4:
            print("stop")
            break

        now = time()
        loop_dt = now - prev_time
        prev_time = now

        if loop_dt <= 0.0:
            continue

        jpos = np.array(state.sdata.jnt_ang, dtype=np.float64)

        real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)

        jvel = np.zeros(6)

        if prev_jpos is not None:
            jvel = (jpos - prev_jpos) / loop_dt

        prev_jpos = jpos.copy()

        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = np.deg2rad(jvel)

        mujoco.mj_forward(m, d)

        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        jacp0 = jacp[:, 0:6]
        jacr0 = jacr[:, 0:6]

        raw_err = real_tcp_pos - desired_xpos_tcp
        err_norm = np.linalg.norm(raw_err)

        xpos_dot0 = jacp0 @ d.qvel[0:6]

        sqrt_Ka_mat = np.diag(np.sqrt(np.diag(K_a_mat)))

        F_p = K_a_mat @ raw_err
        F_d = zeta_a_mat @ sqrt_Ka_mat @ xpos_dot0

        F_ff = np.zeros(3)
        F_ff_limit_measured = np.zeros(3)
        F_ff_limit_used = np.zeros(3)
        F_ff_scale_used = np.zeros(3)

        for i in range(3):
            abs_err_i = abs(raw_err[i])

            if abs_err_i >= F_ff_deadband[i]:
                if raw_err[i] > 0.0:
                    ff_limit = F_ff_move_neg_max[i]
                else:
                    ff_limit = F_ff_move_pos_max[i]

                if abs_err_i < F_ff_near_thresh[i]:
                    ff_scale_i = F_ff_scale_near
                else:
                    ff_scale_i = F_ff_scale_far

                ff_limit_scaled = ff_scale_i * ff_limit

                F_ff_limit_measured[i] = ff_limit
                F_ff_limit_used[i] = ff_limit_scaled
                F_ff_scale_used[i] = ff_scale_i
                F_ff[i] = ff_limit_scaled * np.sign(raw_err[i])

        force0 = F_p + F_d + F_ff

        ori_err0 = (
            np.cross(R_desired[:, 0], real_tcp_R[:, 0]) +
            np.cross(R_desired[:, 1], real_tcp_R[:, 1]) +
            np.cross(R_desired[:, 2], real_tcp_R[:, 2])
        )

        w0 = jacr0 @ d.qvel[0:6]

        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)

        torque0 = - (jacp0.T @ force0) - ori_torque_scale * (jacr0.T @ F_ori_0)

        target_torque_raw = np.clip(torque0, -max_torque, max_torque)

        d_tau_max = tau_rate_limit * loop_dt
        d_tau = target_torque_raw - prev_target_torque
        d_tau = np.clip(d_tau, -d_tau_max, d_tau_max)

        target_torque = prev_target_torque + d_tau
        prev_target_torque = target_torque.copy()

        d.ctrl[0:6] = target_torque

        if np.any(np.abs(jvel) > 70):
            print("over V")
            target_torque[:] = 0.0
            prev_target_torque[:] = 0.0

        try:
            robot.move_servo_t(rc, target_torque, t1, t2, compensation=3)
            sleep(0.005)
        except Exception as e:
            print(f"제어 명령 전송 실패: {e}")

        R_err = R_desired.T @ real_tcp_R

        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        ori_err_deg = np.rad2deg(np.arccos(cos_angle))

        rpy_err = real_tcp_rpy - desired_rpy
        rpy_err = (rpy_err + 180.0) % 360.0 - 180.0

        print(f"[ERR NORM] {err_norm:.4f} | [VEL] {np.linalg.norm(xpos_dot0):.4f}")
        print(f"[POS ERR] {raw_err}")
        print(f"[F_P] {F_p} | [F_D] {F_d} | [F_FF] {F_ff}")
        print(f"[F_FF SCALE FAR] {F_ff_scale_far:.3f}")
        print(f"[F_FF SCALE NEAR] {F_ff_scale_near:.3f}")
        print(f"[F_FF SCALE USED] {F_ff_scale_used}")
        print(f"[F_FF LIMIT MEASURED] {F_ff_limit_measured}")
        print(f"[F_FF LIMIT USED] {F_ff_limit_used}")
        print(f"[REAL RPY] {real_tcp_rpy}")
        print(f"[RPY ERR DEG] {rpy_err}")
        print(f"[ORI ERR] {ori_err0}")
        print(f"[ORI ERR DEG] {ori_err_deg:.3f}")
        print(f"[ORI SCALE] {ori_torque_scale:.3f}")
        print(f"[TAR TOR RAW] {target_torque_raw}")
        print(f"[TAR TOR] {target_torque}")
        print("--------------------------------------")

        viewer.sync()