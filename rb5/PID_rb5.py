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
    
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    
    return Rz @ Ry @ Rx


def rb_get_tcp_pose(current_state):
    if hasattr(current_state.sdata, 'tcp'):
        tcp_info = np.array(current_state.sdata.tcp, dtype=np.float64)
    elif hasattr(current_state.sdata, 'tcp_pos'):
        tcp_info = np.array(current_state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(current_state.sdata, 'cur_pos'):
        tcp_info = np.array(current_state.sdata.cur_pos, dtype=np.float64)
    else:
        return np.zeros(3), np.eye(3), np.zeros(3)

    return tcp_info[0:3] * 0.001, rpy_to_rotmat(*tcp_info[3:6]), tcp_info[3:6]


def key_callback(keycode):
    if keycode == ord('S') or keycode == ord('s'):
        global desired_xpos_tcp, is_at_target_a, prev_F_i, prev_target_torque

        xpos_err_sum[:] = 0.0
        prev_F_i[:] = 0.0
        prev_target_torque[:] = 0.0
        
        if is_at_target_a:
            desired_xpos_tcp = target_b
            is_at_target_a = False
            print(f"\n[TOGGLE] Target B로 이동합니다: {target_b}")
        else:
            desired_xpos_tcp = target_a
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

K_i_vec = np.array([220.0, 240.0, 280.0])
K_i_fine_vec = np.array([190.0, 230.0, 260.0])

xpos_err_sum = np.zeros(3)

integral_limit = np.array([0.05, 0.05, 0.06])

axis_integral_thresh = np.array([0.04, 0.04, 0.055])
axis_deadband = np.array([0.001, 0.001, 0.001])
axis_vel_thresh = np.array([0.05, 0.05, 0.05])

integral_leak = 0.98

F_i_max = np.array([11.0, 12.0, 16.0])
F_i_fine_max = np.array([6.0, 13.0, 16.0])

fine_err_thresh = 0.020
fine_min_scale = 0.30

prev_F_i = np.zeros(3)
prev_target_torque = np.zeros(6)

F_i_rate_limit = np.array([30.0, 35.0, 45.0])
F_i_rate_limit_fine = np.array([8.0, 10.0, 12.0])

tau_rate_limit = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 10.0])
tau_rate_limit_fine = np.array([5.0, 5.0, 5.0, 4.0, 4.0, 3.0])

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

        jpos = np.array(state.sdata.jnt_ang)
        real_tcp_pos, real_tcp_R, real_tcp_rpy = rb_get_tcp_pose(state)
        
        jvel = np.zeros(6)
        if prev_jpos is not None:
            jvel = (jpos - prev_jpos) / loop_dt
        prev_jpos = jpos
        
        d.qpos[:] = np.deg2rad(jpos)
        d.qvel[:] = np.deg2rad(jvel)
        
        mujoco.mj_forward(m, d)
        
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        jacp0 = jacp[:, 0:6]
        jacr0 = jacr[:, 0:6]

        raw_err = real_tcp_pos - desired_xpos_tcp
        xpos_err0 = raw_err

        xpos_dot0 = jacp0 @ d.qvel[0:6]
        sqrt_Ka_mat = np.diag(np.sqrt(np.diag(K_a_mat)))

        F_p = K_a_mat @ xpos_err0
        F_d = zeta_a_mat @ sqrt_Ka_mat @ xpos_dot0

        F_p_max = np.array([20.0, 25.0, 20.0])
        F_p = np.clip(F_p, -F_p_max, F_p_max)

        err_norm = np.linalg.norm(raw_err)
        vel_norm = np.linalg.norm(xpos_dot0)

        for i in range(3):
            if abs(raw_err[i]) < axis_deadband[i]:
                xpos_err_sum[i] *= integral_leak
            elif abs(raw_err[i]) < axis_integral_thresh[i] and abs(xpos_dot0[i]) < axis_vel_thresh[i]:
                xpos_err_sum[i] += raw_err[i] * loop_dt
            else:
                xpos_err_sum[i] *= integral_leak

        xpos_err_sum = np.clip(xpos_err_sum, -integral_limit, integral_limit)

        if err_norm < fine_err_thresh:
            axis_fine_thresh = np.array([0.010, 0.010, 0.010])
            axis_fine_min_scale = np.array([0.4, 0.35, 0.40])

            fine_scale_vec = np.clip(
                np.abs(raw_err) / axis_fine_thresh,
                axis_fine_min_scale,
                1.0
            )

            fine_scale = np.linalg.norm(raw_err) / fine_err_thresh
            fine_scale = np.clip(fine_scale, fine_min_scale, 1.0)

            K_i_active = K_i_fine_vec
            F_i_max_active = F_i_fine_max * fine_scale_vec
            F_i_rate_active = F_i_rate_limit_fine
        else:
            fine_scale_vec = np.ones(3)
            fine_scale = 1.0
            K_i_active = K_i_vec
            F_i_max_active = F_i_max
            F_i_rate_active = F_i_rate_limit

        F_i_target = K_i_active * xpos_err_sum
        F_i_target = np.clip(F_i_target, -F_i_max_active, F_i_max_active)

        dF_i_max = F_i_rate_active * loop_dt
        dF_i = F_i_target - prev_F_i
        dF_i = np.clip(dF_i, -dF_i_max, dF_i_max)

        F_i = prev_F_i + dF_i
        prev_F_i = F_i.copy()

        ori_err0 = (
            np.cross(R_desired[:, 0], real_tcp_R[:, 0]) +
            np.cross(R_desired[:, 1], real_tcp_R[:, 1]) +
            np.cross(R_desired[:, 2], real_tcp_R[:, 2])
        )
        
        w0 = jacr0 @ d.qvel[0:6]

        force0 = F_p + F_d + F_i
        F_ori_0 = (K_o * ori_err0) + (zeta_o * np.sqrt(K_o) * w0)
        
        torque0 = - (jacp0.T @ force0) - (jacr0.T @ F_ori_0)

        max_torque = 50.0
        target_torque_raw = np.clip(torque0, -max_torque, max_torque)

        if err_norm < fine_err_thresh:
            tau_rate_active = tau_rate_limit_fine
        else:
            tau_rate_active = tau_rate_limit

        d_tau_max = tau_rate_active * loop_dt
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

        print(f"[Desired POS] {desired_xpos_tcp}")
        print(f"[REAL POS] {real_tcp_pos}")
        print(f"[POS ERR] {raw_err}")
        print(f"[ERR NORM] {err_norm:.4f}")
        print(f"[VEL NORM] {vel_norm:.4f}")
        print(f"[DES RPY] {desired_rpy}")
        print(f"[REAL RPY] {real_tcp_rpy}")
        print(f"[RPY ERR DEG] {rpy_err}")
        print(f"[F_P] {F_p}")
        print(f"[F_D] {F_d}")
        print(f"[F_I_TARGET] {F_i_target}")
        print(f"[F_I] {F_i}")
        print(f"[X_INT] {xpos_err_sum}")
        print(f"[FINE SCALE] {fine_scale:.3f}")
        print(f"[F_I_MAX_ACTIVE] {F_i_max_active}")
        print(f"[TAR TOR RAW] {target_torque_raw}")
        print(f"[TAR TOR] {target_torque}")
        print(f"[FINE SCALE VEC] {fine_scale_vec}")
        print("--------------------------------------")

        viewer.sync()