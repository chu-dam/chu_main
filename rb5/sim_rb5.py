from time import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np


def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # ZYX yaw-pitch-roll
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr]
    ])

    return Rz @ Ry @ Rx


def orientation_error(R_des, R_cur):
    e = 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0]) +
        np.cross(R_cur[:, 1], R_des[:, 1]) +
        np.cross(R_cur[:, 2], R_des[:, 2])
    )
    return e


target_xpos_tcp = np.array([0.11, -0.502, 0.493])


target_rpy_tcp = np.array([90.0, 0.0, 0.0])
target_R_tcp = rpy_to_rotmat(*target_rpy_tcp)


model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)



d.qpos[:] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
d.qvel[:] = 0.0

mujoco.mj_forward(m, d)


tcp_site_id = m.site("tcp").id

hold_xpos_tcp = d.site("tcp").xpos.copy()
hold_R_tcp = d.site(tcp_site_id).xmat.reshape(3, 3).copy()

print("[INIT TCP POS]", hold_xpos_tcp)
print("[TARGET TCP POS]", target_xpos_tcp)
print("[TARGET TCP RPY]", target_rpy_tcp)
print("Press 's' in the MuJoCo viewer window to start movement.")



state = {
    "started": False
}


def key_callback(keycode):
    if keycode == ord('S') or keycode == ord('s'):
        state["started"] = True
        print("\n[KEY] S pressed -> movement started\n")


G = np.zeros(m.nv, dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)


# D_pos: N/(m/s)
K_pos = np.diag([40.0, 40.0, 40.0])
D_pos = np.diag([20.0, 20.0, 20.0])


K_ori = np.diag([10.0, 10.0, 10.0])
D_ori = np.diag([1.0, 1.0, 1.0])

max_total_torque = 150.0
max_ori_torque = 30.0



with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=key_callback
) as viewer:

    t0 = time()

    while viewer.is_running():
        t = time() - t0


        if state["started"]:
            desired_xpos_tcp = target_xpos_tcp.copy()
            desired_R_tcp = target_R_tcp.copy()
            mode = "MOVE"
        else:
            desired_xpos_tcp = hold_xpos_tcp.copy()
            desired_R_tcp = hold_R_tcp.copy()
            mode = "WAIT"


        # 1. Gravity compensation torque G(q)
        qvel_backup = deepcopy(d.qvel)

        # 순수 중력항만 계산하기 위해 속도 0으로 설정
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        mujoco.mj_rne(m, d, 0, G)

        # 원래 속도 복구
        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)

        gravity_torque = G[0:6]


        # 2. TCP Jacobian
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)

        Jp = jacp[:, 0:6]
        Jr = jacr[:, 0:6]


        # 3. Current TCP state
        current_xpos_tcp = d.site("tcp").xpos.copy()
        current_R_tcp = d.site(tcp_site_id).xmat.reshape(3, 3).copy()

        current_xvel_tcp = Jp @ d.qvel[0:6]
        current_w_tcp = Jr @ d.qvel[0:6]


        # 4. Position Virtual Spring-Damper
        xpos_err = desired_xpos_tcp - current_xpos_tcp

        # desired linear velocity = 0
        xvel_err = -current_xvel_tcp

        F_vsd = K_pos @ xpos_err + D_pos @ xvel_err

        tau_pos = Jp.T @ F_vsd


        # 5. Orientation Virtual Spring-Damper
        ori_err = orientation_error(desired_R_tcp, current_R_tcp)

        # desired angular velocity = 0
        w_err = -current_w_tcp

        M_vsd = K_ori @ ori_err + D_ori @ w_err

        tau_ori = Jr.T @ M_vsd

        # 자세 토크가 너무 커지는 것을 방지
        tau_ori = np.clip(tau_ori, -max_ori_torque, max_ori_torque)

        # 6. Total torque
        torque0 = gravity_torque + tau_pos + tau_ori

        d.ctrl[0:6] = np.clip(torque0, -max_total_torque, max_total_torque)

        print(
            f"[MODE] {mode} | "
            f"[TCP] "
            f"X: {current_xpos_tcp[0]:.6f}, "
            f"Y: {current_xpos_tcp[1]:.6f}, "
            f"Z: {current_xpos_tcp[2]:.6f}"
        )
        print(f"[DES POS] {desired_xpos_tcp}")
        print(f"[POS ERR] {xpos_err}")
        print(f"[F_VSD] {F_vsd}")
        print(f"[ORI ERR] {ori_err}")
        print(f"[M_VSD] {M_vsd}")
        print(f"[TAU_POS] {tau_pos}")
        print(f"[TAU_ORI] {tau_ori}")
        print("--------------------------------------")

        mujoco.mj_step(m, d)
        viewer.sync()