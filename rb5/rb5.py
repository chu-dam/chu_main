from time import time
import mujoco
import mujoco.viewer
import numpy as np

# -----------------------------
# Model path
# -----------------------------
model_path = "/home/chu/manipulator_control/rb5/scene_rb5.xml"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# -----------------------------
# Initial joint position
# -----------------------------
d.qpos[:] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
d.qvel[:] = 0.0

# 초기 상태 반영
mujoco.mj_forward(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time()

    while viewer.is_running():
        t = time() - t0

        # -------------------------------------------------
        # Torque-related calculation removed
        # -------------------------------------------------
        # 기존에 있던 아래 항목들을 모두 제거했습니다.
        #
        # - mass matrix M 계산
        # - gravity torque G 계산
        # - jacobian 계산
        # - position error 계산
        # - orientation error 계산
        # - linear/angular velocity 계산
        # - virtual spring-damper force 계산
        # - orientation force 계산
        # - torque0 계산
        # - d.ctrl[0:6] 토크 입력
        # -------------------------------------------------

        current_tcp_pos = d.site("tcp").xpos

        print(
            f"현재 TCP 위치 -> "
            f"X: {current_tcp_pos[0]:.4f}, "
            f"Y: {current_tcp_pos[1]:.4f}, "
            f"Z: {current_tcp_pos[2]:.4f}"
        )

        mujoco.mj_step(m, d)
        viewer.sync()