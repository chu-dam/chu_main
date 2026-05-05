import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    # 이 py 파일이 있는 폴더 = 현재 실험 폴더
    base_dir = Path(__file__).resolve().parent

    # 같은 폴더 안의 csv 전부 읽기
    csv_files = sorted(base_dir.glob("*.csv"))

    if len(csv_files) == 0:
        print("[ERROR] CSV 파일이 없습니다.")
        return

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.get_cmap("tab20", max(len(csv_files), 1))

    goal_plotted = False

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)

        required_cols = ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
        if not all(col in df.columns for col in required_cols):
            print(f"[SKIP] 필요한 컬럼이 없음: {csv_file.name}")
            continue

        color = cmap(i % 20)

        # pt 번호 추출
        match = re.search(r"pt(\d+)", csv_file.name)
        if match:
            pt_label = f"pt{int(match.group(1)):02d}"
        else:
            pt_label = f"traj{i+1:02d}"

        x = df["tcp_x_m"].to_numpy()
        y = df["tcp_y_m"].to_numpy()
        z = df["tcp_z_m"].to_numpy()

        # 궤적 전체를 point cloud처럼 점으로 표시
        ax.scatter(x, y, z, s=8, alpha=0.55, color=color, label=pt_label)

        # 시작점
        ax.scatter(
            x[0], y[0], z[0],
            s=80, marker="^", color=color, edgecolors="black"
        )

        # 종료점
        ax.scatter(
            x[-1], y[-1], z[-1],
            s=80, marker="x", color=color
        )

        # 시작점 라벨
        ax.text(x[0], y[0], z[0], pt_label, fontsize=7)

        # 목표점은 한 번만 표시
        if not goal_plotted and all(col in df.columns for col in ["goal_x_m", "goal_y_m", "goal_z_m"]):
            gx = df["goal_x_m"].iloc[0]
            gy = df["goal_y_m"].iloc[0]
            gz = df["goal_z_m"].iloc[0]

            ax.scatter(
                gx, gy, gz,
                s=250, marker="*", color="red", edgecolors="black", label="goal"
            )
            ax.text(gx, gy, gz, "GOAL", fontsize=10)
            goal_plotted = True

    ax.set_title(f"3D TCP Point Cloud\n{base_dir.name}")
    ax.set_xlabel("TCP X [m]")
    ax.set_ylabel("TCP Y [m]")
    ax.set_zlabel("TCP Z [m]")

    ax.view_init(elev=22, azim=-55)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        markerscale=2
    )

    plt.tight_layout()

    save_path = base_dir / "point_cloud_plot.png"
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    print(f"[SAVE] {save_path}")

    plt.show()


if __name__ == "__main__":
    main()