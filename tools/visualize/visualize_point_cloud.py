import argparse
from pathlib import Path
from tqdm import tqdm
from tools.data_converter.utils import read_json
from tools.visualize.point_cloud_animator import PointCloudAnimator

def animate_point_clouds(base, infrastructure_pcd_dir: Path, vehicle_pcd_dir: Path):
    # read data info
    c_data_info = read_json(base / "cooperative" / "data_info.json")

    # point cloud animator
    i_animator = PointCloudAnimator(window_name="Infrastructure Frames")
    v_animator = PointCloudAnimator(window_name="Vehicle Frames")

    for data in tqdm(c_data_info):
        # visualize point clouds
        i_animator.update(infrastructure_pcd_dir, "infrastructure", data, last_frame=(data == c_data_info[-1]))
        v_animator.update(vehicle_pcd_dir, "vehicle", data, last_frame=(data == c_data_info[-1]))

    i_animator.close()
    v_animator.close()


def main():
    # root of the dataset repository
    cwd = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser("Visualize point clouds")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example",
        help="Raw data root about SPD."
    )
    parser.add_argument(
        "--infrastructure-pcd-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example" / "infrastructure-side" / "velodyne_new",
        help="Directory containing infrastructure point clouds in PCD format",
    )
    parser.add_argument(
        "--vehicle-pcd-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example" / "vehicle-side" / "velodyne",
        help="Directory containing vehicle point clouds in PCD format",
    )
    args = parser.parse_args()

    # visualize point clouds
    animate_point_clouds(args.root_dir, args.infrastructure_pcd_dir, args.vehicle_pcd_dir)


if __name__ == "__main__":
    main()