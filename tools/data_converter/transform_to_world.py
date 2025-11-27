import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pypcd import pypcd
import open3d as o3d
from tools.data_converter.utils import read_json, mkdir_p
from tools.visualize.visualize_point_cloud import animate_point_clouds

"""
            virtual_lidar_to_world
inf lidar ------------------------> world coordinate <===== label_world
              relative error             ^
                                         |
                                         |
                                         |
                                         |
                                         | novatel_to_world
                                         |
                                         |
                                         |
                                         |
              lidar_to_novatel           |
veh lidar --------------------------> novatel

"""


"""
   (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

   ..code - block:: none


                        front z
                             /
                            /
              (x0, y0, z1) + -----------  + (x1, y0, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                        |  /      .   |  /
                        | / origin    | /
           (x0, y1, z0) + ----------- + -------> x right
                        |             (x1, y1, z0)
                        |
                        v
                   down y

   """


def get_virtual_lidar_to_world(virtual_lidar_to_world_path: Path):
    cfg = read_json(virtual_lidar_to_world_path)
    return cfg['rotation'], cfg['translation']


def get_novatel_to_world(novatel_to_world_path: Path):
    cfg = read_json(novatel_to_world_path)
    return cfg['rotation'], cfg['translation']


def get_lidar_to_novatel(lidar_to_novatel_path: Path):
    cfg = read_json(lidar_to_novatel_path)
    return cfg['transform']['rotation'], cfg['transform']['translation']


def rev_matrix(R):
    R = np.matrix(R)
    return np.array(R.I)


def transform_points(points: np.ndarray, rotation, translation) -> np.ndarray:
    """Apply rotation and translation to Nx3 points. rotation: (3,3), translation: (3,) or (3,1)."""
    pts = np.asarray(points)
    if pts.size == 0:
        return pts
    # ensure shape (3, N)
    p = pts.reshape(-1, 3).T  # (3,N)
    R = np.array(rotation).reshape(3, 3)
    t = np.array(translation).reshape(3, 1)
    out = (R @ p) + t
    return out.T


def read_pcd(pcd_path: Path) -> np.ndarray:
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    pc = o3d.io.read_point_cloud(str(pcd_path))
    return np.asarray(pc.points)


def write_pcd_from_points(src_pcd_path: Path, points: np.ndarray, out_pcd_path: Path):
    # reuse header from source PCD if available; otherwise create minimal
    if src_pcd_path.exists():
        pc = pypcd.PointCloud.from_path(str(src_pcd_path))
        pc.pc_data['x'] = np.array([a[0] for a in points])
        pc.pc_data['y'] = np.array([a[1] for a in points])
        pc.pc_data['z'] = np.array([a[2] for a in points])
        mkdir_p(out_pcd_path.parent)
        pc.save_pcd(str(out_pcd_path), compression='binary_compressed')
    else:
        # fallback: write a simple PCD using open3d
        mkdir_p(out_pcd_path.parent)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_pcd_path), pc_o3d)


def process_infrastructure(base: Path):
    """
    Process infrastructure-side data: transform lidar point clouds to world coordinates.
    """
    i_output_dir = base / "infrastructure-side" / "velodyne_world"
    mkdir_p(i_output_dir)

    i_data_info = read_json(base / "infrastructure-side" / "data_info.json")

    for i_entry in tqdm(i_data_info, desc="Processing infrastructure frames"):
        infra_frame = i_entry["frame_id"]
        try:
            infra_pcd = base / "infrastructure-side" / "velodyne" / f"{infra_frame}.pcd"
            virt_calib = base / "infrastructure-side" / i_entry["calib_virtuallidar_to_world_path"]

            rotation, translation = get_virtual_lidar_to_world(virt_calib)

            pts = read_pcd(infra_pcd)
            pts_world = transform_points(pts, rotation, translation)

            write_pcd_from_points(
                infra_pcd,
                pts_world,
                out_pcd_path=i_output_dir / f"{infra_frame}.pcd"
            )
        except Exception as e:
            print(f"Skipping infra frame {infra_frame}: {e}")


def process_vehicle(base: Path):
    """
    Process vehicle-side data: transform lidar point clouds to world coordinates.
    """
    v_output_dir = base / "vehicle-side" / "velodyne_world"
    mkdir_p(v_output_dir)

    v_data_info = read_json(base / "vehicle-side" / "data_info.json")

    for v_entry in tqdm(v_data_info, desc="Processing vehicle frames"):
        veh_frame = v_entry["frame_id"]
        try:
            veh_pcd = base / "vehicle-side" / "velodyne" / f"{veh_frame}.pcd"
            lidar2novatel_calib = base / "vehicle-side" / v_entry["calib_lidar_to_novatel_path"]
            novatel_calib = base / "vehicle-side" / v_entry["calib_novatel_to_world_path"]

            rotation_l2n, translation_l2n = get_lidar_to_novatel(lidar2novatel_calib)
            rotation_n2w, translation_n2w = get_novatel_to_world(novatel_calib)

            pts = read_pcd(veh_pcd)
            pts_novatel = transform_points(pts, rotation_l2n, translation_l2n)
            pts_world = transform_points(pts_novatel, rotation_n2w, translation_n2w)

            write_pcd_from_points(
                veh_pcd,
                pts_world,
                out_pcd_path=v_output_dir / f"{veh_frame}.pcd"
            )
        except Exception as e:
            print(f"Skipping vehicle frame {veh_frame}: {e}")


def process_dataset(base: Path):
    """
    Processes both infrastructure and vehicle datasets to transform point clouds to world coordinates.
    """
    process_infrastructure(base)
    process_vehicle(base)


def main():
    # root of the dataset repository
    cwd = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Transform infra and vehicle lidars into world coordinates and save them")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example",
        help="Raw data root (contains cooperative, infrastructure-side, vehicle-side)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, animate point clouds after conversion."
    )
    args = parser.parse_args()

    base = args.root_dir

    process_dataset(base)

    if args.visualize:
        # animate the point clouds for infrastructure and vehicle
        animate_point_clouds(
            base,
            base / "infrastructure-side" / "velodyne_world",
            base / "vehicle-side" / "velodyne_world"
        )


if __name__ == '__main__':
    main()
