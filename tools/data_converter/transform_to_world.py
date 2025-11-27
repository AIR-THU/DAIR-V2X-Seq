import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pypcd import pypcd
import open3d as o3d
from tools.data_converter.utils import read_json, mkdir_p

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


def process_dataset(root: Path, out_root: Path):
    root = Path(root)
    out_root = Path(out_root)
    mkdir_p(out_root)

    # read data_info
    c_data_info = read_json(root / "cooperative" / "data_info.json")
    i_data_info = read_json(root / "infrastructure-side" / "data_info.json")
    v_data_info = read_json(root / "vehicle-side" / "data_info.json")

    for data in tqdm(c_data_info, desc="frames"):
        infra_frame = data["infrastructure_frame"]
        veh_frame = data["vehicle_frame"]

        # find calib entries
        i_entry = next((d for d in i_data_info if d["frame_id"] == infra_frame), None)
        v_entry = next((d for d in v_data_info if d["frame_id"] == veh_frame), None)

        # INFRA: transform infra virtual lidar -> world
        try:
            infra_pcd = root / "infrastructure-side" / "velodyne" / f"{infra_frame}.pcd"
            virt_calib = root / "infrastructure-side" / i_entry["calib_virtuallidar_to_world_path"]
            rot_v, trans_v = get_virtual_lidar_to_world(virt_calib)
            pts = read_pcd(infra_pcd)
            pts_world = transform_points(pts, rot_v, trans_v)
            out_path = out_root / "infrastructure-side" / "velodyne_world" / f"{infra_frame}.pcd"
            write_pcd_from_points(infra_pcd, pts_world, out_path)
        except Exception as e:
            print(f"Skipping infra frame {infra_frame}: {e}")

        # VEHICLE: transform vehicle lidar -> novatel -> world
        try:
            veh_pcd = root / "vehicle-side" / "velodyne" / f"{veh_frame}.pcd"
            lidar2novatel_calib = root / "vehicle-side" / v_entry["calib_lidar_to_novatel_path"]
            novatel_calib = root / "vehicle-side" / v_entry["calib_novatel_to_world_path"]

            rot_l2n, trans_l2n = get_lidar_to_novatel(lidar2novatel_calib)
            rot_n2w, trans_n2w = get_novatel_to_world(novatel_calib)

            pts_v = read_pcd(veh_pcd)
            pts_novatel = transform_points(pts_v, rot_l2n, trans_l2n)
            pts_world_v = transform_points(pts_novatel, rot_n2w, trans_n2w)
            out_path_v = out_root / "vehicle-side" / "velodyne_world" / f"{veh_frame}.pcd"
            write_pcd_from_points(veh_pcd, pts_world_v, out_path_v)
        except Exception as e:
            print(f"Skipping vehicle frame {veh_frame}: {e}")


def main():
    p = argparse.ArgumentParser(description="Transform infra and vehicle lidars into world coordinates and save them")
    p.add_argument("--root-dir", type=Path, required=True, help="Raw data root (contains cooperative, infrastructure-side, vehicle-side)")
    p.add_argument("--output-dir", type=Path, required=True, help="Output root where world pcds will be saved")
    args = p.parse_args()

    process_dataset(args.root_dir, args.output_dir)


if __name__ == '__main__':
    main()
