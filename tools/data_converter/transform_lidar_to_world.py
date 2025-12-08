import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pypcd import pypcd
import open3d as o3d
import tools.utils as utils
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


def write_pcd_from_points(src_pcd_path: Path, points: np.ndarray, out_pcd_path: Path):
    # reuse header from source PCD if available; otherwise create minimal
    if src_pcd_path.exists():
        pc = pypcd.PointCloud.from_path(str(src_pcd_path))
        pc.pc_data['x'] = np.array([a[0] for a in points])
        pc.pc_data['y'] = np.array([a[1] for a in points])
        pc.pc_data['z'] = np.array([a[2] for a in points])
        utils.mkdir_p(out_pcd_path.parent)
        pc.save_pcd(str(out_pcd_path), compression='binary_compressed')
    else:
        # fallback: write a simple PCD using open3d
        utils.mkdir_p(out_pcd_path.parent)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_pcd_path), pc_o3d)


def process_infrastructure(base: Path):
    """
    Process infrastructure-side data: transform lidar point clouds to world coordinates.
    """
    input_dir=base / "infrastructure-side" / "velodyne"
    output_dir=base / "infrastructure-side" / "velodyne_world"

    # create output directory
    utils.mkdir_p(output_dir)

    # read data info
    i_data_info = utils.read_json(base / "infrastructure-side" / "data_info.json")

    # for each point cloud file
    for data in tqdm(i_data_info, desc=f"Processing point clouds in {input_dir.parent.name}"):
        # frame ID
        frame_id = data["frame_id"]
        # point cloud path
        pcd_path = base / input_dir.parent.name / data["pointcloud_path"]
        # calibration path
        calib_path = base / "infrastructure-side" / data["calib_virtuallidar_to_world_path"]
        # output path
        out_pcd_path = output_dir / f"{frame_id}.pcd"

        try:
            # get rotation and translation
            rotation, translation = utils.get_rotation_translation(calib_path)

            # virtual lidar to world
            input_points = utils.read_pcd(pcd_path)

            # TODO: relative error
            output_points = utils.transform(input_points.T, translation, rotation)

            write_pcd_from_points(
                pcd_path,
                output_points.T,
                out_pcd_path
            )
        except Exception as e:
            print(f"Skipping infrastructure frame {frame_id}: {e}")


def process_vehicle(base: Path):
    """
    Process vehicle-side data: transform lidar point clouds to world coordinates.
    """
    output_dir = base / "vehicle-side" / "velodyne_world"

    # create output directory
    utils.mkdir_p(output_dir)

    # read data info
    v_data_info = utils.read_json(base / "vehicle-side" / "data_info.json")

    # for each point cloud file
    for data in tqdm(v_data_info, desc="Processing vehicle frames"):
        # frame ID
        frame_id = data["frame_id"]
        # point cloud path
        pcd_path = base / "vehicle-side" / "velodyne" / f"{frame_id}.pcd"
        # calibration path
        calib_lidar_to_novatel_path = base / "vehicle-side" / data["calib_lidar_to_novatel_path"]
        calib_novatel_to_world_path = base / "vehicle-side" / data["calib_novatel_to_world_path"]
        # output path
        out_pcd_path = output_dir / f"{frame_id}.pcd"

        try:
            rotation_lidar_to_novatel, translation_lidar_to_novatel = utils.get_rotation_translation(calib_lidar_to_novatel_path)
            rotation_novatel_to_world, translation_novatel_to_world = utils.get_rotation_translation(calib_novatel_to_world_path)

            input_points = utils.read_pcd(pcd_path)

            novatel_points = utils.transform(input_points.T, translation_lidar_to_novatel, rotation_lidar_to_novatel)
            world_points = utils.transform(novatel_points, translation_novatel_to_world, rotation_novatel_to_world)
            
            write_pcd_from_points(
                pcd_path,
                world_points.T,
                out_pcd_path
            )

        except Exception as e:
            print(f"Skipping vehicle frame {frame_id}: {e}")


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
