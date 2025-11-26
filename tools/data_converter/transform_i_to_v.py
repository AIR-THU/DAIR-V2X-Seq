import argparse
import numpy as np
from pypcd import pypcd
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
from tools.data_converter.utils import read_json, mkdir_p
from tools.visualize.point_cloud_animator import PointCloudAnimator

"""
            virtuallidar_to_world
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
    virtual_lidar_to_world = read_json(virtual_lidar_to_world_path)

    return virtual_lidar_to_world['rotation'], virtual_lidar_to_world['translation']


def get_novatel_to_world(novatel_to_world_path: Path):
    novatel_to_world = read_json(novatel_to_world_path)

    return novatel_to_world['rotation'], novatel_to_world['translation']


def get_lidar_to_novatel(lidar_to_novatel_path: Path):
    lidar_to_novatel = read_json(lidar_to_novatel_path)
    
    return lidar_to_novatel['transform']['rotation'], lidar_to_novatel['transform']['translation']


def get_data(data_info, frame_id):
    for data in data_info:
        if data["frame_id"] == frame_id:
            return data


def transform(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point


def rev_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def transform_virtual_lidar_to_lidar(input_point: np.ndarray, virtual_lidar_to_world_path: Path, novatel_to_world_path: Path, lidar_to_novatel_path: Path, delta_x: float, delta_y: float):
    # virtual lidar to world
    rotation, translation = get_virtual_lidar_to_world(virtual_lidar_to_world_path)
    point = transform(input_point, translation, rotation)

    # world to novatel
    rotation, translation = get_novatel_to_world(novatel_to_world_path)
    new_rotation = rev_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = transform(point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = get_lidar_to_novatel(lidar_to_novatel_path)
    new_rotation = rev_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = transform(point, new_translation, new_rotation) + np.array([delta_x, delta_y, 0]).reshape(3, 1)
    point = point.T
    
    return point


def read_pcd(pcd_path):
    pointpillar = o3d.io.read_point_cloud(pcd_path)

    return np.asarray(pointpillar.points)


def write_pcd(pcd_path, new_points, path_save):
    pc = pypcd.PointCloud.from_path(pcd_path)
    pc.pc_data['x'] = np.array([a[0] for a in new_points])
    pc.pc_data['y'] = np.array([a[1] for a in new_points])
    pc.pc_data['z'] = np.array([a[2] for a in new_points])
    pc.save_pcd(path_save, compression='binary_compressed')


def set_new_label(infrastructure_pcd_path: Path, virtual_lidar_to_world_path: Path, novatel_to_world_path: Path, lidar_to_novatel_path: Path, delta_x: float, delta_y: float, save_path: Path):
    # (n, 3)
    points = read_pcd(infrastructure_pcd_path)
    # (n, 3)
    new_points = transform_virtual_lidar_to_lidar(points.T, virtual_lidar_to_world_path, novatel_to_world_path, lidar_to_novatel_path, delta_x, delta_y)
    # save point cloud
    write_pcd(infrastructure_pcd_path, new_points, save_path)


def transform_i_to_v(base: Path, output_dir: Path):
    # make directory for output label
    mkdir_p(output_dir)

    # read data info
    c_data_info = read_json(base / "cooperative" / "data_info.json")
    i_data_info = read_json(base / "infrastructure-side" / "data_info.json")
    v_data_info = read_json(base / "vehicle-side" / "data_info.json")

    # point cloud animator
    i_animator = PointCloudAnimator(window_name="Infrastructure Frames")
    v_animator = PointCloudAnimator(window_name="Vehicle Frames")

    for data in tqdm(c_data_info):
        # get data according to frame id
        i_data = get_data(i_data_info, data["infrastructure_frame"])
        v_data = get_data(v_data_info, data["vehicle_frame"])

        # transform point cloud from infrastructure to vehicle
        set_new_label(
            infrastructure_pcd_path=base / "infrastructure-side" / "velodyne" / f"{data['infrastructure_frame']}.pcd",
            virtual_lidar_to_world_path=base / "infrastructure-side" / i_data["calib_virtuallidar_to_world_path"],
            novatel_to_world_path=base / "vehicle-side" / v_data["calib_novatel_to_world_path"],
            lidar_to_novatel_path=base / "vehicle-side" / v_data["calib_lidar_to_novatel_path"],
            delta_x=data["system_error_offset"]["delta_x"],
            delta_y=data["system_error_offset"]["delta_y"],
            save_path=output_dir / f"{data['infrastructure_frame']}.pcd"
        )

        # visualize point clouds
        i_animator.update(output_dir, "infrastructure", data, last_frame=(data == c_data_info[-1]))
        v_animator.update(base / "vehicle-side" / "velodyne", "vehicle", data, last_frame=(data == c_data_info[-1]))

    i_animator.close()
    v_animator.close()


def main():
    parser = argparse.ArgumentParser("Convert the point cloud from infrastructure to ego-vehicle")
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Raw data root about SPD."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The data root where the data with ego-vehicle coordinate is generated"
    )
    args = parser.parse_args()

    # convert point cloud from infrastructure to vehicle
    transform_i_to_v(args.root_dir, args.output_dir)

if __name__ == "__main__":
    main()