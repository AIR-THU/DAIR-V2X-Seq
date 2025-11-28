import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import tools.utils as utils

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


def get_data(data_info):
    for data in data_info:
        path_pcd = data["pointcloud_path"]
        name1 = os.path.split(path_pcd)[-1]
        name2 = os.path.split(data["pointcloud_path"])[-1]
        if name1 == name2:
            return data


def transform_world_to_lidar(point, novatel_to_world_path, lidar_to_novatel_path):
    # world to novatel
    rotation, translation = utils.get_rotation_translation(novatel_to_world_path)
    new_rotation = utils.inverse_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = utils.transform(point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = utils.get_rotation_translation(lidar_to_novatel_path)
    new_rotation = utils.inverse_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = utils.transform(point, new_translation, new_rotation)

    return point.reshape(1, 3).tolist()[0]


def label_world_to_v(cooperative_label_path: Path, novatel_to_world_path: Path, lidar_to_novatel_path: Path):
    # read cooperative label
    label_world = utils.read_json(cooperative_label_path)

    new_label = []

    for entry in label_world:
        # get 3d location
        my_3d_point = utils.get_3d_location(entry)

        # transform 3d location from world to vehicle lidar coordinate
        new_3d_point = transform_world_to_lidar(my_3d_point, novatel_to_world_path, lidar_to_novatel_path)

        # get 8 points in world coordinate
        world_8_points = utils.get_world_8_points(entry)

        my_world_8_points = []

        for j in range(8):
            # transform 8 points from world to vehicle lidar coordinate
            point = transform_world_to_lidar(world_8_points[j], novatel_to_world_path, lidar_to_novatel_path)
            my_world_8_points.append(point)

        # write 8 points in vehicle lidar coordinate
        new_world_8_points = utils.write_world_8_points(my_world_8_points)

        l = entry["3d_dimensions"]["l"]
        w = entry["3d_dimensions"]["w"]
        rotation = utils.get_rotation(new_world_8_points, new_3d_point, l, w)

        entry["3d_location"]["x"] = new_3d_point[0]
        entry["3d_location"]["y"] = new_3d_point[1]
        entry["3d_location"]["z"] = new_3d_point[2]
        entry["rotation"] = rotation
        entry["world_8_points"] = new_world_8_points

        new_label.append(entry)
        
    return new_label


def transform_c_to_v(cooperative_label_path: Path, novatel_to_world_path: Path, lidar_to_novatel_path: Path, save_path: Path):
    new_label = label_world_to_v(cooperative_label_path, novatel_to_world_path, lidar_to_novatel_path)
    utils.write_json(new_label, save_path)


def process_dataset(base: Path, output_dir: Path):
    # make directory for output label
    utils.mkdir_p(output_dir)

    # read data info
    c_data_info = utils.read_json(base / "cooperative" / "data_info.json")
    v_data_info = utils.read_json(base / "vehicle-side" / "data_info.json")

    for data in tqdm(c_data_info):
        # vehicle data
        v_data = get_data(v_data_info)

        # generate new label in vehicle lidar coordinate
        transform_c_to_v(
            cooperative_label_path=base / "cooperative" / "label" / f"{data['vehicle_frame']}.json",
            novatel_to_world_path=base / "vehicle-side" / v_data["calib_novatel_to_world_path"],
            lidar_to_novatel_path=base / "vehicle-side" / v_data["calib_lidar_to_novatel_path"],
            save_path=output_dir / f"{data['vehicle_frame']}.json"
        )


def main():
    # root of the dataset repository
    cwd = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser("Generate label from cooperative coordinate to vehicle lidar coordinate.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example",
        help="Raw data root about DAIR-V2X-C.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cwd / "dataset" / "v2x-seq-spd" / "V2X-Seq-SPD-Example" / "cooperative" / "label_vehicle",
        help="The label root where the label is in vehicle lidar coordinate system.",
    )
    args = parser.parse_args()

    # generate label in vehicle lidar coordinate
    process_dataset(args.root_dir, args.output_dir)

if __name__ == "__main__":
    main()