import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math
import argparse
from tools.data_converter.utils import read_json, write_json, mkdir_p

"""
            virtuallidar2world
inf lidar ------------------------> world coordinate <===== label_world
              relative error             ^
                                         |
                                         |
                                         |
                                         |
                                         | novatel2world
                                         |
                                         |
                                         |
                                         |
                 lidar2novatel           |
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


def get_3d_location(data_label_world):
    x = data_label_world["3d_location"]["x"]
    y = data_label_world["3d_location"]["y"]
    z = data_label_world["3d_location"]["z"]
    point = np.array([x, y, z]).reshape(3, 1)
    return point


def write_3d_location(point):
    x = float(point[0])
    y = float(point[1])
    z = float(point[2])
    return x, y, z


def get_3d_8points(obj_size, yaw_lidar, center_lidar):
    lidar_r = np.matrix(
        [
            [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
            [0, 0, 1],
        ]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T


def get_world_8_points(data_label_world):
    size = data_label_world["3d_dimensions"]
    pos = data_label_world["3d_location"]
    box = get_3d_8points(
        [float(size["l"]), float(size["w"]), float(size["h"])],
        float(data_label_world["rotation"]),
        [float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(size["h"]) / 2],
    ).tolist()

    points = []
    for i in range(8):
        point = np.array(box[i]).reshape(3, 1)
        points.append(point)
    return points


def write_world_8_points(points):
    my_points = []
    for i in range(8):
        point = np.array(points[i]).flatten().tolist()
        my_points.append(point)
    return my_points


def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)

    return novatel2world["rotation"], novatel2world["translation"]


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel["transform"]["rotation"]
    translation = lidar2novatel["transform"]["translation"]
    return rotation, translation


def get_data(data_info):
    for data in data_info:
        path_pcd = data["pointcloud_path"]
        name1 = os.path.split(path_pcd)[-1]
        name2 = os.path.split(data["pointcloud_path"])[-1]
        if name1 == name2:
            return data


def trans(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    return output_point


def rev_matrix(R):
    R = np.matrix(R)
    rev_R = R.I

    return np.array(rev_R)


def trans_point_world2v(point, path_novatel2world, path_lidar2novatel):
    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel)
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation)

    return point.reshape(1, 3).tolist()[0]


def label_world2v(path_cooperative_label, path_novatel2world, path_lidar2novatel):
    # read cooperative label
    label_world = read_json(path_cooperative_label)

    new_label = []

    for entry in label_world:
        # get 3d location
        my_3d_point = get_3d_location(entry)

        # transform 3d location from world to vehicle lidar coordinate
        new_3d_point = trans_point_world2v(my_3d_point, path_novatel2world, path_lidar2novatel)

        # get 8 points in world coordinate
        world_8_points = get_world_8_points(entry)

        my_world_8_points = []

        for j in range(8):
            # transform 8 points from world to vehicle lidar coordinate
            point = trans_point_world2v(world_8_points[j], path_novatel2world, path_lidar2novatel)
            my_world_8_points.append(point)

        # write 8 points in vehicle lidar coordinate
        new_world_8_points = write_world_8_points(my_world_8_points)

        l = entry["3d_dimensions"]["l"]
        w = entry["3d_dimensions"]["w"]
        rotation = get_rotation(new_world_8_points, new_3d_point, l, w)

        entry["3d_location"]["x"] = new_3d_point[0]
        entry["3d_location"]["y"] = new_3d_point[1]
        entry["3d_location"]["z"] = new_3d_point[2]
        entry["rotation"] = rotation
        entry["world_8_points"] = new_world_8_points

        new_label.append(entry)
        
    return new_label


def get_rotation(world_8_points, my_3d_point, length, w):
    x = my_3d_point[0]
    a = world_8_points[0][0]
    b = world_8_points[1][0]
    c = world_8_points[2][0]
    r_tan = ((b + c - 2 * x) * length) / ((a + b - 2 * x) * w)
    return math.atan(r_tan)


def gen_new_label(path_cooperative_label, path_novatel2world, path_lidar2novatel, path_save):
    new_label = label_world2v(path_cooperative_label, path_novatel2world, path_lidar2novatel)
    write_json(path_save, new_label)


def get_world2v(path_c: Path, path_dest: Path):
    # make directory for output label
    mkdir_p(path_dest)

    # read data info
    c_data_info = read_json(path_c / "cooperative" / "data_info.json")
    v_data_info = read_json(path_c / "vehicle-side" / "data_info.json")

    for data in tqdm(c_data_info):
        # cooperative label path
        path_cooperative_label = path_c / "cooperative" / "label" / f"{data['vehicle_frame']}.json"

        # vehicle data
        v_data = get_data(v_data_info)

        # vehicle calibration path
        path_novatel2world = path_c / "vehicle-side" / v_data["calib_novatel_to_world_path"]
        path_lidar2novatel = path_c / "vehicle-side" / v_data["calib_lidar_to_novatel_path"]

        # generate new label in vehicle lidar coordinate
        gen_new_label(path_cooperative_label, path_novatel2world, path_lidar2novatel, path_dest / f"{data['vehicle_frame']}.json")


def main():
    parser = argparse.ArgumentParser("Generate label from world coordinate to vehicle lidar coordinate.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("./data/cooperative-vehicle-infrastructure"),
        help="Raw data root about DAIR-V2X-C.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./label_new"),
        help="The label root where the label is in vehicle lidar coordinate system.",
    )
    args = parser.parse_args()

    # generate label in vehicle lidar coordinate
    get_world2v(args.root_dir, args.output_dir)

if __name__ == "__main__":
    main()