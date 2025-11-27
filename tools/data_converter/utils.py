import os
import math
import json
import numpy as np
import open3d as o3d
from pypcd import pypcd
from pathlib import Path


def mkdir_p(path: Path):
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_files_path(path_my_dir, extention=".json"):
    path_list = []
    for (dirpath, _, filenames) in os.walk(path_my_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def write_txt(path, file):
    with open(path, "w") as f:
        f.write(file)


def read_json(path: Path) -> dict:
    """Read a JSON file and return a dictionary."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj: dict, path: Path):
    """Write a dictionary to a JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def read_pcd(pcd_path: Path) -> np.ndarray:
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    pc = o3d.io.read_point_cloud(str(pcd_path))
    return np.asarray(pc.points)


def write_pcd(path_pcd, new_points, path_save):
    pc = pypcd.PointCloud.from_path(path_pcd)
    pc.pc_data['x'] = np.array([a[0] for a in new_points])
    pc.pc_data['y'] = np.array([a[1] for a in new_points])
    pc.pc_data['z'] = np.array([a[2] for a in new_points])
    pc.save_pcd(path_save, compression='binary_compressed')


def show_pcd(path_pcd):
    pcd = read_pcd(path_pcd)
    o3d.visualization.draw_geometries([pcd])


def pcd2bin(pcd_file_path, bin_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    points_32.tofile(bin_file_path)


def inverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def trans_point(input_point, rotation, translation=None):
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]


def transform(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    
    return np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)


def get_lidar_to_camera(lidar_to_camera_path: Path):
    cfg = read_json(lidar_to_camera_path)
    return cfg['rotation'], cfg['translation']


def get_lidar_to_novatel(lidar_to_novatel_path: Path):
    cfg = read_json(lidar_to_novatel_path)
    return cfg['transform']['rotation'], cfg['transform']['translation']


def get_virtuallidar2world(path_virtuallidar2world):
    virtuallidar2world = read_json(path_virtuallidar2world)
    rotation = virtuallidar2world['rotation']
    translation = virtuallidar2world['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation