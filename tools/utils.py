import os
import json
import cv2
import pickle
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
    input_point = np.array(input_point).reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point


def get_lidar_to_camera(lidar_to_camera_path: Path):
    cfg = read_json(lidar_to_camera_path)
    return cfg['rotation'], cfg['translation']


def get_lidar_to_novatel(lidar_to_novatel_path: Path):
    cfg = read_json(lidar_to_novatel_path)
    return cfg['transform']['rotation'], cfg['transform']['translation']


def get_novatel_to_world(novatel_to_world_path: Path):
    cfg = read_json(novatel_to_world_path)
    return cfg['rotation'], cfg['translation']


def get_virtual_lidar_to_world(virtual_lidar_to_world_path: Path):
    cfg = read_json(virtual_lidar_to_world_path)
    return cfg['rotation'], cfg['translation']


def get_label(label):
    h = float(label[0]["h"])
    w = float(label[0]["w"])
    length = float(label[0]["l"])
    x = float(label[1]["x"])
    y = float(label[1]["y"])
    z = float(label[1]["z"])
    rotation_y = float(label[-1])
    return h, w, length, x, y, z, rotation_y


def get_lidar_to_cam(calib):
    if "Tr_velo_to_cam" in calib.keys():
        velo2cam = np.array(calib["Tr_velo_to_cam"]).reshape(3, 4)
        r_velo2cam = velo2cam[:, :3]
        t_velo2cam = velo2cam[:, 3].reshape(3, 1)
    else:
        r_velo2cam = np.array(calib["rotation"])
        t_velo2cam = np.array(calib["translation"])
    return r_velo2cam, t_velo2cam


def get_cam_calib_intrinsic(calib_path):
    my_json = read_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.zeros([3, 4])
    calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")

    return calib


def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)

        for start, end in line_indices:
            radius = 5
            color = (0, 0, 250)
            thickness = 1
            cv2.circle(img, (corners[start, 0], corners[start, 1]), radius, color, thickness)
            cv2.circle(img, (corners[end, 0], corners[end, 1]), radius, color, thickness)
            color = (0, 255, 0)
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

    return img.astype(np.uint8)


def get_rgb(img_path):
    return cv2.imread(img_path)


def points_cam_to_img(points_3d, calib_intrinsic, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    points_3d: N x 8 x 3
    calib_intrinsic: 3 x 4
    return: N x 8 x 2
    """
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)

    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(calib_intrinsic, points_4.T.swapaxes(1, 2).reshape(4, -1))
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        return np.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


def compute_corners_3d(dim, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2]
    z_corners = [h, h, 0, 0, h, h, 0, 0]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)

    return corners_3d


def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    corners_3d = compute_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)

    return corners_3d


def get_cam_8_points(labels, calib_lidar2cam_path):
    """
    Project 3D bounding boxes from LiDAR coordinates to camera coordinates.

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

    Args:
        labels (list): Iterable of labels. Each label must contain dimensions and pose
            in the order expected by get_label (h, w, l, x, y, z, rotation).
        calib_lidar2cam_path (str | pathlib.Path): Path to a JSON file containing the
            lidar-to-camera extrinsic.

    Returns:
        list[np.ndarray]: A list where each element is an 8x3 ndarray of the box corner
            coordinates expressed in the camera coordinate frame.
    """

    # read calib_lidar2cam
    calib_lidar2cam = read_json(calib_lidar2cam_path)

    # get r_velo2cam and t_velo2cam
    r_velo2cam, t_velo2cam = get_lidar_to_cam(calib_lidar2cam)

    
    camera_8_points_list = []

    for label in labels:
        h, w, l, x, y, z, yaw_lidar = get_label(label)
        z = z - h / 2

        bottom_center = [x, y, z]
        obj_size = [l, w, h]

        # 3D bounding box corners in LiDAR coordinates
        lidar_8_points = compute_box_3d(obj_size, bottom_center, yaw_lidar)
        # 3D bounding box corners in camera coordinates
        camera_8_points = r_velo2cam * np.matrix(lidar_8_points).T + t_velo2cam
        camera_8_points_list.append(camera_8_points.T)

    return camera_8_points_list


def visualize_label_in_img(camera_8_points_list, img_path, cam_intrinsic_path, save_path) -> bool:
    # image frame id
    index = img_path.name.split(".")[0]
    # calibration intrinsic
    calib_intrinsic = get_cam_calib_intrinsic(cam_intrinsic_path)
    # read image
    img = get_rgb(img_path)

    # convert camera 8 points to numpy array
    cam8points = np.array(camera_8_points_list)
    # number of bounding boxes
    num_bbox = cam8points.shape[0]
    # project 3D bounding boxes to image plane
    uv_origin = points_cam_to_img(cam8points, calib_intrinsic)
    uv_origin = (uv_origin - 1).round()

    # plot 3D bounding boxes on image
    plot_rect3d_on_img(img, num_bbox, uv_origin)
    cv2.imwrite(save_path / f"{index}.png", img)
    print(index)

    return True