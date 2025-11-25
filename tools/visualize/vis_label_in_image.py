import argparse
from pathlib import Path
from vis_utils import get_cam_8_points, read_json, vis_label_in_img


def vis_label_in_image(root_path: Path, output_path: Path):
    # get data_info
    data_infos = read_json(root_path / "data_info.json")

    for data_info in data_infos:
        # path of the image to be visualized
        image_path = root_path / data_info["image_path"]

        # infrastructure or vehicle
        if root_path.name in {"infrastructure-side", "single-infrastructure-side"}:
            label_path = root_path / data_info["label_camera_std_path"]
            calib_lidar2cam_path = root_path / data_info["calib_virtuallidar_to_camera_path"]
        elif root_path.name in {"vehicle-side", "single-vehicle-side"}:
            label_path = root_path / data_info["label_camera_std_path"]
            calib_lidar2cam_path = root_path / data_info["calib_lidar_to_camera_path"]

        labels = []

        objects = read_json(label_path)
        for o in objects:
            if "rotation" not in o.keys():
                o["rotation"] = 0.0
            labels.append([o["3d_dimensions"], o["3d_location"], o["rotation"]])

        # convert the label to the 8 points of the camera coordinate system
        camera_8_points_list = get_cam_8_points(labels, calib_lidar2cam_path)  
        cam_intrinsic_path = root_path /  data_info["calib_camera_intrinsic_path"]

        # draw and save the image with projected 3D bounding boxes
        vis_label_in_img(camera_8_points_list, image_path, cam_intrinsic_path, output_path)


def main():
    # parameters
    parser = argparse.ArgumentParser(description="Visualize labels in images (infrastructure or vehicle side).")
    parser.add_argument("--root-path", type=str, required=True,
                        help="Root directory containing data_info.json and related files")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for visualized images")
    args = parser.parse_args()

    # cast to path objects
    args.root_path = Path(args.root_path)
    args.output_path = Path(args.output_path)

    # validate input path
    if not args.root_path.exists():
        parser.error(f"Input path does not exist: {args.root_path}")

    # ensure output directory exists (create parents as needed)
    args.output_path.mkdir(parents=True, exist_ok=True)

    # visualize labels in images
    vis_label_in_image(args.root_path, args.output_path)

if __name__ == "__main__":
    main()