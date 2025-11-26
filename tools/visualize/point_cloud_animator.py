import time
from pathlib import Path
from pypcd import pypcd
import numpy as np
import open3d as o3d


class PointCloudAnimator:
    def __init__(self, window_name: str = "Animated Frames", width: int = 800, height: int = 600, point_size: float = 2.0, sleep_time: float = 0.03):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.point_size = point_size
        self.sleep_time = sleep_time
        self._vis = None
        self._pcd = None

    def _create_window(self, pts: np.ndarray) -> None:
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name=self.window_name, width=self.width, height=self.height)
        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(pts)
        self._vis.add_geometry(self._pcd)
        opt = self._vis.get_render_option()
        opt.point_size = self.point_size

    def update(self, dir: Path, source: str, data: dict, last_frame: bool) -> bool:
        """Load a .pcd file (based on data[f'{source}_frame']) and update the visualizer.

        Returns True if a point cloud was shown/updated, False if the point cloud was empty.
        """
        cloud = pypcd.PointCloud.from_path(str(dir / f"{data[f'{source}_frame']}.pcd"))
        pts = np.vstack((
            np.asarray(cloud.pc_data['x']),
            np.asarray(cloud.pc_data['y']),
            np.asarray(cloud.pc_data['z'])
        )).T

        if pts.size == 0:
            return False

        if self._vis is None:
            self._create_window(pts)
        else:
            self._pcd.points = o3d.utility.Vector3dVector(pts)
            self._vis.update_geometry(self._pcd)

        self._vis.poll_events()
        self._vis.update_renderer()
        time.sleep(self.sleep_time)

        if last_frame and self._vis is not None:
            # Block and allow user interaction for the final frame, then cleanup.
            self._vis.run()
            self._vis.destroy_window()
            self._vis = None
            self._pcd = None

        return True

    def close(self) -> None:
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None
            self._pcd = None

    def __del__(self) -> None:
        self.close()