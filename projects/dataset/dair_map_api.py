#---------------------------------------------------------------------------------#
# V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting (https://arxiv.org/abs/2305.05938)  #
# Source code: https://github.com/AIR-THU/DAIR-V2X-Seq                              #
# Copyright (c) DAIR-V2X. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from argoverse.map_representation.map_api import ArgoverseMap

from argoverse.utils.centerline_utils import centerline_to_polygon

from argoverse.utils.json_utils import read_json_file
from argoverse.utils.manhattan_search import compute_polygon_bboxes
from argoverse.utils.interpolate import interp_arc


from argoverse.map_representation.lane_segment import LaneSegment

GROUND_HEIGHT_THRESHOLD = 0.3  # 30 centimeters
MAX_LABEL_DIST_TO_LANE = 20  # meters
OUT_OF_RANGE_LANE_DIST_THRESHOLD = 5.0  # 5 meters
ROI_ISOCONTOUR = 5.0

work_dir = "/y/minkycho/DAIR-V2X-Seq"
ROOT = f"{work_dir}/tutorial/merged_map"

# known City IDs from newest to oldest
PEKING_ID = 10000

# Any numeric type
Number = Union[int, float]
_PathLike = Union[str, "os.PathLike[str]"]


class DAIRV2XMap(ArgoverseMap):
    """
    This class provides the interface to our vector maps and rasterized maps. Exact lane boundaries
    are not provided, but can be hallucinated if one considers an average lane width.
    """

    def __init__(self, root: _PathLike = ROOT) -> None:
        """Initialize the DAIR Map."""
        self.root = root

        self.city_name_to_city_id_dict = {"PEK": PEKING_ID}
        self.render_window_radius = 150
        self.im_scale_factor = 50

        self.city_lane_centerlines_dict = self.build_centerline_index
        (
            self.city_halluc_bbox_table,
            self.city_halluc_tableidx_to_laneid_map,
        ) = self.build_hallucinated_lane_bbox_index()

        # get hallucinated lane extends and driveable area from binary img
        self.city_to_lane_polygons_dict: Mapping[str, np.ndarray] = {}
        self.city_to_lane_bboxes_dict: Mapping[str, np.ndarray] = {}

        for city_name in self.city_name_to_city_id_dict.keys():
            lane_polygons = np.array(self.get_vector_map_lane_polygons(city_name), dtype=object)
            lane_bboxes = compute_polygon_bboxes(lane_polygons)

            self.city_to_lane_polygons_dict[city_name] = lane_polygons
            self.city_to_lane_bboxes_dict[city_name] = lane_bboxes

    @property
    def build_centerline_index(self) -> Mapping[str, Mapping[str, LaneSegment]]:
        """
        Build dictionary of centerline for each city, with lane_id as key

        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """
        city_lane_centerlines_dict = {}
        # '''
        # for city_name, city_id in self.city_name_to_city_id_dict.items():
        #     xml_fpath = self.map_files_root / f"pruned_argoverse_{city_name}_{city_id}_vector_map.xml"
        #     city_lane_centerlines_dict[city_name] = load_lane_segments_from_xml(xml_fpath)
        # '''
        for city_name, city_id in self.city_name_to_city_id_dict.items():

            map_dir = 'dair_v2x_seq_vector_map.json'
            lane_segment_dict = {}

            with open(os.path.join(self.root, map_dir), encoding='utf-8') as f :
                for lane_id, lane in json.load(f).items():
                    lane_segment = LaneSegment(
                        id = lane_id,
                        has_traffic_control = lane['has_traffic_control'],
                        turn_direction = lane['turn_direction'],
                        is_intersection = lane['is_intersection'],
                        l_neighbor_id = lane['l_neighbor_id'],
                        r_neighbor_id = lane['r_neighbor_id'],
                        predecessors = lane['predecessors'],
                        successors = lane['successors'],
                        centerline = lane['centerline']
                        )
                    lane_segment_dict[lane_id] = lane_segment
            city_lane_centerlines_dict[city_name] = lane_segment_dict

        return city_lane_centerlines_dict

    def build_hallucinated_lane_bbox_index(
        self,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.

        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = {}
        city_halluc_tableidx_to_laneid_map = {}

        for city_name, city_id in self.city_name_to_city_id_dict.items():
            json_fpath = self.map_files_root / "dair_v2x_seq_tableidx_to_laneid_map.json"
            city_halluc_tableidx_to_laneid_map[city_name] = read_json_file(json_fpath)

            npy_fpath = self.map_files_root / "dair_v2x_seq_halluc_bbox_table.npy"
            city_halluc_bbox_table[city_name] = np.load(npy_fpath)

        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map

    def append_height_to_2d_city_pt_cloud(self, pt_cloud_xy: np.ndarray, city_name: str) -> np.ndarray:
        """Accept 2d point cloud in xy plane and return 3d point cloud (xyz).

        Args:
            pt_cloud_xy: Numpy array of shape (N,2)
            city_name: 'PEK' for Peking

        Returns:
            pt_cloud_xyz: Numpy array of shape (N,3)
        """
        pts_z = np.zeros(len(pt_cloud_xy))
        return np.hstack([pt_cloud_xy, pts_z[:, np.newaxis]])

    def get_lane_segment_polygon(self, lane_segment_id: str, city_name: str) -> np.ndarray:
        """
        Hallucinate a 3d lane polygon based around the centerline. We rely on the average
        lane width within our cities to hallucinate the boundaries. We rely upon the
        rasterized maps to provide heights to points in the xy plane.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: 'PEK' for Peking

        Returns:
            lane_polygon: Array of polygon boundary (K,3), with identical and last boundary points
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][lane_segment_id].centerline
        lane_polygon = centerline_to_polygon(np.array(lane_centerline)[:, :2])
        return self.append_height_to_2d_city_pt_cloud(lane_polygon, city_name)

    def lane_waypt_to_query_dist(
        self, query_xy_city_coords: np.ndarray, nearby_lane_objs: List[LaneSegment]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Compute the distance from a query to the closest waypoint in nearby lanes.

        Args:
        query_xy_city_coords: Numpy array of shape (2,)
        nearby_lane_objs: list of LaneSegment objects

        Returns:
        per_lane_dists: array with distance to closest waypoint for each centerline
        min_dist_nn_indices: array with ranked indices of centerlines, closest first
        dense_centerlines: list of arrays, each representing (N,2) centerline
        """
        per_lane_dists: List[float] = []
        dense_centerlines: List[np.ndarray] = []
        for nn_idx, lane_obj in enumerate(nearby_lane_objs):
            centerline = lane_obj.centerline
            # densely sample more points
            sample_num = 50
            centerline = interp_arc(
                sample_num, np.array(centerline)[:, 0], np.array(centerline)[:, 1]
                )
            dense_centerlines += [centerline]
            # compute norms to waypoints
            waypoint_dist = np.linalg.norm(centerline - query_xy_city_coords, axis=1).min()
            per_lane_dists += [waypoint_dist]
        per_lane_dists = np.array(per_lane_dists)
        min_dist_nn_indices = np.argsort(per_lane_dists)
        return per_lane_dists, min_dist_nn_indices, dense_centerlines

    def get_nearest_centerline(
        self, query_xy_city_coords: np.ndarray, city_name: str, visualize: bool = False
    ) -> Tuple[LaneSegment, float, np.ndarray]:
        """
        KD Tree with k-closest neighbors or a fixed radius search on the lane centroids
        is unreliable since (1) there is highly variable density throughout the map and (2)
        lane lengths differ enormously, meaning the centroid is not indicative of nearby points.
        If no lanes are found with MAX_LABEL_DIST_TO_LANE, we increase the search radius.

        A correct approach is to compare centerline-to-query point distances, e.g. as done
        in Shapely. Instead of looping over all points, we precompute the bounding boxes of
        each lane.

        We use the closest_waypoint as our criterion. Using the smallest sum to waypoints
        does not work in many cases with disproportionately shaped lane segments.

        and then choose the lane centerline with the smallest sum of 3-5
        closest waypoints.

        Args:
            query_xy_city_coords: Numpy array of shape (2,) representing xy position of query in city coordinates
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
            visualize:

        Returns:
            lane_object: Python dictionary with fields describing a lane.
                Keys include: 'centerline', 'predecessor', 'successor', 'turn_direction',
                             'is_intersection', 'has_traffic_control', 'is_autonomous', 'is_routable'
            conf: real-valued confidence. less than 0.85 is almost always unreliable
            dense_centerline: numpy array
        """
        query_x = query_xy_city_coords[0]
        query_y = query_xy_city_coords[1]

        lane_centerlines_dict = self.city_lane_centerlines_dict[city_name]

        search_radius = MAX_LABEL_DIST_TO_LANE
        while True:
            nearby_lane_ids = self.get_lane_ids_in_xy_bbox(
                query_x, query_y, city_name, query_search_range_manhattan=search_radius
            )
            if not nearby_lane_ids:
                search_radius *= 2  # double search radius
            else:
                break

        nearby_lane_objs = [lane_centerlines_dict[lane_id] for lane_id in nearby_lane_ids]

        cache = self.lane_waypt_to_query_dist(query_xy_city_coords, nearby_lane_objs)
        per_lane_dists, min_dist_nn_indices, dense_centerlines = cache

        closest_lane_obj = nearby_lane_objs[min_dist_nn_indices[0]]
        dense_centerline = dense_centerlines[min_dist_nn_indices[0]]

        # estimate confidence
        conf = 1.0 - (per_lane_dists.min() / OUT_OF_RANGE_LANE_DIST_THRESHOLD)
        conf = max(0.0, conf)  # clip to ensure positive value

        if visualize:
            # visualize dists to nearby centerlines
            fig = plt.figure(figsize=(22.5, 8))
            ax = fig.add_subplot(111)

            (query_x, query_y) = query_xy_city_coords.squeeze()
            ax.scatter([query_x], [query_y], 100, color="k", marker=".")
            # make another plot now!

            self.plot_nearby_halluc_lanes(ax, city_name, query_x, query_y)

            for i, line in enumerate(dense_centerlines):
                ax.plot(line[:, 0], line[:, 1], color="y")
                ax.text(line[:, 0].mean(), line[:, 1].mean(), str(per_lane_dists[i]))

            ax.axis("equal")
            plt.show()
            plt.close("all")
        return closest_lane_obj, conf, dense_centerline