#---------------------------------------------------------------------------------#
# V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting (https://arxiv.org/abs/2305.05938)  #
# Source code: https://github.com/AIR-THU/DAIR-V2X-Seq                              #
# Copyright (c) DAIR-V2X. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""A simple python script template."""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scipy.interpolate as interp

import sys
sys.path.append("../..")
from projects.dataset.dair_map_api import DAIRMap

_ZORDER = {
    "TARGET_AGENT": 15, "AV": 10, "COVEH": 10,
    "AGENT_2": 8, "AGENT_3": 8, "AGENT_4": 8, "AGENT_5": 8,
    "OTHERS": 3, "VIC_OTHERS": 5, "CAR_OTHERS": 3}


def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def _plot_actor_bounding_box(
    ax: plt.Axes, final_x, final_y, heading: float, color: str, bbox_size: Tuple[float, float], _zorder
) -> None:

    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = final_x - (d / 2) * math.cos(heading + theta_2)
    pivot_y = final_y - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), color=color, zorder=_zorder
    )
    ax.add_patch(vehicle_bounding_box)


def viz_static(
    df: pd.DataFrame,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = False,
    smoothen: bool = True,
) -> None:
    # Seq data
    city_name = df["city"].values[0]

    if lane_centerlines is None:
        # Get API for DAIR Dataset map
        avm = DAIRMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    _, ax = plt.subplots(figsize=(12, 8))

    x_min = min(df["x"])
    x_max = max(df["x"])
    y_min = min(df["y"])
    y_max = max(df["y"])

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        lane_ids = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline
            
            if (
            (min(lane_cl, key=lambda x:x[0])[0]) < x_max
            and (min(lane_cl, key=lambda x:x[1])[1]) < y_max
            and (max(lane_cl, key=lambda x:x[0])[0]) > x_min
            and (max(lane_cl, key=lambda x:x[1])[1]) > y_min
            ):
                lane_centerlines.append(lane_cl)
                lane_ids.append(lane_id)
    
    for lane_id in lane_ids:
        polygon = avm.get_lane_segment_polygon(lane_id, city_name)
        plt.fill(polygon[:, 0], polygon[:, 1], color='#7A7A7A', alpha=1.0)
        plt.plot(polygon[:, 0], polygon[:, 1], '-', linewidth=0.5, color='#E0E0E0', alpha=1.0)
    
    frames = df.groupby("id")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {
        "TARGET_AGENT": "#d33e4c", 
        "AGENT_2": "orange", "AGENT_3": "orange", "AGENT_4": "orange", "AGENT_5": "orange",
        "AV": "#007672", 
        "OTHERS": "#d3e8ef", "CAR_OTHERS": "#d3e8ef", "VIC_OTHERS": "#75bbfd"
        }
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        
        object_type = group_data["tag"].values[0]
        obj_class = group_data["type"].values[0]

        cor_x = group_data["x"].values
        cor_y = group_data["y"].values
        cor_theta = group_data["theta"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        final_x = cor_x[-1]
        final_y = cor_y[-1]
        final_theta = cor_theta[-1]

        marker_type = "o"
        marker_size = 3

        if object_type == 'AV':
            plt.plot(
                final_x,
                final_y,
                "o",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=7,
                zorder=_ZORDER[object_type]
            )
        else:
            if obj_class == 'VEHICLE':
                _plot_actor_bounding_box(
                    ax,
                    final_x,
                    final_y,
                    final_theta,
                    color_dict[object_type],
                    (3.2, 1.6),
                    _zorder=_ZORDER[object_type]
                )
            elif obj_class == 'BICYCLE':
                _plot_actor_bounding_box(
                    ax,
                    final_x,
                    final_y,
                    final_theta,
                    color_dict[object_type],
                    (1.6, 0.56),
                    _zorder=_ZORDER[object_type]
                )
            else:
                plt.plot(
                final_x,
                final_y,
                marker_type,
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type]
            )

        object_type_tracker[object_type] += 1
    
    plt.axis("off")
    if show:
        plt.show()