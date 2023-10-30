import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from argoverse.utils.centerline_utils import centerline_to_polygon

def merge_map_json(maps):

    lane_dict = {}
    for map in tqdm(maps, desc='merge_map_json'):
        
        with open(os.path.join(map_dir, map), encoding='utf-8') as f:

            for lane_id, lane in json.load(f)['LANE'].items():
                
                centerline = []
                for pt in lane['centerline']:
                    xy = []
                    xy.append(float(pt[1:14]))
                    xy.append(float(pt[-15:-1]))
                    centerline.append(xy)
            
                lane['centerline'] = centerline
                
                lane_dict[lane_id] = lane

    with open(os.path.join(output_dir, 'yizhuang_PEK_vector_map.json'),"w") as f:
        json.dump(lane_dict,f)


def merge_halluc(maps):

    idx = 0
    map_dict = {}
    polygon_ls = []

    lane_dict = {}
    for map in maps:
        with open(os.path.join(map_dir, map), encoding='utf-8') as f :
            for lane_id, lane in json.load(f)['LANE'].items():
                lane_dict[lane_id] = lane

    for lane_id, lane in tqdm(lane_dict.items(), desc='merge_halluc'):
        map_dict[str(idx)] = lane_id

        lane_ls = []
        
        for pt in lane['centerline']:
            xy = []
            xy.append(float(pt[1:14]))
            xy.append(float(pt[-15:-1]))
            lane_ls.append(xy)
            
        polygon = centerline_to_polygon(np.array(lane_ls))

        x_min, y_min = 500000, 5000000
        x_max, y_max = 0, 0

        for iter in polygon:

            x_min = min(x_min, iter[0])
            x_max = max(x_max, iter[0])
            y_min = min(y_min, iter[1])
            y_max = max(y_max, iter[1])

        polygon_xy = [x_min, y_min, x_max, y_max]
        polygon_ls.append(polygon_xy)

        idx += 1

    polygon_ls = np.array(polygon_ls)

    with open(os.path.join(output_dir, 'yizhuang_PEK_tableidx_to_laneid_map.json'),"w") as f:
        json.dump(map_dict,f)

    np.save(os.path.join(output_dir, 'yizhuang_PEK_halluc_bbox_table.npy'), polygon_ls)

def merge_multiple_maps():

    maps = os.listdir(map_dir)

    merge_map_json(maps)

    merge_halluc(maps)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='merge maps')
    parser.add_argument("--data_root", type=str, default="../../dataset/v2x-seq-tfd/V2X-Seq-TFD-Example")

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    print(args)

    map_dir = os.path.join(args.data_root, 'maps')
    output_dir = os.path.join(args.data_root, 'map_files')
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    merge_multiple_maps()