import os
import json
import pandas as pd
import numpy as np
import multiprocessing  
import warnings
import argparse
warnings.filterwarnings("ignore")

def cut(num, c):
    
    str_num = str(num)

    return int(str_num[:str_num.index('.') + 1 + c])

def format(iter):

    pre_df = pd.read_csv(os.path.join(data_path, iter))

    pre_df = pre_df[[
            'header.lidar_timestamp', 
            'id', 
            'type',
            'sub_type',
            'position.x', 
            'position.y',
            'position.z',
            'length',
            'width',
            'height',
            'theta',
            'velocity.x',
            'velocity.y',
            'from_side',
            'car_side_id',
            'road_side_id',
            'tag'
            ]]
    pre_df.columns = [
            'timestamp', 'id', 'type', 'sub_type', 
            'x', 'y', 'z', 'length', 'width', 'height', 'theta', 
            'v_x', 'v_y', 'from_side', 'car_side_id', 'road_side_id',
            'tag'
            ]
    
    t_min, t_max = min(pre_df.timestamp), max(pre_df.timestamp) + 5.0
    t_obs = t_min + 4.9

    car_df = pd.read_csv(os.path.join(car_path, iter))
    car_obs_df = car_df.loc[(car_df.timestamp >= t_min) & (car_df.timestamp <= t_obs)]
    tgt_id = car_obs_df.loc[car_obs_df.tag == 'TARGET_AGENT', 'id'].values[0]

    pre_df.drop(pre_df.loc[pre_df.type == 'EGO_VEHICLE', 'type'].index, inplace=True)
    pre_df.tag = pre_df.tag.map(lambda x: id2tag[x])
    pre_df.drop(pre_df.loc[pre_df.tag == 'AV'].index, inplace=True)

    pre_obs_df = pre_df.loc[(pre_df.timestamp >= t_min) & (pre_df.timestamp <= t_obs)]

    new_df_0 = pd.DataFrame()
    new_df = pd.DataFrame()
    
    ids = pre_obs_df.id.unique()
    for id in ids[:]:
        id_df = pre_obs_df.loc[pre_obs_df.id == id]
        car_id = id_df.car_side_id.drop_duplicates(inplace=False).to_list()
        if -1 in car_id:
            car_id.remove(-1)
        road_id = id_df.road_side_id.drop_duplicates(inplace=False).to_list()
        if -1 in road_id:
            road_id.remove(-1)
        if len(car_id):
            if tgt_id in car_id:
                new_id = tgt_id
            else:
                new_id = car_id[-1]
        elif len(road_id):
            new_id = road_id[-1]
        else:
            new_id = 0
        id_df.id = new_id

        if new_id == tgt_id:
            id_df.tag = 'TARGET_AGENT'
        else:
            id_df.tag = 'OTHERS'
        new_df_0 = new_df_0.append(id_df)

    for timestamp_iter in new_df_0.timestamp.unique():

        tmp = new_df_0.loc[
            new_df_0.timestamp == timestamp_iter
            ].drop_duplicates(subset='id', inplace=False)
        
        new_df = new_df.append(tmp, ignore_index=True)
    
    car_fut_df = car_df.loc[(car_df.timestamp > t_obs) & (car_df.timestamp <= t_max)]
    
    av_df = car_df.loc[car_df.tag == 'AV']
    av_df.loc[av_df.tag == 'AV', 'id'] = 0
    av_df.insert(16, 'from_side', -1)
    av_df.insert(17, 'car_side_id', -1)
    av_df.insert(18, 'road_side_id', -1)
    
    new_df = new_df.append(car_fut_df)
    new_df=new_df.reset_index(drop=True)
    new_df.drop(index=new_df.loc[new_df.tag == 'AV'].index, inplace=True)
    new_df = new_df.append(av_df)

    new_df.to_csv(os.path.join(data_dest, iter), index=False)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='converting tfd to argoverse')
    parser.add_argument("--data_root", type=str, default="../../dataset/v2x-seq-tfd/V2X-Seq-TFD-Example")
    parser.add_argument("--split", help="split.", type=str, default='train') #train; val; test_obs

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()

    id2tag = {
        0: 'AV', 
        1: 'TARGET_AGENT', 
        2: 'AGENT_2', 
        3: 'AGENT_3', 
        4: 'AGENT_4', 
        5: 'AGENT_5', 
        6: 'OTHERS'
    }

    data_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/tfd_fusion', args.split, 'data')
    data_dest = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/fusion_for_prediction', args.split, 'data')
    
    data_ls = os.listdir(data_path)

    car_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/vehicle-trajectories', args.split, 'data')

    if not os.path.exists(data_dest):
        os.makedirs(data_dest)

    # run for train and val dataset
    pool = multiprocessing.Pool(processes = 16)
    pool.map_async(format, data_ls[:]).get()
    pool.close()
    pool.join()