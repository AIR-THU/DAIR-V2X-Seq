#---------------------------------------------------------------------------------#
# V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting (https://arxiv.org/abs/2305.05938)  #
# Source code: https://github.com/AIR-THU/DAIR-V2X-Seq                              #
# Copyright (c) DAIR-V2X. All rights reserved.                                #
#---------------------------------------------------------------------------------#
import os
from pathlib import Path

from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import sys
sys.path.append("../..")
from dataset.dair_map_api import DAIRV2XMap

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm

from utils import TemporalData


class TFDInfraDataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        makedirs(self.processed_dir)
        self._resume_file_names = list(set(self._processed_file_names).difference(set(os.listdir(self.processed_dir))))
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        self._resume_paths = [os.path.join(self.processed_dir, f) for f in self._resume_file_names]
        super(TFDInfraDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def tl_raw_dir(self) -> str:
        return os.path.join(Path(self.root).parent, 'traffic-light', self._directory)

    @property
    def road_raw_dir(self) -> str:
        return os.path.join(Path(self.root).parent, 'infrastructure-trajectories', self._directory, 'data')

    @property
    def vic_raw_dir(self) -> str:
        return os.path.join(Path(self.root).parent, 'cooperative-trajectories', self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def resume_file_names(self) -> List[str]:
        return [os.path.splitext(f)[0] + '.csv' for f in self._resume_file_names]

    @property
    def resume_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.resume_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        
        dm = DAIRV2XMap()
        #for raw_path in tqdm(self.raw_paths):
        for raw_path in tqdm(self.resume_paths):

            kwargs = process_argoverse(self._split, raw_path, dm, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      dm: DAIRV2XMap,
                      radius: float,
                      tl_path: Optional[str] = None,
                      road_path: Optional[str] = None,
                      vic_path: Optional[str] = None) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps

    timestamps = list(np.sort(df['timestamp'].unique()))
    historical_timestamps = timestamps[: 50]
    historical_df = df[df['timestamp'].isin(historical_timestamps)]
    actor_ids = list(historical_df['id'].unique())
    df = df[df['id'].isin(actor_ids)]
    num_nodes = len(actor_ids)
    
    # av_df = df[df['tag'] == 'AV'].iloc
    # av_index = actor_ids.index(av_df[0]['id'])
    av_df = df[df['tag'] == 'TARGET_AGENT'].iloc
    av_index = actor_ids.index(av_df[0]['id'])

    fut_timestamps = timestamps[50:]
    fut_df = df[df['timestamp'].isin(fut_timestamps)]
    agent_df = fut_df[fut_df['tag'] == 'TARGET_AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['id'])
    #city = df['city'].values[0]
    city = 'PEK'

    # make the scene centered at AV
    origin = torch.tensor([av_df[49]['x'], av_df[49]['y']], dtype=torch.float64)
    '''
    av_heading_vector = origin - torch.tensor([av_df[48]['x'], av_df[48]['y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    '''
    theta = torch.tensor(av_df[49]['theta'], dtype=torch.float64)
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]], dtype=torch.float64)
    # initialization
    x = torch.zeros(num_nodes, 100, 2, dtype=torch.float64)
    #last_positions = torch.zeros(num_nodes, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool) # False means valid
    bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float64)

    for actor_id, actor_df in df.groupby('id'):
        actor_hist_df = actor_df[actor_df['timestamp'].isin(historical_timestamps)]
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
        padding_mask[node_idx, node_steps] = False  # has frame at this timestamp
        
        xy = torch.from_numpy(np.stack([actor_df['x'].values, actor_df['y'].values], axis=-1)).double()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            rotate_angles[node_idx] = actor_hist_df['theta'].values[-1]
        else:  # make no predictions for the actor if the number of valid time steps is less than 5
            padding_mask[node_idx, 50:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 50] = padding_mask[:, : 49] & ~padding_mask[:, 1: 50]

    positions = x.clone()
    x[:, 50:] = x[:, 50:] - x[:, 49].unsqueeze(-2)
    x[:, 1: 50] = torch.where((padding_mask[:, : 49] | padding_mask[:, 1: 50]).unsqueeze(-1),
                              torch.zeros(num_nodes, 49, 2),
                              x[:, 1: 50] - x[:, : 49]) # difference
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    df_49 = df[df['timestamp'] == timestamps[49]]
    node_inds_49 = [actor_ids.index(actor_id) for actor_id in df_49['id']]
    node_positions_49 = torch.from_numpy(np.stack([df_49['x'].values, df_49['y'].values], axis=-1)).double()

    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(dm, node_inds_49, node_positions_49, origin, rotate_mat, city, radius)
    
    y = None if split == 'test' else x[:, 50:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]
    
    return {
        'x': x[:, : 50].float(),  # [N, 50, 2]
        'positions': positions.float(),  # [N, 100, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y.float(),  # [N, 50, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 100]
        'bos_mask': bos_mask,  # [N, 50]
        'rotate_angles': rotate_angles.float(),  # [N]
        'lane_vectors': lane_vectors.float(),  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors.float(),  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0).float(),
        'theta': theta.float(),
    }


def get_lane_features(dm: DAIRV2XMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(dm.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).double()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(dm.get_lane_segment_centerline(lane_id, city)[:, : 2]).double()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = dm.lane_is_in_intersection(lane_id, city)
        turn_direction = dm.get_lane_turn_direction(lane_id, city)
        traffic_control = dm.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        elif turn_direction == 'UTURN':
            turn_direction = 3
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
