#---------------------------------------------------------------------------------#
# V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting (https://arxiv.org/abs/2305.05938)  #
# Source code: https://github.com/AIR-THU/DAIR-V2X-Seq                              #
# Copyright (c) DAIR-V2X. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import os
import argparse
from os.path import join as pjoin
import copy
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse

import warnings

# import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append("..")
sys.path.append(".")
sys.path.append("../../../..")
from dataset.dair_data_loader import DAIRV2XDataLoader
from dataset.dair_map_api import DAIRV2XMap
#sys.path.append("../../..")
from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D

warnings.filterwarnings("ignore")

RESCALE_LENGTH = 1.0    # the rescale length th turn the lane vector into equal distance pieces


class TFDPreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="tnt",
                 obs_horizon=50,
                 obs_range=50,
                 pred_horizon=50,
                 normalized=True,
                 save_dir=None):
        super(TFDPreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)

        self.LANE_WIDTH = {'PEK': 3.8}

        self.split = split
        self.normalized = normalized

        self.dm = DAIRV2XMap()
        self.loader = DAIRV2XDataLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split, "data"))

        self.save_dir = save_dir

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)

        df = copy.deepcopy(seq.seq_df)

        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    def process(self, dataframe: pd.DataFrame,  seq_id, map_feat=True):
        data = self.read_tfd_data(dataframe)
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id

        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_tfd_data(df: pd.DataFrame):
        city = 'PEK'

        """timestamp, id, tag, x, y"""
        agt_ts = np.sort(np.unique(df['timestamp'].values))
        
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.x.to_numpy().reshape(-1, 1),
            df.y.to_numpy().reshape(-1, 1)), 1)

        theta = df.theta.to_numpy().reshape(-1, 1)

        steps = [mapping[x] for x in df['timestamp'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['id', 'tag']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idxs = [i for i,j in enumerate(obj_type) if j == 'TARGET_AGENT']

        agt_trajs = []
        agt_steps = []
        agt_theta = []
        
        for agt_idx in agt_idxs:

            idcs = objs[keys[agt_idx]]

            agt_trajs.append(trajs[idcs])
            agt_steps.append(steps[idcs])
            agt_theta.append(theta[idcs])
        
        keys_ = [keys[i] for i in range(len(keys)) if (i not in agt_idxs)]

        ctx_trajs, ctx_steps = [], []
        for key in keys_:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city

        data['trajs'] = agt_trajs + ctx_trajs #[0]agent/rest others
        data['steps'] = agt_steps + ctx_steps
        data['agt_theta'] = agt_theta
        data['num_agent'] = 1

        return data

    def get_obj_feats(self, data):
        # get the origin and compute the oritentation of the target agent
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float64)
        # comput the rotation matrix
        if self.normalized:
            pre, conf = self.dm.get_lane_direction(data['trajs'][0][self.obs_horizon-1], data['city'])
            if conf <= 0.1:
                pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float64)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float64)

        # get the target candidates and candidate gt
        #for all agents
        
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float64)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float64)
        ctr_line_candts = self.dm.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'], viz=False)
        
        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T

        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)

        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):
            
            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float64)
            has_pred = np.zeros(self.pred_horizon, bool)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            # colect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]
            
            if len(step_obs) <= 1:
                continue
            
            feat = np.zeros((self.obs_horizon, 3), np.float64)
            has_obs = np.zeros(self.obs_horizon, bool)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float64)
        has_obss = np.asarray(has_obss, bool)
        gt_preds = np.asarray(gt_preds, np.float64)
        has_preds = np.asarray(has_preds, bool)


        # # target candidate filtering
        # tar_candts = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        # inlier = np.logical_and(np.fabs(tar_candts[:, 0]) <= self.obs_range, np.fabs(tar_candts[:, 1]) <= self.obs_range)
        # if not np.any(candts_gt[inlier]):
        #     raise Exception("The gt of target candidate exceeds the observation range!")

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.dm.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.dm.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)

            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.dm.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float64))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float64))

            x = np.zeros((num_segs, 2), np.float64)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float64))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float64))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--root", type=str, default="./../../../../../dataset/v2x-seq-tfd-example/cooperative-vehicle-infrastructure/fusion_for_prediction/")
    parser.add_argument("-s", "--small", default=False)
    args = parser.parse_args()

    print(args)
    args.dest = args.root
    raw_dir = os.path.join(args.root)
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")

    for split in ["train", "val"]:
        # construct the preprocessor and dataloader
        tfd_processor = TFDPreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        loader = DataLoader(tfd_processor,
                            batch_size=1 if sys.gettrace() else 32,     # 1 batch in debug mode
                            num_workers=0 if sys.gettrace() else 32,    # use only 0 worker in debug mode
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):
            if args.small:
                if split == "train" and i >= 200:
                    break
                elif split == "val" and i >= 50:
                    break
                elif split == "test" and i >= 50:
                    break
