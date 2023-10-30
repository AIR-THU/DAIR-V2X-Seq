import numpy as np
import copy
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from utils import match, cal_matched_ids

MIN_ASSO_PTS_NUMS = 3
MIN_ASSO_RATIO = 0.25

class Fusion(object):

    def __init__(self,iou_threshold = 0.3, iou_threshold_3d = 0.01, hungarian=True, dist_flag = 'iou_2d',is_solve_wrong_association=True):
        self.logger = logger

        self.iou_threshold = iou_threshold
        self.iou_threshold_3d = iou_threshold_3d
        self.hungarian = hungarian
        self.dist_flag = dist_flag
        self.is_solve_wrong_association = is_solve_wrong_association
        
        self.id_counter = 1
    
    def get_tracks_fusion_info_for_pairs(self, tracks1,tracks2,track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v):   
        new_tracks2 = copy.deepcopy(tracks2)

        if self.dist_flag == 'iou_2d':
            v_ind, r_ind = cal_matched_ids(tracks1,new_tracks2,self.iou_threshold,self.hungarian)
        else:
            v_ind, r_ind = match(tracks1,new_tracks2)

        # go through mated tracklets
        for track1_index in range(len(tracks1)):
            tracklet1 = tracks1[track1_index]
            track_id1 = tracklet1[2] 

            # MATCHED TRACKLETS:
            if track1_index in v_ind:
                tracklet2 = new_tracks2[r_ind[v_ind == track1_index]][0]
                track_id2 = tracklet2[2]

                for i2v_id,i2v_value in track_fusion_info_i2v[track_id2].items():
                    if i2v_value > 0 and i2v_id != track_id1:
                        if track_id1 in track_relation_veh and i2v_id in track_relation_veh[track_id1]:
                            if 'related_inf_id' not in track_relation_veh[track_id1][i2v_id]:
                                track_relation_veh[track_id1][i2v_id]['related_inf_id'] = set()
                            track_relation_veh[track_id1][i2v_id]['related_inf_id'].add(track_id2)

                for v2i_id,v2i_value in track_fusion_info_v2i[track_id1].items():
                    if v2i_value > 0 and v2i_id != track_id2:
                        if track_id2 in track_relation_inf and v2i_id in track_relation_inf[track_id2]:
                            if 'related_inf_id' not in track_relation_inf[track_id2][v2i_id]:
                                track_relation_inf[track_id2][v2i_id]['related_inf_id'] = set()
                            track_relation_inf[track_id2][v2i_id]['related_inf_id'].add(track_id1)
                        
                track_fusion_info_v2i[track_id1][track_id2] += 1
                track_fusion_info_i2v[track_id2][track_id1] += 1

        return 

    def build_relation_table_for_tracks(self,tracks):
        relation_table = {}
        tracks_frame_id = np.sort(np.unique(tracks[:,0]))
        for frame_id in tracks_frame_id:
            cur_frame_tracks = tracks[tracks[:,0] == frame_id]
            cur_frame_tracks_id = np.unique(cur_frame_tracks[:,2])

            cur_frame_tracks_id_nums = len(cur_frame_tracks_id)
            for i in range(0,cur_frame_tracks_id_nums):
                track_id1 = cur_frame_tracks_id[i]
                if track_id1 not in relation_table:
                    relation_table[track_id1] = {}   

                for j in range(i+1,cur_frame_tracks_id_nums):  
                    track_id2 = cur_frame_tracks_id[j] 
                    if track_id2 not in relation_table:
                        relation_table[track_id2] = {}  

                    relation_table[track_id1][track_id2] = {'flag':-1}
                    relation_table[track_id2][track_id1] = {'flag':-1}
        return relation_table    

    def buile_track_fusion_info(self,tracks1,tracks2):
        track_fusion_info_v2i,track_fusion_info_i2v = {},{}

        track_ids_1 = np.unique(tracks1[:,2])
        track_ids_2 = np.unique(tracks2[:,2])

        for i in range(len(track_ids_1)):
            track_id1 = track_ids_1[i]
            if track_id1 not in track_fusion_info_v2i:
                track_fusion_info_v2i[track_id1] = {}
            for j in range(len(track_ids_2)):
                track_id2 = track_ids_2[j]
                track_fusion_info_v2i[track_id1][track_id2] = 0

                if track_id2 not in track_fusion_info_i2v:
                    track_fusion_info_i2v[track_id2] = {} 

                track_fusion_info_i2v[track_id2][track_id1] = 0              
   
        return track_fusion_info_v2i,track_fusion_info_i2v
    
    def mutual_exclusion(self,track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v):
        cannot_fusion_v2i = set()

        #mutual exclusion
        #return ((veh_id1,inf_id1),(veh_id2,inf_id2))
        for id1,value1 in track_relation_veh.items():
            for id2,value2 in value1.items():
                if 'flag' in value2 and value2['flag'] == -1 and 'related_inf_id' in value2 and len(value2['related_inf_id']) > 0:
                    for related_inf_id in value2['related_inf_id']:
                        #related_inf_id and id1, id2
                        if track_fusion_info_i2v[related_inf_id][id1] > track_fusion_info_i2v[related_inf_id][id2]:
                            cannot_fusion_v2i.add((id2,related_inf_id))
                        elif  track_fusion_info_i2v[related_inf_id][id1] < track_fusion_info_i2v[related_inf_id][id2]:
                            cannot_fusion_v2i.add((id1,related_inf_id))
                        else:
                            # print('error: id1/related_inf_id, id2/related_inf_id!')
                            cannot_fusion_v2i.add((id1,related_inf_id))
                            cannot_fusion_v2i.add((id2,related_inf_id))
    
        for id1,value1 in track_relation_inf.items():
            for id2,value2 in value1.items():
                if 'flag' in value2 and value2['flag'] == -1 and 'related_inf_id' in value2 and len(value2['related_inf_id']) > 0:
                    for related_inf_id in value2['related_inf_id']:
                        #related_inf_id and id1, id2
                        if track_fusion_info_v2i[related_inf_id][id1] > track_fusion_info_v2i[related_inf_id][id2]:
                            cannot_fusion_v2i.add((related_inf_id,id2))
                        elif  track_fusion_info_v2i[related_inf_id][id1] < track_fusion_info_v2i[related_inf_id][id2]:
                            cannot_fusion_v2i.add((related_inf_id,id1))
                        else:
                            # print('error: id1/related_inf_id, id2/related_inf_id!')
                            cannot_fusion_v2i.add((related_inf_id,id1))
                            cannot_fusion_v2i.add((related_inf_id,id2))
                        
        return cannot_fusion_v2i
    
    def solve_wrong_association(self,track_fusion_info,track_fusion_info_inv,tracks_len,tracks_len_inv,is_reverse=False):
        cannot_fusion_v2i = set()

        for track_id, v2i_value in track_fusion_info.items():
            #only be matched with one track
            matched_id_list = []
            for matched_id,association_pts_nums in v2i_value.items():
                if association_pts_nums > 0:
                    matched_id_list.append(matched_id)
            matched_id_nums = len(matched_id_list)
            if matched_id_nums == 1:
                matched_id = matched_id_list[0]
                association_pts_nums = track_fusion_info[track_id][matched_id]                    
                track_len = tracks_len[track_id]
                association_ratio = association_pts_nums / track_len

                #judge matched_id 
                matched_id_list_of_matched_id = []
                for tmp_id, pts_nums in track_fusion_info_inv[matched_id].items():
                    if pts_nums > 0:
                        matched_id_list_of_matched_id.append(tmp_id)
                if len(matched_id_list_of_matched_id) == 1:
                    #mutual unique match, exam both track_id and matched_id.
                    association_pts_nums_inv = track_fusion_info_inv[matched_id][track_id]
                    track_len_inv = tracks_len_inv[matched_id]
                    association_ratio_inv = association_pts_nums_inv / track_len_inv

                    if (association_pts_nums < MIN_ASSO_PTS_NUMS or association_ratio < MIN_ASSO_RATIO) and (association_pts_nums_inv < MIN_ASSO_PTS_NUMS or association_ratio_inv  < MIN_ASSO_RATIO):
                        #can not
                        if is_reverse:
                            cannot_fusion_v2i.add((matched_id,track_id))
                            #print('cannot fusion: v_id: %s,r_id: %s,association_pts_nums: %d, track_len: %d,association_ratio: %f'%(matched_id,track_id,association_pts_nums,track_len,association_ratio))
                        else:                            
                            cannot_fusion_v2i.add((track_id,matched_id))
                            #print('cannot fusion: v_id: %s,r_id: %s,association_pts_nums: %d, track_len: %d,association_ratio: %f'%(track_id,matched_id,association_pts_nums,track_len,association_ratio))
                else:
                    #track_id2matched_id is one2many,only exam this track_id
                    if association_pts_nums < MIN_ASSO_PTS_NUMS or association_ratio < MIN_ASSO_RATIO:
                        #can not
                        if is_reverse:
                            cannot_fusion_v2i.add((matched_id,track_id))
                            #print('cannot fusion: v_id: %s,r_id: %s,association_pts_nums: %d, track_len: %d,association_ratio: %f'%(matched_id,track_id,association_pts_nums,track_len,association_ratio))
                        else:                            
                            cannot_fusion_v2i.add((track_id,matched_id))
                            #print('cannot fusion: v_id: %s,r_id: %s,association_pts_nums: %d, track_len: %d,association_ratio: %f'%(track_id,matched_id,association_pts_nums,track_len,association_ratio))

        return cannot_fusion_v2i
    
    def correct_tracks_fusion_info(self,track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v,tracks_len_veh,tracks_len_inf):
        #solve mutual exclusion
        cannot_fusion_v2i_1 = self.mutual_exclusion(track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v)

        cannot_fusion_v2i_3 = set()
        cannot_fusion_v2i_4 = set()
        if self.is_solve_wrong_association:
            #solve wrong_association
            cannot_fusion_v2i_3 = self.solve_wrong_association(track_fusion_info_v2i,track_fusion_info_i2v,tracks_len_veh,tracks_len_inf)
            cannot_fusion_v2i_4 = self.solve_wrong_association(track_fusion_info_i2v,track_fusion_info_v2i,tracks_len_inf,tracks_len_veh,True)

        #total
        cannot_fusion_v2i = cannot_fusion_v2i_1.union(cannot_fusion_v2i_3,cannot_fusion_v2i_4)

        return cannot_fusion_v2i

    def get_track_len(self,tracks_data):
        track_len = {}

        tracks_ids = np.sort(np.unique(tracks_data[:,2]))
        for track_id in tracks_ids:
            track_len[track_id] = len(tracks_data[tracks_data[:,2] == track_id])
        
        return track_len

    def get_tracks_fusion_info_per_seq(self, tracks1_data, tracks2_data):
        #build relation_table_for_tracks1
        track_relation_veh = self.build_relation_table_for_tracks(tracks1_data)
        track_relation_inf = self.build_relation_table_for_tracks(tracks2_data)

        tracks_len_veh = self.get_track_len(tracks1_data)
        tracks_len_inf = self.get_track_len(tracks2_data)

        track_fusion_info_v2i,track_fusion_info_i2v = self.buile_track_fusion_info(tracks1_data,tracks2_data)

        tracks1_frame_id = np.sort(np.unique(tracks1_data[:,0]))
        for track1_frame_id in tracks1_frame_id:
            track1_frame_id = int(track1_frame_id)
            # track1_frame_id_str = str(int(track1_frame_id)).zfill(6)
            if track1_frame_id in self.veh2inf_frame_id:
                #coop pairs
                tracks1 = tracks1_data[tracks1_data[:,0] == track1_frame_id]
    
                track2_frame_id = self.veh2inf_frame_id[track1_frame_id]
                tracks2 = tracks2_data[tracks2_data[:,0] == track2_frame_id]
                
                if len(tracks1) > 0 and len(tracks2):
                    #get fusion info              
                    self.get_tracks_fusion_info_for_pairs(tracks1,tracks2,track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v)
                elif len(tracks1) > 0:
                    print('only veh tracks,no inf tracks! veh frame is: ',track1_frame_id)

                elif len(tracks2) > 0:
                    print('only inf tracks,no veh tracks! inf frame is: ',track2_frame_id)                
                else:
                    continue
            
        #correct fusion info
        cannot_fusion_v2i = self.correct_tracks_fusion_info(track_relation_veh,track_relation_inf,track_fusion_info_v2i,track_fusion_info_i2v,tracks_len_veh,tracks_len_inf)

        return cannot_fusion_v2i
    
    def fix_type(self):
        cols = len(self.new_tracks_fusion[0])
        new_tracks_fusion = np.zeros(shape=(0, cols),dtype=self.new_tracks_fusion.dtype) - 1
        new_tracks_fusion_tocken = []

        tracks_id = np.unique(self.new_tracks_fusion[:,2])
        for track_id in tracks_id:
            cur_track = self.new_tracks_fusion[self.new_tracks_fusion[:,2] == track_id]
            cur_track_tocken = self.new_tracks_fusion_tocken[self.new_tracks_fusion[:,2] == track_id]

            #need del
            if track_id == 24:
                cur_track_from_inf = cur_track[np.where((cur_track[:,24] == 2))]
                track_id_inf = set(cur_track_from_inf[:,30])
                cur_track_from_coop = cur_track[np.where((cur_track[:,24] == 3))]
                track_id_coop = set(cur_track_from_coop[:,30])
                track_id = track_id

            cur_track_from_veh = cur_track[cur_track[:,24] == 1]
            cur_track_from_veh_tocken = cur_track_tocken[cur_track[:,24] == 1]
            cur_track_from_veh_tocken_list = cur_track_from_veh_tocken.tolist()

            cur_track_from_others = cur_track[np.where((cur_track[:,24] == 2) | (cur_track[:,24] == 3))]
            cur_track_from_others_tocken = cur_track_tocken[np.where((cur_track[:,24] == 2) | (cur_track[:,24] == 3))]
            cur_track_from_others_tocken_list = cur_track_from_others_tocken.tolist()

            if len(cur_track_from_others) > 0:                
                type_set_others = list(set(cur_track_from_others[:,1]))
                max_type_nums = 0
                max_type = -1

                if len(type_set_others) > 1:
                    for type in type_set_others:
                        cur_type_nums = len(cur_track_from_others[cur_track_from_others[:,1] == type])
                        if cur_type_nums > max_type_nums:
                            max_type_nums = cur_type_nums
                            max_type = type 
                    
                    for tmp_track in cur_track_from_others:
                        tmp_track[1] = max_type
                else:
                    max_type_nums = len(type_set_others)
                    max_type = type_set_others[0]                

                if len(cur_track_from_veh) > 0:
                    type_set_veh = list(set(cur_track_from_veh[:,1])) 
                    if not(len(type_set_veh) == 1 and type_set_veh[0] == max_type):                
                        for tmp_track in cur_track_from_veh:
                            tmp_track[1] = max_type

                new_tracks_fusion = np.vstack([new_tracks_fusion,cur_track_from_others])
                new_tracks_fusion_tocken = new_tracks_fusion_tocken + cur_track_from_others_tocken_list
            
            if len(cur_track_from_veh) > 0:
                new_tracks_fusion = np.vstack([new_tracks_fusion,cur_track_from_veh])
                new_tracks_fusion_tocken = new_tracks_fusion_tocken + cur_track_from_veh_tocken_list

        self.new_tracks_fusion = new_tracks_fusion
        self.new_tracks_fusion_tocken = new_tracks_fusion_tocken



    


