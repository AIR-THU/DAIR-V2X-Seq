import numpy as np 
import copy

def cal_box_iou_wh(bbox_1,bbox_2):
    '''
        cal two boxes iou matric        
    '''
    overlap_left = max(bbox_1[0],bbox_2[0])
    overlap_right = min(bbox_1[2]+bbox_1[0],bbox_2[2]+bbox_2[0])
    overlap_top = max(bbox_1[1],bbox_2[1])
    overlap_bottom = min(bbox_1[3]+bbox_1[1],bbox_2[3]+bbox_2[1])
    overlap_w = max(0,overlap_right - overlap_left)
    overlap_h = max(0,overlap_bottom - overlap_top)
    overlap_area = overlap_w * overlap_h
    area_1 = (bbox_1[2]) * (bbox_1[3])
    area_2 = (bbox_2[2]) * (bbox_2[3])

    iou = overlap_area / (area_1+area_2-overlap_area)
    
    return iou

def cal_box_iou_centerpts(centerpoints1, w1, h1,centerpoints2, w2, h2):
    ct_x1 = centerpoints1[0]
    ct_y1 = centerpoints1[1]
    ct_x2 = centerpoints2[0]
    ct_y2 = centerpoints2[1]

    box1 = [ct_x1 - w1 / 2, ct_y1 - h1 / 2, w1, h1]
    box2 = [ct_x2 - w2 / 2, ct_y2 - h2 / 2, w2, h2]
    iou = cal_box_iou_wh(box1,box2)

    return iou

def cal_box_iou(bbox_1,bbox_2):
    '''
        cal two boxes iou matric
    '''
    overlap_left = max(bbox_1[0],bbox_2[0])
    overlap_right = min(bbox_1[2],bbox_2[2])
    overlap_top = max(bbox_1[1],bbox_2[1])
    overlap_bottom = min(bbox_1[3],bbox_2[3])
    overlap_w = max(0,overlap_right - overlap_left)
    overlap_h = max(0,overlap_bottom - overlap_top)
    overlap_area = overlap_w * overlap_h
    area_1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
    area_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])

    iou = overlap_area / (area_1+area_2-overlap_area)
    
    return iou

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# greedy matching algorithm *****************************************************
def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)

def type_filt(type1,type2):
    type_diff = True
    if type1 == type2 or (type1 in [2,4,5,6] and type2 in [2,4,5,6]) or (type1 in [1,3,7,8,9] and type2 in [1,3,7,8,9]): 
        type_diff = False
    
    return type_diff

def diff_filt(center1, center2, size, type1, type2):
    diff = np.abs(center1 - center2) / size

    type_diff = type_filt(type1,type2)

    return diff[0] > 1 or diff[1] > 1 or diff[2] > 1 or type_diff

def match(frame1, frame2, lidar_pos=None):
    frame1_nums = len(frame1)
    frame2_nums = len(frame2)

    cost_matrix = np.zeros((frame1_nums, frame2_nums))
    for i in range(frame1_nums):
        for j in range(frame2_nums):
            veh_lwh = [frame1[i][12],frame1[i][11],frame1[i][10]]

            center1 = np.array([frame1[i][17],frame1[i][18],frame1[i][19]])
            center2 = np.array([frame2[j][17],frame2[j][18],frame2[j][19]])                

            cost_matrix[i][j] = np.sum((center1 - center2) ** 2) ** 0.5
            if diff_filt(center1,center2,veh_lwh,frame1[i][1],frame2[j][1]):
                cost_matrix[i][j] = 1e6
                
    # print(cost_matrix, linear_sum_assignment(cost_matrix))
    matched_ids_temp = linear_assignment(cost_matrix)
    index1, index2 = matched_ids_temp[:,0], matched_ids_temp[:,1]

    cost = []
    accepted = []
    for i in range(len(index1)):
        if cost_matrix[index1[i]][index2[i]] < 1e5:
            accepted.append(i)
            cost.append(cost_matrix[index1[i]][index2[i]])

    return index1[accepted], index2[accepted], cost


def cal_matched_ids(tracks1,tracks2,iou_threshold,hungarian):
    matched_ids = np.empty(shape=(0,2),dtype=np.int64)

    tracks1_len = len(tracks1)
    tracks2_len = len(tracks2)
    if tracks1_len > 0 and tracks2_len > 0:
        # initialize distance matrix
        iou_matrix = np.ndarray(shape=(tracks1_len, tracks2_len))
        for index1, cur_box1 in enumerate(tracks1):
            for index2, cur_box2 in enumerate(tracks2):
                if cur_box1[1] == cur_box2[1]: # the same class
                    iou_matrix[index1][index2] = cal_box_iou_centerpts(cur_box1[13:15], cur_box1[12], cur_box1[11],cur_box2[13:15], cur_box2[12], cur_box2[11])
                else:
                    iou_matrix[index1][index2] = 0

        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_ids = np.stack(np.where(a), axis=1)
        else:
            track_distances = -iou_matrix
            if hungarian:
                # use hungarian algorithm to find best matching tracklets
                matched_ids_temp = linear_assignment(copy.deepcopy(track_distances))
            else:
                # use greedy algorithm
                matched_ids_temp = greedy_assignment(copy.deepcopy(track_distances))                    
            matched_ids = []
            for m in matched_ids_temp:
                if(iou_matrix[m[0], m[1]] > iou_threshold): 
                    matched_ids.append(m) 
        matched_ids = np.array(matched_ids).reshape(-1, 2) 

    # return matched_ids
    return matched_ids[:,0], matched_ids[:,1]