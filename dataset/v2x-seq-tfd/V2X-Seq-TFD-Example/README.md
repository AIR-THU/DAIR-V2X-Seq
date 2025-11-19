# Trajectory Forecasting Dataset
This dataset is designed for vehicle-infrastructure cooperative trajectory forecasting tasks. For the recent information and introduction, please visit https://github.com/AIR-THU/DAIR-V2X-Seq/tree/main/dataset/v2x-seq-tfd.

## Dataset Structure
```
V2X-Seq-TFD/V2X-Seq-TFD-Example             # (An example sequence of) trajectory forecasting dataset 
    └── single-infrastructure               # Trajectory forecasting dataset with infrastructure-only view
        └── trajectories			      
            ├── {scene_id}.csv                    
        └── traffic-light                        
            ├── {scene_id}.csv                
    └── single-vehicle                      # Trajectory forecasting dataset with vehicle-only view
        └── trajectories			      
            ├── {scene_id}.csv  	                
    └── cooperative-vehicle-infrastructure  # Trajectory forecasting dataset with vehicle-infrastructure cooperative view
        ├── infrastructure-trajectories     # Trajectory forecasting dataset with infrastructure view
            ├── {scene_id}.csv    
        ├── vehicle-trajectories            # Trajectory forecasting dataset with vehicle view		      
            ├── {scene_id}.csv          
        ├── cooperative-trajectories        # Trajectory forecasting dataset with vehicle-infrastructure cooperative view		      
            ├── {scene_id}.csv                
        ├── traffic-light                        
            ├── {scene_id}.csv
    └── maps                                # HD Maps for each intersection
        └── hdmap{intersection_id}.json
```

## Dataset Introduction
In this part, we provide detailed information for each file in trajectory forecasting dataset.

### Trajectory with Single View
Here we provide the detailed information for the trajectory file 
 - single-infrastructure/trajectories/{scene_id}.csv
 - single-vehicle/trajectories/{scene_id}.csv
 - cooperative-vehicle-infrastructure/infrastructure-trajectories/{scene_id}.csv
 - cooperative-vehicle-infrastructure/vehicle-trajectories/{scene_id}.csv

```csv
{
    "city",
    "timestamp",
    "id",
    "type",         # Object class in [Vehicle, Bicycle, Pedestrain]
    "sub_type",     # Object subclass in [Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist, Motorcyclist]
    "tag",          # Flag that is target agent or not
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "theta",
    "v_x",
    "v_y",
    "intersect_id"
}
```

### Trajectory with Cooperative View
Here we provide the detailed information for the trajectory file
- cooperative-vehicle-infrastructure/cooperative-trajectories/{scene_id}.csv

``` csv
{
    "city",
    "timestamp",
    "id",
    "type",         # Object class in [Vehicle, Bicycle, Pedestrain]
    "sub_type",     # Object subclass in [Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist, Motorcyclist]
    "tag",          # Flag that is target agent or not
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "theta",
    "v_x",
    "v_y",
    "intersect_id",
    "vic_tag",        # Flag that the trajectory is complete(car) or incomplete(vic) in observation horizon.
    "from_side",      # Observation view from [infrastructure view, vehicle view]
    "car_side_id",
    "road_side_id"
}
```

### Traffic Light
In this part, we introduce the traffic_light/{scene_id}.csv.

```csv
{
    "city",
    "timestamp",
    "x",
    "y",
    "direction",    # Traffic light orientation
    "lane_id",      # lane_id controlled by the corresponding traffic light signal
    "color_1",      # The color of the first signal in traffic light
    "remain_1",     # The remain time of the first signal in traffic light 
    "color_2",      # The color of the second signal in traffic light
    "remain_2",     # The remain time of the second signal in traffic light 
    "color_3",      # The color of the third signal in traffic light
    "remain_3",     # The remain time of the third signal in traffic light 
    "intersect_id"
}
```

### HD Maps
In this part, we introduce hdmap{intersection_id}.json.

```csv
{
    "LANE": {
        "lane_id": {
        "has_traffic_control"
        "lane_type"
        "turn_direction"
        "is_intersection"
        "l_neighbor_id"
        "r_neighbor_id"
        "predecessors"
        "successors"
        "centerline"
        }
    },
    "STOPLINE": {
        "stoplane_id": {
        "centerline"
        }
    },
    "CROSSWALK": {
        "crosswalk_id": {
        "polygon"
        }
    }
}
```

## Reference
```
@inproceedings{v2x-seq,
  title={V2X-Seq: A large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting},
  author={Yu, Haibao and Yang, Wenxian and Ruan, Hongzhi and Yang, Zhenwei and Tang, Yingjuan and Gao, Xu and Hao, Xin and Shi, Yifeng and Pan, Yifeng and Sun, Ning and Song, Juan and Yuan, Jirui and Luo, Ping and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```