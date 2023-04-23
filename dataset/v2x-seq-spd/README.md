# Sequential Perception Dataset
The dataset is for the vehicle-infrastructure cooperative 3D tracking task. Please refer https://github.com/AIR-THU/DAIR-V2X-Seq/dataset/v2x-seq-spd for the latest introduction.

## Dataset Structure
In this part, we provide the detailed structure for the sequential perception dataset.
```
V2X-Seq-SPD/V2X-Seq-SPD-Example         # (An example sequence of) sequential percpetion dataset 
    └──  infrastructure-side            # Infrastructure-side data
        ├── image		        
            ├── {id}.jpg
        ├── velodyne                    
            ├── {id}.pcd               
        ├── calib                     
            ├── camera_intrinsic        # Camera intrinsic parameter       
                ├── {id}.json         
            ├── virtuallidar_to_world   # Extrinsic parameter from virtual lidar to world 
                ├── {id}.json          
            ├── virtuallidar_to_camera  # Extrinsic parameter from virtual lidar to camera
                ├── {id}.json          
        ├── label			
            ├── camera                  # Labeled data in infrastructure virtual LiDAR coordinate system fitting objects in image with image camptured timestamp
                ├── {id}.json
            ├── virtuallidar            # Labeled data in infrastructure virtual LiDAR coordinate system fitting objects in point cloud with point cloud captured timestamp
                ├── {id}.json
        └── data_info.json              # More detailed information for each infrastructure-side frame
    └── vehicle-side                    # Vehicle-side data
        ├── image		        
            ├── {id}.jpg
        ├── velodyne                 
            ├── {id}.pcd               
        ├── calib                     
            ├── camera_intrinsic        # Camera intrinsic parameter   
                ├── {id}.json
            ├── lidar_to_camera         # extrinsic parameter from lidar to camera
                ├── {id}.json
            ├── lidar_to_novatel        # extrinsic parameter from lidar to NovAtel
                ├── {id}.json
            ├── novatel_to_world        # location in the world coordinate
                ├── {id}.json
        ├── label			
            ├── camera                  # Labeled data in vehicle LiDAR coordinate system fitting objects in image with image camptured timestamp
                ├── {id}.json
            ├── lidar                   # Labeled data in vehicle LiDAR coordinate system fitting objects in point cloud with point cloud captured timestamp
                ├── {id}.json
        └── data_info.json              # More detailed information for each vehicle-side frame
    └── cooperative                     # Coopetative-view files
        ├── label                       # Vehicle-infrastructure cooperative (VIC) annotation files. Labeled data in vehicle LiDAR coordinate system with the vehicle point cloud timestamp
            ├── {id}.json                
        └── data_info.json              # More detailed information for vehicle-infrastructure cooperative frame pair
    └── maps                            # HD Maps for each intersection
```

---

## Introduction of data-info.json
In this part, we introduce the content in the three files: infrastructure-side/data_info.json, vehicle-side/data_info.json, and cooperative/data_info.json.

### infrastructure-side/data_info.json

```json
{
  "image_path",
  "pointcloud_path",
  "calib_camera_intrinsic_path",
  "calib_virtuallidar_to_camera_path",
  "calib_virtuallidar_to_world_path",
  "label_camera_std_path",
  "label_lidar_std_path",
  "intersection_loc",
  "camera_ip",
  "camera_id",
  "lidar_id",
  "image_timestamp", # timestamp of image captured
  "pointcloud_timestamp", # timestamp of point cloud captured
  "frame_id", 
  "valid_frames_splits", 
  "num_frames", 
  "sequence_id",
}
```

### vehicle-side/data_info.json

```json
{
  "image_path",
  "pointcloud_path",
  "calib_camera_intrinsic_path",
  "calib_lidar_to_camera_path", 
  "calib_lidar_to_novatel_path", 
  "calib_novatel_to_world_path",
  "label_camera_std_path",
  "label_lidar_std_path",
  "intersection_loc",
  "image_timestamp",  # timestamp of image captured
  "pointcloud_timestamp", # timestamp of point cloud captured
  "frame_id", 
  "start_frame_id",
  "end_frame_id",
  "num_frames", 
  "sequence_id",
}
```

#### cooperative/data_info.json

```json
{
  "vehicle_frame",
  "infrastructure_frame",
  "vehicle_sequence",
  "infrastructure_sequence",
  "system_error_offset",
}
```

---

## Introduction of Annotation Files
In this part, we provide the detailed information to explain the annotation for each frame, including infrastructure-side frame, vehicle-side frame, and cooperative frame.

### Tracking Annotation for Infrastructure and Vehicle frames
Here we explain the contents in tracking annotation for infrastructure-side frame and vehicle-side frame.

```json
{
  "token": token,
  "type": type, 
  "track_id": track_id,
  "truncated_state": truncated_state,  
  "occluded_state": occluded_state,
  "alpha": alpha,
  "2d_box": {                          
    "xmin": xmin, 
    "ymin": ymin, 
    "xmax": xmax, 
    "ymax": ymax
  }, 
  "3d_dimensions": {                  
    "h": height, 
    "w": width, 
    "l": length
  }, 
  "3d_location": {               
    "x": x, 
    "y": y, 
    "z": z
  }, 
  "rotation": rotation              
}
```

**Comment**

- 10 object classes, including: Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist, Motorcyclist, Barrowlist, and TrafficCone.

- No tracking annotation for TrafficCone class


---
## Cooperative Annotation File

Here we explain the contents in tracking annotation file for cooperative frame.

```json
{
  "token": token,
  "type": type, 
  "track_id": track_id,
  "truncated_state": truncated_state,  
  "occluded_state": occluded_state,
  "alpha": alpha,
  "2d_box": {                          
    "xmin": xmin, 
    "ymin": ymin, 
    "xmax": xmax, 
    "ymax": ymax
  }, 
  "3d_dimensions": {                  
    "h": height, 
    "w": width, 
    "l": length
  }, 
  "3d_location": {               
    "x": x, 
    "y": y, 
    "z": z
  }, 
  "rotation": rotation,
  "from_side": from_side,
  "veh_pointcloud_timestamp": veh_pointcloud_timestamp,
  "inf_pointcloud_timestamp": inf_pointcloud_timestamp,
  "veh_frame_id": veh_frame_id,
  "inf_frame_id": inf_frame_id,
  "veh_track_id": veh_track_id,
  "inf_track_id": inf_track_id,
  "veh_token": veh_token,
  "inf_token": inf_token
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

```
@inproceedings{dair-v2x,
  title={Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection},
  author={Yu, Haibao and Luo, Yizhen and Shu, Mao and Huo, Yiyi and Yang, Zebang and Shi, Yifeng and Guo, Zhenglong and Li, Hanyu and Hu, Xing and Yuan, Jirui and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21361--21370},
  year={2022}
}
```