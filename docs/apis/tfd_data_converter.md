# Usage of Fusing Cross-view Trajectories

You can fuse infrastructure-view and vehicle-view trajectories as followings.

1. Download TFD/TFD-Example dataset
    ```shell
    # For example: download TFD-Example into ./dataset/v2x-seq-tfd with path ${DATA_ROOT}
    bash tools/dataset_example_download.sh
    ```

2. Fusing two-view trajectories
    ```shell
   # Preprocess Cooperative-view Trajectories
   python tools/trajectory_fusion/fusion_for_prediction.py --data-root ${DATA_ROOT}
   python tools/data_converter/tfd_argoverse_converter.py --data-root ${DATA_ROOT}
    ```