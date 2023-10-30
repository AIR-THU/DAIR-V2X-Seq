# Benchmark with HiVT

## Environment Preparation
1. Create a conda environment and install dependencies as specified in [HiVT](https://github.com/ZikangZhou/HiVT):
    ```shell
    conda create -n HiVT python=3.8
    conda activate HiVT
    conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
    conda install pytorch-lightning==1.5.2 -c conda-forge
    ```

2. Install the [Argoverse 1 API](https://github.com/argoverse/argoverse-api).

## Data Preparation
1. Download TFD/TFD-Example into ./dataset/v2x-seq-tfd with ${DATA_ROOT}
   ```shell
   # export DATA_ROOT = 'DAIR-V2X-Seq/dataset/v2x-seq-tfd/V2X-Seq-TFD-Example'
   # Download TFD-Example into ./dataset/v2x-seq-tfd
   bash tools/dataset_example_download.sh
   ```

2. Merge Multiple Maps into One Map
   ```shell
   # Merge Multiple Maps into One Map
   python tools/data_converter/maps_merge.py --data_root ${DATA_ROOT}
   ```

3. Preprocess Cooperative-view Trajectories
   ```shell
   # Preprocess Cooperative-view Trajectories --split train/val
   python tools/trajectory_fusion/fusion_for_prediction.py --data_root ${DATA_ROOT} --split train
   python tools/data_converter/tfd_converter.py --data_root ${DATA_ROOT} --split train
   ```

## Training

1. Train the PP-VIC with HiVT for Online task
   ```shell
   cd projects/HiVT_plugin
   bash hivt_train.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction
   ```

2. Train the HiVT for Offline task
   ```shell
   cd projects/HiVT_plugin

   # Pretrained with infrastructure trajectories
   bash hivt_pretrain.sh ${GPU_ID} ${DATA_ROOT}/single-infrastructure/trajectories

   # Finetune with vehicle trajectories and pretrained model
   bash hivt_finetune.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-trajectories ${CKPT}
   ```


## Evaluation

1. Evaluate the Trained PP-VIC with HiVT for Online task. We have provided the trained [online.ckpt](../../../../projects/HiVT_plugin/checkpoints/online.ckpt).
   ```shell
   cd projects/HiVT_plugin
   bash hivt_eval.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction ${CKPT}
   ```

2. Evaluate the HiVT for Offline task. We have provided the trained  [offline.ckpt](../../../../projects/HiVT_plugin/checkpoints/offline.ckpt).
   ```shell
   cd projects/HiVT_plugin
   bash hivt_eval.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-trajectories ${CKPT}
   ```