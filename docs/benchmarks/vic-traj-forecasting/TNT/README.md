# Benchmark with TNT

## Environment Preparation
1. Create a conda environment and install dependencies as specified in [TNT](https://github.com/Henry1iu/TNT-Trajectory-Prediction).

2. Install the apex module.
   ```shell
   git clone https://github.com/ptrblck/apex.git
   cd apex
   git checkout apex_no_distributed
   pip install -v --no-cache-dir ./
   ```

3. Install the [Argoverse 1 API](https://github.com/argoverse/argoverse-api).

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

4. DATA Preprocess
   ```shell
   # Preprocess Raw Data
   cd projects/TNT_plugin
   python core/util/preprocessor/tfd_preprocess.py --root ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction

   python core/util/preprocessor/tfd_preprocess.py --root ${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-trajectories

   python core/util/preprocessor/tfd_preprocess.py --root ${DATA_ROOT}/single-vehicle/trajectories

   python core/util/preprocessor/tfd_preprocess.py --root ${DATA_ROOT}/single-infrastructure/trajectories
   ```

## Training

1. Train the PP-VIC with TNT for Online task
   ```shell
   cd projects/TNT_plugin
   bash tnt_train.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data
   ```

2. Train the TNT for Offline task
   ```shell
   cd projects/TNT_plugin

   # Pretrained with infrastructure trajectories
   bash tnt_pretrain.sh ${GPU_ID} ${DATA_ROOT}/single-infrastructure/trajectories/interm_data

   # Finetune with vehicle trajectories and pretrained model
   bash tnt_finetune.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-trajectories/interm_data ${PTH}
   ```


## Evaluation

1. Evaluate the Trained PP-VIC with TNT for Online task. We have provided the trained [online.pth](../../../../projects/TNT_plugin/checkpoints/online.pth).
   ```shell
   cd projects/TNT_plugin
   bash tnt_eval.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data ${PTH}
   ```

2. Evaluate the TNT for Offline task. We have provided the trained  [offline.pth](../../../../projects/TNT_plugin/checkpoints/offline.pth).
   ```shell
   cd projects/TNT_plugin
   bash tnt_eval.sh ${GPU_ID} ${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-trajectories/interm_data ${PTH}
   ```