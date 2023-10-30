# Get Started for V2X-Seq Dataset and Benchmark

## SPD Dataset Usage

SPD is the successor to the DAIR-V2X-C dataset. While maintaining the overall structure, we have cleaned the dataset, annotated tracking IDs for each object, and provided vector maps.

- Refer to the [SPD dataset structure](../dataset/v2x-seq-spd/README.md).
- Check out the [HD Map usage](../docs/apis/map_loader.md).
- For more dataset usage details, visit the [DAIR-V2X documentation](https://github.com/AIR-THU/DAIR-V2X/blob/main/docs/get_started.md).

## SPD Benchmarks

We offer early fusion, late fusion, and middle fusion benchmarks like FF-Tracking for the VIC3D Tracking task. To learn about training and evaluating these benchmarks, visit the following link:
- [VIC3D Tracking Benchmarks](https://github.com/AIR-THU/DAIR-V2X/tree/main/configs)

## TFD Dataset Usage

The TFD dataset comprises trajectories, vector maps, and traffic light signals.
- Explore the [TFD dataset structure](../dataset/v2x-seq-tfd/README.md).
- Learn about loading trajectories, vector maps, traffic light signals and the visualization of TFD using the [TFD Tutorial](../projects/dataset/dair_v2x_tfd_tutorial.ipynb).

## TFD Benchmarks

We provide various benchmarks, including PP-VIC, for solving Online-VIC Forecasting and Offline-VIC Forecasting tasks. Find basic guidance in the [TFD Benchmark README](../docs/benchmarks/vic-traj-forecasting/README.md). Detailed training and evaluation of Baselines with HiVT and TNT are as follows:

- For training and evaluation of Baselines with HiVT, refer to the [HiVT README](../docs/benchmarks/vic-traj-forecasting/HiVT/README.md).
- For training and evaluation of Baselines with TNT, refer to the [TNT README](../docs/benchmarks/vic-traj-forecasting/TNT/README.md).

### Example of PP-VIC Evaluation with HiVT

Here's how to evaluate PP-VIC with HiVT for solving the Online-VIC Forecasting task using the TFD-Example dataset.

1. Create a conda environment and install dependencies as specified in [HiVT](https://github.com/ZikangZhou/HiVT):
    ```shell
    conda create -n HiVT python=3.8
    conda activate HiVT
    conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
    conda install pytorch-lightning==1.5.2 -c conda-forge
    ```

2. Install the [Argoverse 1 API](https://github.com/argoverse/argoverse-api).

3. Dataset Preparation:
    ```shell
    # Download TFD-Example into ./dataset/v2x-seq-tfd/V2X-Seq-TFD-Example
    bash tools/dataset_example_download.sh

    # Merge Multiple Maps into One Map
    python tools/data_converter/maps_merge.py

    # Preprocess Cooperative-view Trajectories
    python tools/trajectory_fusion/fusion_for_prediction.py
    python tools/data_converter/tfd_argoverse_converter.py
    ```

4. Evaluation:
    ```shell
    # parsers: GPU_ID, DATA_ROOT, CKPT
    # DATA_ROOT=../../dataset/v2x-seq-tfd/V2X-Seq-TFD-Example
    cd projects/HiVT_plugin
    bash tools/hivt_eval.sh 0 ${DATA_ROOT}/cooperative-vehicle-infrastructure/fusion_for_prediction ./checkpoints/online.ckpt
    ```