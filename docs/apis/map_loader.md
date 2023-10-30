# Usage of Map Loader

You can load and merge 28 HD maps into one map as followings.

1. Download TFD/TFD-Example dataset
    ```shell
    # For example: download TFD-Example into ./dataset/v2x-seq-tfd with path ${DATA_ROOT}
    bash tools/dataset_example_download.sh
    ```

2. Load and Merge HD Maps
    ```shell
    mkdir ${DATA_ROOT}/map_files
    python tools/data_converter/merge_maps.py 
    ```