# Usually you only need to customize these variables #
GPU_ID=$1                                               #
ROOT_DIR=$2                                 #

CUDA_VISIBLE_DEVICES=${GPU_ID}
python train.py \
    --root ${ROOT_DIR}