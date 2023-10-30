# Usually you only need to customize these variables #
GPU_ID=$1                                               #
ROOT_DIR=$2                                      #
CKPT=$3                                                    #    

CUDA_VISIBLE_DEVICES=${GPU_ID}
python eval.py \
    --root ${ROOT_DIR} \
    --ckpt_path ${CKPT}