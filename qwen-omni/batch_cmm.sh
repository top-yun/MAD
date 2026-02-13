GPU_DEVICE=0,1,2,3,4,5,6,7
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
MODAL_TYPE=av
DATASET_PATH=/mnt/T5/dataset/AVHBench
OUTPUT_PATH=./suppl/evaluation_cmm.json
DEVICE_MAP=auto
BATCH_SIZE=1
NUM_WORKERS=2
SAVE_INTERVAL=100
LOAD_8BIT=False
LOAD_4BIT=False
VERBOSE=False

export DECORD_EOF_RETRY_MAX=2048000
export FORCE_QWENVL_VIDEO_READER="torchvision"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --num_processes=$n_gpu --main_process_port 29506  \
    eval_batch_cmm.py \
    --max-new-tokens 1 \
    --category over-reliance_unimodal_priors \
    --output-path $OUTPUT_PATH \
    --modal-type av \
    --model-path 'Qwen/Qwen2.5-Omni-3B' \
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
# accelerate launch --num_processes=$n_gpu --main_process_port 29505  \
# eval_batch_cmm.py \
# --category over-reliance_unimodal_priors \
# --output-path ./results/my_output.json \
# --modal-type v \