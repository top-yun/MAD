GPU_DEVICE=0,1,2,3,4,5,6,7
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
MODAL_TYPE=av
DATASET_PATH=/mnt/T5/dataset/AVHBench
OUTPUT_PATH=./results/evaluation_results_accelerate.json
DEVICE_MAP=auto
BATCH_SIZE=1
NUM_WORKERS=2
SAVE_INTERVAL=100
LOAD_8BIT=False
LOAD_4BIT=False
VERBOSE=False

CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --num_processes=$n_gpu --main_process_port 29501  \
eval_batch.py \
--modal-type $MODAL_TYPE \
