GPU_DEVICE=0,1,2,3,4,5,6,7
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
MODAL_TYPE=av
DATASET_PATH=/mnt/T5/dataset/AVHBench
OUTPUT_PATH=./suppl/avhbench.json
DEVICE_MAP=auto
BATCH_SIZE=1
NUM_WORKERS=2
SAVE_INTERVAL=100
LOAD_8BIT=False
LOAD_4BIT=False
VERBOSE=False

gamma=2.5

export NCCL_TIMEOUT=3600 
CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --num_processes=$n_gpu --main_process_port 29508 \
eval_batch_mad.py \
--model-path 'Qwen/Qwen2.5-Omni-7B' \
--modal-type $MODAL_TYPE \
--max-new-tokens 1 \
--use-contrast-decode \
--gamma $gamma \
--dataset-path $DATASET_PATH \
--output-path $OUTPUT_PATH \
