GPU_DEVICE=0,1,2,3,4,5,6,7
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
MODAL_TYPE=av
DATASET_PATH=/mnt/T5/dataset/AVHBench
OUTPUT_PATH=./suppl/cmm_mad.json
DEVICE_MAP=auto
BATCH_SIZE=1
NUM_WORKERS=2
SAVE_INTERVAL=100
LOAD_8BIT=False
LOAD_4BIT=False
VERBOSE=False

# for gamma in $(seq 1.5 1.0 2.5)
# do
#     export NCCL_TIMEOUT=3600 
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
#     accelerate launch --num_processes=$n_gpu --main_process_port 29508 \
#     eval_batch_cmm_mad.py \
#     --category over-reliance_unimodal_priors \
#     --output-path $OUTPUT_PATH \
#     --modal-type av \
#     --gamma $gamma \

# done

export NCCL_TIMEOUT=3600 
CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --num_processes=$n_gpu --main_process_port 29508 \
eval_batch_cmm_mad.py \
--category over-reliance_unimodal_priors \
--output-path $OUTPUT_PATH \
--modal-type av \
--gamma 2.5 \
--model-path 'Qwen/Qwen2.5-Omni-3B' \