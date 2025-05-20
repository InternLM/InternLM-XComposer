#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT=29501
export CPUS_PER_TASK=12

export NNODES=${MLP_WORKER_NUM:-1}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-1}
export NODE_RANK=${MLP_ROLE_INDEX:-0}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=${MLP_WORKER_0_PORT:-29501}

export MODEL="path/to/internlm-xcomposer2d5-7b-reward/"
export DATA="data.txt"

# SRUN_ARGS=${SRUN_ARGS:-""}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "run command: torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT finetune.py"

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --lora_r 512 \
    --hd_num 9 \
    --output_dir output/ixc_reward_lora \
    --num_train_epochs 1 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 8192 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True
