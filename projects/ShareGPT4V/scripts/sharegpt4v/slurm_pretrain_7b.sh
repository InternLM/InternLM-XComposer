#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=2
export MASTER_PORT=29502
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=data/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json
# You should download the pretrained projector from LLaVA-v1.5
export CKPT_PATH=pretrained/projector/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin 
export SAVE_PATH=share4v-7b_pretrained_pt-1.2m_ft-vit-l12-mlp-llm-lr-2e-5
export TUNE_ENTIRE_MODEL=true
export TUNE_VIT_FROM=12
export BASE_LR=2e-5
export GRADIENT_ACCU_STEPS=1

SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Your Partion \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA} \
    ${SRUN_ARGS} \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} share4v/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${CKPT_PATH} \
    --mm_projector_type mlp2x_gelu \
    --tune_entire_model ${TUNE_ENTIRE_MODEL} \
    --tune_vit_from_layer ${TUNE_VIT_FROM} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb tensorboard \
    --run_name ${SAVE_PATH}'