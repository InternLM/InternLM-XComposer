#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/mme.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/mme.sh

CKPT_NAME=${1-'share4v-7b'}
CKPT_PATH=${2-'checkpoints'}

python -m share4v.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/MME/share4v_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release \
    --answers-file ./playground/data/eval/MME/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT_NAME}

cd eval_tool

python calculation.py --results_dir answers/${CKPT_NAME}
