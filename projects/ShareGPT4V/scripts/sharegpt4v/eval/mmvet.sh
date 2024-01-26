#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/mmvet.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/mmvet.sh

CKPT_NAME=${1-'share4v-7b'}
CKPT_PATH=${2-'checkpoints'}

python -m share4v.eval.model_vqa \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/mm-vet/share4v-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/resources/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT_NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT_NAME}.json

