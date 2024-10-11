#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/qbench.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/qbench.sh

CKPT=${1-"share4v-7b"}
CKPT_DIR=${2-"checkpoints"}
SPLIT="dev"

python -m share4v.eval.model_vqa_qbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_${SPLIT}.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT}_answers.jsonl \
    --conv-mode share4v_v1 \
    --lang en

python playground/data/eval/qbench/format_qbench.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT}_answers.jsonl

python playground/data/eval/qbench/qbench_eval.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_${SPLIT}_${CKPT}_answers.jsonl