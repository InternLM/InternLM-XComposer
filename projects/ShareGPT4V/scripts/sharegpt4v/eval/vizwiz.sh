#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/vizwiz.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/vizwiz.sh

CKPT_NAME=${1-'share4v-7b'}
CKPT_PATH=${2-'checkpoints'}

python -m share4v.eval.model_vqa_loader \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/vizwiz/share4v_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/share4v_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT_NAME}.json
