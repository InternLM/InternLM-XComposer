#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/mmbench_en.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/mmbench_en.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT=${1-"share4v-7b"}
CKPT_DIR=${2-"checkpoints"}
SPLIT="mmbench_dev_20230712"
LANG="en"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_mmbench \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang $LANG \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT
mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT} \
    --experiment merge
