#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/sqa.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/sqa.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_NAME=${1-'shar4v-7b'}
CKPT_PATH=${2-'checkpoints'}

for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_science \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/scienceqa/share4v_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/scienceqa/answers/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python share4v/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/merge.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT_NAME}/result.json
