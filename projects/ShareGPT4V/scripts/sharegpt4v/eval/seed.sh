#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/seed.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/seed.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_PATH=${CKPT_PATH:-'checkpoints'}
CKPT_NAME=${1-'share4v-7b'}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_loader \
        --model-path ${CKPT_PATH}/${CKPT_NAME} \
        --question-file ./playground/data/eval/seed_bench/share4v-seed-bench-image.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/seed_bench/answers/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/${CKPT_NAME}.jsonl

