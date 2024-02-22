#!/bin/bash
set -e

# CUDA_VISIBLE_DEVICES=0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

SPLIT="mmbench_dev_20230712"

CHUNKS=${#GPULIST[@]}

echo chunks:$CHUNKS

RAW_CKPT=$1
CKPT=$(echo $RAW_CKPT | cut -d "/" -f 2)

echo ckpt:$CKPT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m dualfocus.eval.model_vqa_mmbench \
        --model-path $RAW_CKPT \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=./playground/data/eval/mmbench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --pred-file $output_file \
    --save-file ./playground/data/eval/mmbench/answers_upload/${CKPT}.xlsx \

python scripts/excel_test.py ./playground/data/eval/mmbench/answers_upload/${CKPT}.xlsx