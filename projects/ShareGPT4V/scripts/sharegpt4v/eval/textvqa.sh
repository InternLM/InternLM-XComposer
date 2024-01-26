#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/share4v/eval/textvqa.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/share4v/eval/textvqa.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT=${1-'share4v-7b'}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_loader \
        --model-path checkpoints/${CKPT} \
        --question-file ./playground/data/eval/textvqa/share4v_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/textvqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m share4v.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/merge.jsonl
