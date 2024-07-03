#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:1 bash scripts/sharegpt4v/eval/llavabench.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0 bash scripts/sharegpt4v/eval/llavabench.sh

CKPT_NAME=${1-'share4v-7b'}
CKPT_PATH=${2-'checkpoints'}

python -m share4v.eval.model_vqa \
    --model-path ${CKPT_PATH}/${CKPT_NAME} \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/resources/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python share4v/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/resources/context.jsonl \
    --rule share4v/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/resources/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT_NAME}-eval1.jsonl

python share4v/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT_NAME}-eval1.jsonl
