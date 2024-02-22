#!/bin/bash
set -e

SPLIT="llava_gqa_testdev_balanced_mcq"
GQADIR="./playground/data/eval/gqa/data"

RAW_CKPT=$1
CKPT=$(echo $RAW_CKPT | cut -d "/" -f 2)

PARTITION=${2:-llm4}
JOB_NAME=${3:-llava_test}
QUOTA_TYPE=${4:-reserved}
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
CHUNKS=$GPUS

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=$GPUS \
    --ntasks-per-node=$GPUS_PER_NODE \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    python -u -m dualfocus.eval.model_vqa_gqa \
        --model-path $RAW_CKPT \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx -1 \
        --temperature 0 \
        --conv-mode vicuna_v1


output_file=./playground/data/eval/gqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
