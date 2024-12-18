#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo ${CHUNKS}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m benchmarks.mlvu.mlvu \
        --video-folder $1 \
        --save-folder outputs/mlvu \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait
python -m eval_mlvu --folder outputs/mlvu --num_chunks $CHUNKS