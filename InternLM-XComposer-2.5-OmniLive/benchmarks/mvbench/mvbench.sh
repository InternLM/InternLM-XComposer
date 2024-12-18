#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo ${CHUNKS}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m benchmarks.mvbench.mvbench \
        --video-folder $1 \
        --save-folder outputs/mvbench \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait
python -m eval_mvbench --folder outputs/mvbench --num_chunks $CHUNKS