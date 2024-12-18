#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo ${CHUNKS}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m benchmarks.video_mme.video_mme \
        --video-folder $1 \
        --save-folder outputs/video_mme \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait
python -m eval_video_mme --folder outputs/video_mme --num_chunks $CHUNKS