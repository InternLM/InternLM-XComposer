# Change the model name to the model you want to evaluate

EVAL_MODEL="IXC2d5_OL"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
TOTAL_CHUNKS=${#GPULIST[@]}

echo "Total GPUs: ${TOTAL_CHUNKS}"
echo "Running on GPU list: ${gpu_list}"

TASK="real"
BENCHMARK="Streaming"
DATA_FILE="benchmarks/streamingbench/src/data/questions_${TASK}.json"


for IDX in $(seq 0 $((TOTAL_CHUNKS-1))); do
    OUTPUT_FILE="./outputs/streamingbench/${TASK}_output_IXC2d5_OL_${IDX}.json"
    CUDA_VISIBLE_DEVICES=${IDX} python -m benchmarks.streamingbench.src.eval --model_name $EVAL_MODEL --video_folder $1 \
                      --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE \
                      --num_chunks $TOTAL_CHUNKS --chunk_id ${IDX} &
done
wait

cd data
python -m benchmarks.streamingbench.src.data.count --model $EVAL_MODEL --task $TASK --src "./outputs/streamingbench" --num_chunks $TOTAL_CHUNKS